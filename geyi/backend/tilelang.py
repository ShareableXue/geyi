"""Deterministic TileLang-shaped backend.

The generated source is intentionally local and dependency-free. It preserves
the backend/project boundary while the real TileLang/CANN toolchain is absent.
"""

from __future__ import annotations

import importlib.util
import json
import keyword
import py_compile
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from geyi.backend.model import CompiledArtifact, GeneratedProject, sha256_file
from geyi.contract.model import SemanticContract
from geyi.planner.plan import DETERMINISTIC_BACKEND, TranslationPlan


SUPPORTED_TEMPLATES = {
    "tilelang.elementwise_binary_1d",
    "tilelang.elementwise_unary_1d",
    "tilelang.copy_cast_1d",
    "tilelang.transpose2d",
    "tilelang.row_reduce_sum",
}


class BackendError(RuntimeError):
    pass


class TileLangBackend:
    name = DETERMINISTIC_BACKEND

    def can_generate(self, plan: TranslationPlan) -> bool:
        return plan.backend == self.name and plan.template in SUPPORTED_TEMPLATES

    def generate(self, contract: SemanticContract, plan: TranslationPlan, root: Path) -> GeneratedProject:
        if not self.can_generate(plan):
            raise BackendError("TileLang backend cannot generate this plan")

        root.mkdir(parents=True, exist_ok=True)
        (root / "tests").mkdir(parents=True, exist_ok=True)

        stem = source_stem(plan)
        kernel_source = root / ("%s.py" % stem)
        build_file = root / "build.json"
        test_file = root / "tests" / "golden_cases.json"
        metadata_file = root / "metadata.json"

        kernel_source.write_text(render_kernel(plan), encoding="utf-8")
        build_file.write_text(json_dumps({"compiler": "py_compile", "artifact": "%s.pyc" % stem}), encoding="utf-8")
        test_file.write_text(json_dumps({"coverage_cases": plan.parameters["coverage_cases"]}), encoding="utf-8")

        project = GeneratedProject(
            root=str(root),
            backend=self.name,
            kernel_sources=[kernel_source.name],
            host_sources=[],
            bindings=[],
            build_files=[build_file.name],
            tests=[str(Path("tests") / test_file.name)],
            metadata={
                "contract_hash": contract.contract_hash,
                "strategy": plan.strategy,
                "template": plan.template,
                "operation": plan.parameters["operation"],
                "execution_mode": "local_golden_simulator",
                "npu_execution_claimed": False,
            },
            assumptions=list(plan.required_assumptions),
        )
        metadata_file.write_text(
            json_dumps(
                {
                    "generated_project": project.to_dict(),
                    "plan": plan.to_dict(),
                }
            ),
            encoding="utf-8",
        )
        return project

    def compile(self, project: GeneratedProject, build_root: Path) -> CompiledArtifact:
        build_root.mkdir(parents=True, exist_ok=True)
        if not project.kernel_sources:
            raise BackendError("generated project has no kernel source")

        source = Path(project.root) / project.kernel_sources[0]
        if not source.exists():
            raise BackendError("generated source does not exist: %s" % source)

        artifact = build_root / ("%s.pyc" % source.stem)
        py_compile.compile(
            str(source),
            cfile=str(artifact),
            dfile=source.name,
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        )

        compiled = CompiledArtifact(
            path=str(artifact),
            artifact_hash=sha256_file(artifact),
            backend=self.name,
            compiler="py_compile",
            reused=False,
            metadata={
                "source": str(source),
                "source_hash": sha256_file(source),
                "execution_mode": "local_golden_simulator",
                "python_version": sys.version.split()[0],
                "python_cache_tag": sys.implementation.cache_tag,
                "python_magic_number": importlib.util.MAGIC_NUMBER.hex(),
            },
        )
        (build_root / "artifact_metadata.json").write_text(json_dumps(compiled.to_dict()), encoding="utf-8")
        return compiled

    def load_project(self, root: Path) -> GeneratedProject:
        metadata_file = root / "metadata.json"
        if not metadata_file.exists():
            raise BackendError("generated metadata missing: %s" % metadata_file)
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        return GeneratedProject.from_dict(payload["generated_project"], root=str(root))

    def load_artifact(self, build_root: Path, reused: bool) -> CompiledArtifact:
        metadata_file = build_root / "artifact_metadata.json"
        if not metadata_file.exists():
            raise BackendError("artifact metadata missing: %s" % metadata_file)
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        artifact_path = build_root / Path(payload["path"]).name
        return CompiledArtifact.from_dict(payload, path=str(artifact_path), reused=reused)


def render_kernel(plan: TranslationPlan) -> str:
    operation = str(plan.parameters["operation"])
    if plan.template == "tilelang.elementwise_binary_1d":
        return render_elementwise_binary(plan, operation)
    if plan.template == "tilelang.elementwise_unary_1d":
        return render_elementwise_unary(plan, operation)
    if plan.template == "tilelang.copy_cast_1d":
        return render_copy_cast(plan, operation)
    if plan.template == "tilelang.transpose2d":
        return render_transpose2d(plan)
    if plan.template == "tilelang.row_reduce_sum":
        return render_row_reduce_sum(plan)
    raise BackendError("unsupported template: %s" % plan.template)


def render_elementwise_binary(plan: TranslationPlan, operation: str) -> str:
    inputs = safe_names(plan.parameters["inputs"])
    output = safe_name(plan.parameters["output"])
    if len(inputs) != 2:
        raise BackendError("binary elementwise requires two inputs")
    symbol = {"add": "+", "mul": "*"}.get(operation)
    if symbol is None:
        raise BackendError("unsupported binary operation: %s" % operation)
    return header(plan) + """
def {entry}({a}, {b}, {out}, n):
    n = int(n)
    validate_1d(n, {a}, {b}, {out})
    for idx in range(n):
        {out}[idx] = float({a}[idx]) {symbol} float({b}[idx])
    return {out}
""".format(entry=safe_entry(plan), a=inputs[0], b=inputs[1], out=output, symbol=symbol)


def render_elementwise_unary(plan: TranslationPlan, operation: str) -> str:
    inputs = safe_names(plan.parameters["inputs"])
    output = safe_name(plan.parameters["output"])
    if len(inputs) != 1:
        raise BackendError("unary elementwise requires one input")
    expr = {
        "relu": "value if value > 0.0 else 0.0",
        "neg": "-value",
        "exp": "math.exp(value)",
    }.get(operation)
    if expr is None:
        raise BackendError("unsupported unary operation: %s" % operation)
    return header(plan, needs_math=operation == "exp") + """
def {entry}({a}, {out}, n):
    n = int(n)
    validate_1d(n, {a}, {out})
    for idx in range(n):
        value = float({a}[idx])
        {out}[idx] = {expr}
    return {out}
""".format(entry=safe_entry(plan), a=inputs[0], out=output, expr=expr)


def render_copy_cast(plan: TranslationPlan, operation: str) -> str:
    inputs = safe_names(plan.parameters["inputs"])
    output = safe_name(plan.parameters["output"])
    if len(inputs) != 1:
        raise BackendError("copy/cast requires one input")
    expr = "float(value)" if operation == "cast" else "value"
    return header(plan) + """
def {entry}({a}, {out}, n):
    n = int(n)
    validate_1d(n, {a}, {out})
    for idx in range(n):
        value = {a}[idx]
        {out}[idx] = {expr}
    return {out}
""".format(entry=safe_entry(plan), a=inputs[0], out=output, expr=expr)


def render_transpose2d(plan: TranslationPlan) -> str:
    inputs = safe_names(plan.parameters["inputs"])
    output = safe_name(plan.parameters["output"])
    if len(inputs) != 1:
        raise BackendError("transpose2d requires one input")
    return header(plan) + """
def {entry}({a}, {out}, rows, cols):
    rows = int(rows)
    cols = int(cols)
    validate_2d(rows, cols, {a}, {out})
    for row in range(rows):
        for col in range(cols):
            {out}[col * rows + row] = float({a}[row * cols + col])
    return {out}
""".format(entry=safe_entry(plan), a=inputs[0], out=output)


def render_row_reduce_sum(plan: TranslationPlan) -> str:
    inputs = safe_names(plan.parameters["inputs"])
    output = safe_name(plan.parameters["output"])
    if len(inputs) != 1:
        raise BackendError("row_reduce_sum requires one input")
    return header(plan) + """
def {entry}({a}, {out}, rows, cols):
    rows = int(rows)
    cols = int(cols)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if len({a}) < rows * cols or len({out}) < rows:
        raise ValueError("input/output buffers are smaller than the requested shape")
    for row in range(rows):
        acc = 0.0
        for col in range(cols):
            acc += float({a}[row * cols + col])
        {out}[row] = acc
    return {out}
""".format(entry=safe_entry(plan), a=inputs[0], out=output)


def header(plan: TranslationPlan, needs_math: bool = False) -> str:
    imports = "import math\n\n" if needs_math else ""
    return '''"""Generated by Geyi deterministic rule backend.

Backend identity: tilelang
Execution mode: local golden simulator
"""

{imports}BACKEND = "tilelang"
TEMPLATE = "{template}"
OPERATION = "{operation}"


def validate_1d(n, *buffers):
    if n < 0:
        raise ValueError("n must be non-negative")
    for buffer in buffers:
        if len(buffer) < n:
            raise ValueError("input/output buffers are smaller than n")


def validate_2d(rows, cols, *buffers):
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    size = rows * cols
    for buffer in buffers:
        if len(buffer) < size:
            raise ValueError("input/output buffers are smaller than rows * cols")

'''.format(imports=imports, template=plan.template, operation=plan.parameters["operation"])


def source_stem(plan: TranslationPlan) -> str:
    return "tilelang_%s" % safe_entry(plan)


def safe_entry(plan: TranslationPlan) -> str:
    return safe_name(plan.operator_entry or "kernel")


def safe_names(names: List[str]) -> List[str]:
    return [safe_name(name) for name in names]


def safe_name(name: str) -> str:
    candidate = re.sub(r"\W", "_", str(name))
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = "arg_%s" % candidate
    return candidate


def json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
