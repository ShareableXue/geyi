"""Minimal Phase 0 TileLang backend.

The generated source is intentionally local and dependency-free. It preserves
the backend/project boundary while the real TileLang/CANN toolchain is absent.
"""

from __future__ import annotations

import json
import py_compile
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict

from geyi.backend.model import CompiledArtifact, GeneratedProject, sha256_file
from geyi.contract.model import SemanticContract
from geyi.planner.plan import PHASE0_BACKEND, TranslationPlan


class BackendError(RuntimeError):
    pass


class TileLangBackend:
    name = PHASE0_BACKEND

    def can_generate(self, plan: TranslationPlan) -> bool:
        return plan.backend == self.name and plan.template == "tilelang.elementwise_binary_1d"

    def generate(self, contract: SemanticContract, plan: TranslationPlan, root: Path) -> GeneratedProject:
        if not self.can_generate(plan):
            raise BackendError("TileLang backend cannot generate this plan")

        root.mkdir(parents=True, exist_ok=True)
        (root / "tests").mkdir(parents=True, exist_ok=True)

        kernel_source = root / "tilelang_vector_add.py"
        build_file = root / "build.json"
        test_file = root / "tests" / "golden_cases.json"
        metadata_file = root / "metadata.json"

        kernel_source.write_text(render_kernel(plan), encoding="utf-8")
        build_file.write_text(json_dumps({"compiler": "py_compile", "artifact": "tilelang_vector_add.pyc"}), encoding="utf-8")
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

        artifact = build_root / "tilelang_vector_add.pyc"
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
    entry = plan.operator_entry or "vector_add"
    inputs = plan.parameters["inputs"]
    output = plan.parameters["output"]
    if len(inputs) != 2:
        raise BackendError("Phase 0 vector_add requires two inputs")

    return '''"""Generated by Geyi Phase 0.

Backend identity: tilelang
Execution mode: local golden simulator
"""

BACKEND = "tilelang"
TEMPLATE = "tilelang.elementwise_binary_1d"


def {entry}({a}, {b}, {out}, n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if len({a}) < n or len({b}) < n or len({out}) < n:
        raise ValueError("input/output buffers are smaller than n")
    for idx in range(n):
        {out}[idx] = float({a}[idx]) + float({b}[idx])
    return {out}
'''.format(entry=entry, a=inputs[0], b=inputs[1], out=output)


def json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
