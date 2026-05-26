"""Golden verifier for deterministic generated projects."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from geyi.backend.model import CompiledArtifact, GeneratedProject
from geyi.contract.model import SemanticContract
from geyi.planner.plan import TranslationPlan
from geyi.verifier.report import Coverage, VerificationLevel, VerificationReport


def verify_with_golden(
    contract: SemanticContract,
    plan: TranslationPlan,
    project: GeneratedProject,
    artifact: CompiledArtifact,
    reproducible_command: Optional[str] = None,
    cache_hit: bool = False,
    llm_usage: Optional[List[dict]] = None,
) -> VerificationReport:
    module = load_generated_module(project)
    entry = plan.operator_entry or contract.entry
    kernel = getattr(module, entry)
    tolerance = dict(plan.parameters.get("tolerance") or {"atol": 1e-5, "rtol": 1e-5})
    cases = list(plan.parameters.get("coverage_cases") or [])

    case_results = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    passed = True
    for case in cases:
        result = run_case(kernel, plan, case, tolerance)
        case_results.append(result)
        max_abs_diff = max(max_abs_diff, float(result["max_abs_diff"]))
        max_rel_diff = max(max_rel_diff, float(result["max_rel_diff"]))
        passed = passed and bool(result["passed"])

    level = VerificationLevel.GOLDEN if passed else VerificationLevel.UNVERIFIED
    commands: List[str] = []
    if reproducible_command:
        commands.append(reproducible_command)

    return VerificationReport(
        level=level,
        contract_hash=contract.contract_hash,
        artifact_hash=artifact.artifact_hash,
        coverage=Coverage(
            shapes=[case_shape(case) for case in cases],
            dtypes=coverage_dtypes(plan),
            strides=coverage_strides(plan),
            edge_cases=[str(case["case"]) for case in cases],
            hardware=["local_cpu"],
        ),
        tolerance={"atol": float(tolerance["atol"]), "rtol": float(tolerance["rtol"])},
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        assumptions=[assumption.id for assumption in contract.assumptions],
        unknowns=[unknown.id for unknown in contract.unknowns],
        strategy=plan.strategy,
        backend=plan.backend,
        llm_used=bool(llm_usage),
        passed=passed,
        reproducible_commands=commands,
        case_results=case_results,
        cache={"hit": cache_hit, "artifact_reused": artifact.reused},
        llm_calls=list(llm_usage or []),
    )


def run_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    operation = str(plan.parameters["operation"])
    if operation in {"add", "mul"}:
        return run_elementwise_binary_case(kernel, plan, case, tolerance)
    if operation in {"relu", "neg", "exp"}:
        return run_elementwise_unary_case(kernel, plan, case, tolerance)
    if operation == "fused_add_relu":
        return run_fused_add_relu_case(kernel, plan, case, tolerance)
    if operation in {"copy", "cast"}:
        return run_copy_cast_case(kernel, plan, case, tolerance)
    if operation == "transpose2d":
        return run_transpose_case(kernel, plan, case, tolerance)
    if operation == "row_sum":
        return run_row_reduce_sum_case(kernel, plan, case, tolerance)
    raise RuntimeError("unsupported golden operation: %s" % operation)


def run_elementwise_binary_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    n = int(case["n"])
    dtypes = dict(plan.parameters.get("dtypes") or {})
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    a = make_values(n, dtype=dtypes.get(inputs[0], "float32"), seed=17)
    b = make_values(n, dtype=dtypes.get(inputs[1], "float32"), seed=43)
    out = [-777.0 for _ in range(n)]
    kernel(a, b, out, n)

    if plan.parameters["operation"] == "add":
        expected = [float(a[index]) + float(b[index]) for index in range(n)]
    else:
        expected = [float(a[index]) * float(b[index]) for index in range(n)]
    return with_case_metadata(case, compare(out, expected, tolerance), output)


def run_elementwise_unary_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    n = int(case["n"])
    dtypes = dict(plan.parameters.get("dtypes") or {})
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    a = make_values(n, dtype=dtypes.get(inputs[0], "float32"), seed=29)
    out = [-777.0 for _ in range(n)]
    kernel(a, out, n)

    operation = plan.parameters["operation"]
    if operation == "relu":
        expected = [float(value) if float(value) > 0.0 else 0.0 for value in a]
    elif operation == "neg":
        expected = [-float(value) for value in a]
    else:
        import math

        expected = [math.exp(float(value)) for value in a]
    return with_case_metadata(case, compare(out, expected, tolerance), output)


def run_fused_add_relu_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    n = int(case["n"])
    dtypes = dict(plan.parameters.get("dtypes") or {})
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    a = make_values(n, dtype=dtypes.get(inputs[0], "float32"), seed=19)
    b = make_values(n, dtype=dtypes.get(inputs[1], "float32"), seed=53)
    out = [-777.0 for _ in range(n)]
    kernel(a, b, out, n)

    expected = []
    for index in range(n):
        value = float(a[index]) + float(b[index])
        expected.append(value if value > 0.0 else 0.0)
    return with_case_metadata(case, compare(out, expected, tolerance), output)


def run_copy_cast_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    n = int(case["n"])
    dtypes = dict(plan.parameters.get("dtypes") or {})
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    a = make_values(n, dtype=dtypes.get(inputs[0], "float32"), seed=31)
    out = [-777.0 for _ in range(n)]
    kernel(a, out, n)

    if plan.parameters["operation"] == "cast":
        expected = [float(value) for value in a]
    else:
        expected = list(a)
    return with_case_metadata(case, compare(out, expected, tolerance), output)


def run_transpose_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    rows = int(case["rows"])
    cols = int(case["cols"])
    size = rows * cols
    dtypes = dict(plan.parameters.get("dtypes") or {})
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    matrix = make_values(size, dtype=dtypes.get(inputs[0], "float32"), seed=37)
    out = [-777.0 for _ in range(size)]
    kernel(matrix, out, rows, cols)

    expected = [-777.0 for _ in range(size)]
    for row in range(rows):
        for col in range(cols):
            expected[col * rows + row] = float(matrix[row * cols + col])
    return with_case_metadata(case, compare(out, expected, tolerance), output)


def run_row_reduce_sum_case(kernel, plan: TranslationPlan, case: Dict[str, object], tolerance: Dict[str, float]) -> dict:
    rows = int(case["rows"])
    cols = int(case["cols"])
    size = rows * cols
    dtypes = dict(plan.parameters.get("dtypes") or {})
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    matrix = make_values(size, dtype=dtypes.get(inputs[0], "float32"), seed=41)
    out = [-777.0 for _ in range(rows)]
    kernel(matrix, out, rows, cols)

    expected = []
    for row in range(rows):
        acc = 0.0
        for col in range(cols):
            acc += float(matrix[row * cols + col])
        expected.append(acc)
    return with_case_metadata(case, compare(out, expected, tolerance), output)


def compare(actual: List[float], expected: List[float], tolerance: Dict[str, float]) -> dict:
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for actual_value, golden in zip(actual, expected):
        abs_diff = abs(float(actual_value) - float(golden))
        rel_diff = abs_diff / max(abs(float(golden)), 1e-12)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

    atol = float(tolerance.get("atol", 1e-5))
    rtol = float(tolerance.get("rtol", 1e-5))
    threshold = atol + rtol * max([abs(float(value)) for value in expected] or [0.0])
    return {
        "passed": max_abs_diff <= threshold,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "tolerance_threshold": threshold,
    }


def with_case_metadata(case: Dict[str, object], result: dict, output: str) -> dict:
    payload = dict(result)
    payload["case"] = str(case["case"])
    payload["output"] = output
    for key, value in case.items():
        if key != "case":
            payload[key] = value
    return payload


def make_values(n: int, dtype: str, seed: int) -> List[float | int]:
    if dtype in {"int32", "int64"}:
        return [int(((index * 37 + seed) % 251) - 125) for index in range(n)]
    return [float(((index * 37 + seed) % 251) - 125) / 17.0 for index in range(n)]


def case_shape(case: Dict[str, object]) -> Dict[str, int]:
    if "n" in case:
        return {"n": int(case["n"])}
    return {"rows": int(case["rows"]), "cols": int(case["cols"])}


def coverage_dtypes(plan: TranslationPlan) -> List[str]:
    dtypes = dict(plan.parameters.get("dtypes") or {})
    values = sorted({str(value) for value in dtypes.values()})
    return values or [str(plan.parameters.get("dtype") or "float32")]


def coverage_strides(plan: TranslationPlan) -> List[List[int]]:
    rank = int(plan.parameters.get("rank") or 1)
    if rank == 2:
        return [[0, 1]]
    return [[1]]


def load_generated_module(project: GeneratedProject):
    if not project.kernel_sources:
        raise RuntimeError("generated project has no kernel source")
    path = Path(project.root) / project.kernel_sources[0]
    module_name = "geyi_generated_%s" % uuid4().hex
    loader = SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise RuntimeError("could not load generated source: %s" % path)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
