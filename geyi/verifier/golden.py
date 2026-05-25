"""Golden verifier for the Phase 0 generated project."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import List, Optional
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
        name = str(case["case"])
        n = int(case["n"])
        result = run_case(kernel, n, tolerance)
        result["case"] = name
        result["n"] = n
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
            shapes=[{"n": int(case["n"])} for case in cases],
            dtypes=[str(plan.parameters.get("dtype") or "float32")],
            strides=[[1]],
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
        llm_used=False,
        passed=passed,
        reproducible_commands=commands,
        case_results=case_results,
        cache={"hit": cache_hit, "artifact_reused": artifact.reused},
    )


def run_case(kernel, n: int, tolerance) -> dict:
    a = make_values(n, seed=17)
    b = make_values(n, seed=43)
    out = [-777.0 for _ in range(n)]
    kernel(a, b, out, n)

    expected = [float(a[index]) + float(b[index]) for index in range(n)]
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for actual, golden in zip(out, expected):
        abs_diff = abs(float(actual) - float(golden))
        rel_diff = abs_diff / max(abs(float(golden)), 1e-12)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

    atol = float(tolerance.get("atol", 1e-5))
    rtol = float(tolerance.get("rtol", 1e-5))
    threshold = atol + rtol * max([abs(value) for value in expected] or [0.0])
    return {
        "passed": max_abs_diff <= threshold,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "tolerance_threshold": threshold,
    }


def make_values(n: int, seed: int) -> List[float]:
    return [float(((index * 37 + seed) % 251) - 125) / 17.0 for index in range(n)]


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
