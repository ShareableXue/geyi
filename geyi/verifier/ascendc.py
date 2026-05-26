"""Verification report adapter for AscendC hardware runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from geyi.backend.model import CompiledArtifact, GeneratedProject
from geyi.contract.model import SemanticContract
from geyi.planner.plan import TranslationPlan
from geyi.verifier.golden import case_shape, coverage_dtypes, coverage_strides
from geyi.verifier.report import Coverage, VerificationLevel, VerificationReport


def verify_ascendc(
    contract: SemanticContract,
    plan: TranslationPlan,
    project: GeneratedProject,
    artifact: CompiledArtifact,
    reproducible_command: Optional[str] = None,
    cache_hit: bool = False,
) -> VerificationReport:
    target = str(plan.parameters.get("target") or "scaffold")
    if target == "cann":
        return verify_ascendc_cann(contract, plan, project, artifact, reproducible_command, cache_hit)
    return verify_ascendc_scaffold(contract, plan, artifact, reproducible_command, cache_hit)


def verify_ascendc_scaffold(
    contract: SemanticContract,
    plan: TranslationPlan,
    artifact: CompiledArtifact,
    reproducible_command: Optional[str],
    cache_hit: bool,
) -> VerificationReport:
    cases = list(plan.parameters.get("coverage_cases") or [])
    commands = []
    if reproducible_command:
        commands.append(reproducible_command)
    commands.append("cd <generated-project> && bash run.sh")
    return VerificationReport(
        level=VerificationLevel.COMPILES_ONLY,
        contract_hash=contract.contract_hash,
        artifact_hash=artifact.artifact_hash,
        coverage=Coverage(
            shapes=[case_shape(case) for case in cases],
            dtypes=coverage_dtypes(plan),
            strides=coverage_strides(plan),
            edge_cases=[str(case["case"]) for case in cases],
            hardware=["not_run"],
        ),
        tolerance=dict(plan.parameters.get("tolerance") or {"atol": 1e-5, "rtol": 1e-5}),
        max_abs_diff=None,
        max_rel_diff=None,
        assumptions=[assumption.id for assumption in contract.assumptions],
        unknowns=[unknown.id for unknown in contract.unknowns],
        strategy=plan.strategy,
        backend=plan.backend,
        llm_used=False,
        passed=True,
        reproducible_commands=commands,
        case_results=[],
        cache={"hit": cache_hit, "artifact_reused": artifact.reused, "target": "scaffold"},
    )


def verify_ascendc_cann(
    contract: SemanticContract,
    plan: TranslationPlan,
    project: GeneratedProject,
    artifact: CompiledArtifact,
    reproducible_command: Optional[str],
    cache_hit: bool,
) -> VerificationReport:
    report_path = Path(project.root) / "build" / "output" / "verification_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cases = list(plan.parameters.get("coverage_cases") or [])
    passed = bool(payload.get("passed"))
    commands = []
    if reproducible_command:
        commands.append(reproducible_command)
    commands.append("cd %s && bash run.sh" % project.root)
    return VerificationReport(
        level=VerificationLevel.GOLDEN if passed else VerificationLevel.UNVERIFIED,
        contract_hash=contract.contract_hash,
        artifact_hash=artifact.artifact_hash,
        coverage=Coverage(
            shapes=[case_shape(case) for case in cases],
            dtypes=coverage_dtypes(plan),
            strides=coverage_strides(plan),
            edge_cases=[str(case["case"]) for case in cases],
            hardware=["ascend_npu"],
        ),
        tolerance=dict(plan.parameters.get("tolerance") or {"atol": 1e-5, "rtol": 1e-5}),
        max_abs_diff=payload.get("max_abs_diff"),
        max_rel_diff=payload.get("max_rel_diff"),
        assumptions=[assumption.id for assumption in contract.assumptions],
        unknowns=[unknown.id for unknown in contract.unknowns],
        strategy=plan.strategy,
        backend=plan.backend,
        llm_used=False,
        passed=passed,
        reproducible_commands=commands,
        case_results=list(payload.get("case_results") or []),
        cache={"hit": cache_hit, "artifact_reused": artifact.reused, "target": "cann"},
    )
