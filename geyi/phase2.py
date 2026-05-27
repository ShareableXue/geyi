"""Phase 2 constrained LLM-planner pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from geyi.analysis import AnalysisResult, analyze
from geyi.backend.model import CompiledArtifact, GeneratedProject
from geyi.config import DEFAULT_SESSION_ROOT
from geyi.llm.client import LLMProvider, create_provider
from geyi.llm.diagnostics import diagnose_precision_mismatch
from geyi.llm.planner import LLMPlanResult, plan_with_llm
from geyi.llm.repair import collect_compile_diagnostic, repair_compile_error
from geyi.phase1 import create_backend
from geyi.phase3 import apply_phase3_optimization_artifacts
from geyi.planner.plan import PlanError, TranslationPlan, create_deterministic_plan
from geyi.verifier.ascendc import verify_ascendc
from geyi.verifier.golden import verify_with_golden
from geyi.verifier.report import VerificationReport


@dataclass
class Phase2RunResult:
    analysis: AnalysisResult
    plan: TranslationPlan
    project: GeneratedProject
    artifact: CompiledArtifact
    verification_report: VerificationReport
    out_path: Path
    cache_hit: bool
    llm_plan_result: Optional[LLMPlanResult]


def run_phase2(
    source: str,
    spec: str,
    out: Optional[str] = None,
    session_root: str = DEFAULT_SESSION_ROOT,
    reproducible_command: Optional[str] = None,
    backend: str = "tilelang",
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
    allow_llm_plan: bool = False,
    llm_provider: str = "mock",
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    opt_level: str = "none",
    provider: Optional[LLMProvider] = None,
) -> Phase2RunResult:
    analysis = analyze(source, spec=spec, session_root=session_root, write_session=True)
    if analysis.session is None:
        raise RuntimeError("Phase 2 requires session artifacts")

    session = analysis.session
    contract = analysis.contract
    llm_result: Optional[LLMPlanResult] = None
    llm_calls: list[dict] = []

    if contract.recommended_path == "rule":
        plan = create_deterministic_plan(contract, backend=backend, target=target, npu_arch=npu_arch)
    elif allow_llm_plan and contract.recommended_path in {"llm_plan", "template"}:
        active_provider = provider or create_provider(llm_provider, model=llm_model, base_url=llm_base_url)
        llm_result = plan_with_llm(
            contract,
            source,
            session,
            active_provider,
            backend=backend,
            target=target,
            npu_arch=npu_arch,
        )
        plan = llm_result.plan
        llm_calls.extend(llm_result.llm_calls)
    else:
        raise PlanError(
            "contract routes to %s; rerun with --allow-llm-plan for Phase 2 planner path"
            % contract.recommended_path
        )

    session.write_json("plan.json", plan.to_dict())
    out_path = Path(out) if out else Path(".geyi") / "out" / contract.entry
    backend_impl = create_backend(plan.backend)
    cache_hit = False

    project, artifact, plan = generate_compile_with_optional_repair(
        contract=contract,
        plan=plan,
        backend_impl=backend_impl,
        session_path=session.path,
        session=analysis.session,
        provider=provider or (create_provider(llm_provider, model=llm_model, base_url=llm_base_url) if allow_llm_plan else None),
        llm_calls=llm_calls,
        backend=backend,
        target=target,
        npu_arch=npu_arch,
    )
    session.write_json("plan.json", plan.to_dict())

    if plan.backend == "ascendc":
        report = verify_ascendc(
            contract,
            plan,
            project,
            artifact,
            reproducible_command=reproducible_command,
            cache_hit=cache_hit,
        )
    else:
        report = verify_with_golden(
            contract,
            plan,
            project,
            artifact,
            reproducible_command=reproducible_command,
            cache_hit=cache_hit,
            llm_usage=llm_calls if plan.strategy == "llm_plan" else None,
        )
    session.write_json("verification_report.json", report.to_dict())
    if not report.passed:
        session.write_json("diagnostics/precision.json", diagnose_precision_mismatch(report))
    session.write_log("run.log", render_phase2_log(contract.entry, out_path, cache_hit, report, plan, llm_calls))
    mirror_to_out(session.path, out_path, contract.contract_hash, artifact.artifact_hash, plan, cache_hit)
    apply_phase3_optimization_artifacts(session, out_path, contract, plan, report, opt_level=opt_level)

    return Phase2RunResult(
        analysis=analysis,
        plan=plan,
        project=project,
        artifact=artifact,
        verification_report=report,
        out_path=out_path,
        cache_hit=cache_hit,
        llm_plan_result=llm_result,
    )


def generate_compile_with_optional_repair(
    contract,
    plan: TranslationPlan,
    backend_impl,
    session_path: Path,
    session,
    provider: Optional[LLMProvider],
    llm_calls: list[dict],
    backend: str,
    target: str,
    npu_arch: str,
):
    try:
        project = backend_impl.generate(contract, plan, session_path / "generated")
        if plan.backend == "ascendc":
            artifact = backend_impl.compile(project, session_path / "build", target=str(plan.parameters["target"]))
        else:
            artifact = backend_impl.compile(project, session_path / "build")
        return project, artifact, plan
    except Exception as exc:
        if plan.strategy != "llm_plan" or provider is None:
            raise
        diagnostic = collect_compile_diagnostic(exc, "compile_or_generate", session_path / "generated")
        outcome = repair_compile_error(
            contract,
            plan,
            diagnostic,
            session,
            provider,
            backend=backend,
            target=target,
            npu_arch=npu_arch,
        )
        llm_calls.extend(outcome.llm_calls)
        if outcome.status != "repaired" or outcome.plan is None:
            session.write_json("handoff/repair.json", outcome.to_dict())
            raise RuntimeError("Phase 2 repair escalated: %s" % outcome.reason) from exc
        repaired_plan = outcome.plan
        project = backend_impl.generate(contract, repaired_plan, session_path / "generated")
        artifact = backend_impl.compile(project, session_path / "build")
        return project, artifact, repaired_plan


def mirror_to_out(
    session_path: Path,
    out_path: Path,
    contract_hash: str,
    artifact_hash: str,
    plan: TranslationPlan,
    cache_hit: bool,
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    for name in ["contract.json", "plan.json", "verification_report.json"]:
        shutil.copy2(str(session_path / name), str(out_path / name))
    copy_tree(session_path / "generated", out_path / "generated")
    copy_tree(session_path / "build", out_path / "build")
    write_json(
        out_path / "cache_manifest.json",
        {
            "phase": "phase2" if plan.strategy == "llm_plan" else "phase1",
            "contract_hash": contract_hash,
            "artifact_hash": artifact_hash,
            "backend": plan.backend,
            "target": plan.parameters.get("target"),
            "cache_hit_for_this_run": cache_hit,
            "strategy": plan.strategy,
            "llm_used": plan.strategy == "llm_plan",
        },
    )


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_phase2_log(
    entry: str,
    out_path: Path,
    cache_hit: bool,
    report: VerificationReport,
    plan: TranslationPlan,
    llm_calls: list[dict],
) -> str:
    return "\n".join(
        [
            "Phase 2 run completed for %s" % entry,
            "strategy=%s" % plan.strategy,
            "backend=%s" % plan.backend,
            "template=%s" % plan.template,
            "out=%s" % out_path,
            "cache_hit=%s" % cache_hit,
            "llm_calls=%d" % len(llm_calls),
            "level=%s" % report.level.value,
            "passed=%s" % report.passed,
            "artifact_hash=%s" % report.artifact_hash,
        ]
    )
