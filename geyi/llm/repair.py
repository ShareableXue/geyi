"""Compile-repair loop for Phase 2 constrained plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geyi.context.orchestrator import ContextOrchestrator
from geyi.contract.model import SemanticContract
from geyi.llm.client import LLMProvider
from geyi.llm.planner import AVAILABLE_TEMPLATES, planner_output_to_plan
from geyi.llm.prompt_builder import compile_repair_messages
from geyi.llm.schemas import PlannerSchemaError, validate_planner_json
from geyi.planner.plan import TranslationPlan
from geyi.session import SessionStore


@dataclass
class CompileDiagnostic:
    stage: str
    message: str
    exception_type: str
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "exception_type": self.exception_type,
            "source_path": self.source_path,
        }


@dataclass
class RepairOutcome:
    status: str
    diagnostic: CompileDiagnostic
    plan: TranslationPlan | None
    llm_calls: list[dict[str, Any]]
    child_session_path: str | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "diagnostic": self.diagnostic.to_dict(),
            "plan": self.plan.to_dict() if self.plan else None,
            "llm_calls": self.llm_calls,
            "child_session_path": self.child_session_path,
            "reason": self.reason,
        }


def collect_compile_diagnostic(exc: Exception, stage: str, source_path: Path | None = None) -> CompileDiagnostic:
    return CompileDiagnostic(
        stage=stage,
        message=str(exc),
        exception_type=exc.__class__.__name__,
        source_path=str(source_path) if source_path else None,
    )


def repair_compile_error(
    contract: SemanticContract,
    failed_plan: TranslationPlan,
    diagnostic: CompileDiagnostic,
    session: SessionStore,
    provider: LLMProvider,
    backend: str = "tilelang",
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
) -> RepairOutcome:
    orchestrator = ContextOrchestrator(session.path)
    child = orchestrator.start_child("repair")
    messages = compile_repair_messages(
        contract,
        failed_plan.to_dict(),
        diagnostic.to_dict(),
        AVAILABLE_TEMPLATES,
    )
    response = orchestrator.complete(
        child,
        provider,
        messages,
        task="repair",
        metadata={"contract_hash": contract.contract_hash, "stage": diagnostic.stage},
    )
    llm_calls = [response.usage.to_dict()]
    try:
        planner_output = validate_planner_json(response.content)
        if planner_output.cannot_translate:
            outcome = RepairOutcome(
                status="escalated",
                diagnostic=diagnostic,
                plan=None,
                llm_calls=llm_calls,
                child_session_path=str(child.path),
                reason="repair planner returned cannot_translate",
            )
        else:
            repaired_plan = planner_output_to_plan(
                contract,
                planner_output,
                backend=backend,
                target=target,
                npu_arch=npu_arch,
            )
            outcome = RepairOutcome(
                status="repaired",
                diagnostic=diagnostic,
                plan=repaired_plan,
                llm_calls=llm_calls,
                child_session_path=str(child.path),
                reason="repair planner returned a schema-valid constrained plan",
            )
            child.write_json("output/repaired_plan.json", repaired_plan.to_dict())
    except (PlannerSchemaError, ValueError) as exc:
        outcome = RepairOutcome(
            status="escalated",
            diagnostic=diagnostic,
            plan=None,
            llm_calls=llm_calls,
            child_session_path=str(child.path),
            reason="repair output rejected: %s" % exc,
        )
    child.write_json("output/repair_outcome.json", outcome.to_dict())
    session.write_json("llm/repair_report.json", outcome.to_dict())
    return outcome
