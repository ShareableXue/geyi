"""Constrained LLM planner for Phase 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geyi.context.compression import source_snippet
from geyi.context.orchestrator import ContextOrchestrator
from geyi.contract.model import SemanticContract
from geyi.llm.client import LLMCredentialError, LLMProvider, LLMProviderError
from geyi.llm.prompt_builder import planner_messages, schema_repair_messages
from geyi.llm.schemas import PlannerOutput, PlannerSchemaError, validate_planner_json
from geyi.planner.plan import (
    DETERMINISTIC_BACKEND,
    PlanError,
    TranslationPlan,
    coverage_cases_for_intent,
    default_tolerance,
    tensor_dtypes,
)
from geyi.session import SessionStore


AVAILABLE_TEMPLATES = [
    {
        "backend": "tilelang",
        "template": "tilelang.fused_add_relu_1d",
        "intent": "elementwise.fused_add_relu",
        "constraints": ["rank=1", "contiguous", "two inputs", "one output"],
    },
    {
        "backend": "tilelang",
        "template": "tilelang.elementwise_binary_1d",
        "intent": "elementwise.add|elementwise.mul",
        "constraints": ["rank=1", "contiguous"],
    },
]


@dataclass
class LLMPlanResult:
    plan: TranslationPlan
    planner_output: PlannerOutput
    llm_calls: list[dict[str, Any]]
    child_session_path: str


class PlannerHandoffRequired(RuntimeError):
    def __init__(self, handoff: dict[str, Any]):
        super().__init__(str(handoff.get("reason") or "planner handoff required"))
        self.handoff = handoff


def plan_with_llm(
    contract: SemanticContract,
    source_path: str | None,
    session: SessionStore,
    provider: LLMProvider,
    backend: str = DETERMINISTIC_BACKEND,
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
) -> LLMPlanResult:
    handoff = handoff_for_blocking_unknowns(contract)
    if handoff:
        session.write_json("handoff/planner.json", handoff)
        raise PlannerHandoffRequired(handoff)

    orchestrator = ContextOrchestrator(session.path)
    child = orchestrator.start_child("planner")
    messages = planner_messages(
        contract,
        source_snippet(source_path),
        AVAILABLE_TEMPLATES,
        diagnostics=None,
    )
    try:
        response = orchestrator.complete(
            child,
            provider,
            messages,
            task="planner",
            metadata={"contract_hash": contract.contract_hash, "entry": contract.entry},
        )
    except LLMCredentialError as exc:
        handoff = {
            "reason": "LLM provider credentials unavailable",
            "contract_hash": contract.contract_hash,
            "entry": contract.entry,
            "provider": provider.name,
            "model": provider.model,
            "message": str(exc),
            "question": "Configure the provider API key or rerun with --llm-provider mock for offline validation.",
        }
        child.write_json("output/credential_handoff.json", handoff)
        session.write_json("handoff/credentials.json", handoff)
        raise PlannerHandoffRequired(handoff) from exc
    except LLMProviderError as exc:
        handoff = {
            "reason": "LLM provider request failed",
            "contract_hash": contract.contract_hash,
            "entry": contract.entry,
            "provider": provider.name,
            "model": provider.model,
            "message": str(exc),
            "question": "Check provider/model/base URL settings, then rerun the same command.",
        }
        child.write_json("output/provider_error_handoff.json", handoff)
        session.write_json("handoff/provider_error.json", handoff)
        raise PlannerHandoffRequired(handoff) from exc
    llm_calls = [response.usage.to_dict()]

    try:
        planner_output = validate_planner_json(response.content)
    except PlannerSchemaError as exc:
        child.write_json(
            "output/schema_rejection.json",
            {"error": str(exc), "raw_output": response.content, "repair_prompt": True},
        )
        repair_child = orchestrator.start_child("planner_schema_repair")
        repair_response = orchestrator.complete(
            repair_child,
            provider,
            schema_repair_messages(response.content, str(exc)),
            task="planner",
            metadata={"contract_hash": contract.contract_hash, "reason": "invalid_planner_json"},
        )
        llm_calls.append(repair_response.usage.to_dict())
        planner_output = validate_planner_json(repair_response.content)
        repair_child.write_json("output/planner_output.json", planner_output.to_dict())

    child.write_json("output/planner_output.json", planner_output.to_dict())
    if planner_output.cannot_translate:
        handoff = handoff_from_planner_output(contract, planner_output)
        session.write_json("handoff/planner.json", handoff)
        raise PlannerHandoffRequired(handoff)

    plan = planner_output_to_plan(
        contract,
        planner_output,
        backend=backend,
        target=target,
        npu_arch=npu_arch,
    )
    child.write_json("output/translation_plan.json", plan.to_dict())
    session.write_json(
        "llm/planner_report.json",
        {
            "provider": provider.name,
            "model": provider.model,
            "llm_calls": llm_calls,
            "planner_child_session": str(child.path),
            "schema_valid": True,
            "planner_output": planner_output.to_dict(),
        },
    )
    return LLMPlanResult(
        plan=plan,
        planner_output=planner_output,
        llm_calls=llm_calls,
        child_session_path=str(child.path),
    )


def planner_output_to_plan(
    contract: SemanticContract,
    output: PlannerOutput,
    backend: str = DETERMINISTIC_BACKEND,
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
) -> TranslationPlan:
    if output.cannot_translate:
        raise PlanError("cannot build a TranslationPlan from cannot_translate planner output")
    if not contract.intents:
        raise PlanError("LLM planner requires a contract intent")

    intent = contract.intents[0]
    if output.selected_backend != backend and backend != "auto":
        raise PlanError("planner selected backend %s, but CLI requested %s" % (output.selected_backend, backend))
    if output.selected_backend != "tilelang":
        raise PlanError("Phase 2 currently supports LLM planner backend=tilelang only")
    if output.selected_template != "tilelang.fused_add_relu_1d":
        raise PlanError("unsupported Phase 2 planner template: %s" % output.selected_template)
    if intent.kind != "elementwise" or intent.subkind != "fused_add_relu":
        raise PlanError("tilelang.fused_add_relu_1d requires contract intent elementwise.fused_add_relu")
    if len(intent.inputs) != 2 or len(intent.outputs) != 1:
        raise PlanError("fused_add_relu planner template requires two inputs and one output")

    output_name = intent.outputs[0]
    required_tensors = list(intent.inputs) + [output_name]
    parameters = {
        "phase": "phase2",
        "target": target if target != "auto" else "local_cpu",
        "npu_arch": npu_arch,
        "pattern": intent.kind,
        "operation": "fused_add_relu",
        "inputs": list(intent.inputs),
        "output": output_name,
        "rank": 1,
        "shape": list(contract.tensors[output_name].shape),
        "input_shapes": {name: list(contract.tensors[name].shape) for name in intent.inputs},
        "output_shape": list(contract.tensors[output_name].shape),
        "dtypes": tensor_dtypes(contract, required_tensors),
        "dtype": contract.tensors[output_name].dtype,
        "axes": list(intent.axes),
        "reduction_axes": list(intent.reduction_axes),
        "tolerance": default_tolerance(),
        "coverage_cases": coverage_cases_for_intent(intent),
    }
    for key, value in output.parameter_bindings.items():
        if key in {"phase", "operation", "inputs", "output", "contract_hash"}:
            raise PlanError("planner may not override protected plan parameter: %s" % key)
        parameters[key] = value

    return TranslationPlan(
        contract_hash=contract.contract_hash,
        strategy="llm_plan",
        backend=output.selected_backend,
        template=output.selected_template,
        operator_entry=contract.entry,
        parameters=parameters,
        optimization_hints={},
        required_assumptions=sorted({assumption.id for assumption in contract.assumptions} | set(output.required_assumptions)),
        expected_verification="golden",
        confidence=contract.confidence,
        notes=[
            "Phase 2 constrained LLM planner path.",
            output.intent_confirmation,
            "LLM selected template; Geyi still owns code generation, compile, and verification.",
        ]
        + list(output.risks),
    )


def handoff_for_blocking_unknowns(contract: SemanticContract) -> dict[str, Any] | None:
    blocking = [
        unknown
        for unknown in contract.unknowns
        if unknown.impact in {"correctness", "both"} and not unknown.id.startswith("template_gap.")
    ]
    if not blocking:
        return None
    return {
        "reason": "planner blocked by correctness unknowns",
        "contract_hash": contract.contract_hash,
        "entry": contract.entry,
        "blocked_unknowns": [unknown.__dict__ for unknown in blocking],
        "question": "Please add the missing correctness annotation in geyi.yaml, then rerun the same command.",
    }


def handoff_from_planner_output(contract: SemanticContract, output: PlannerOutput) -> dict[str, Any]:
    return {
        "reason": "LLM planner could not safely choose a constrained template",
        "contract_hash": contract.contract_hash,
        "entry": contract.entry,
        "planner_output": output.to_dict(),
        "annotation_request": output.annotation_request,
    }


def read_plan_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
