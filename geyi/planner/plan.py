"""Phase 0 deterministic rule planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geyi.contract.model import SemanticContract, to_jsonable


PHASE0_BACKEND = "tilelang"
PHASE0_TEMPLATE = "tilelang.elementwise_binary_1d"


class PlanError(ValueError):
    """Raised when a contract is outside the supported planning surface."""


@dataclass
class TranslationPlan:
    contract_hash: str
    strategy: str
    backend: str
    template: Optional[str]
    operator_entry: Optional[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    required_assumptions: List[str] = field(default_factory=list)
    expected_verification: str = "golden"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def create_phase0_plan(contract: SemanticContract) -> TranslationPlan:
    """Build a rule plan for the Phase 0 vector-add path."""

    validate_phase0_contract(contract)
    intent = contract.intents[0]
    output = intent.outputs[0]
    output_tensor = contract.tensors[output]
    shape = output_tensor.shape or ["n"]

    return TranslationPlan(
        contract_hash=contract.contract_hash,
        strategy="rule",
        backend=PHASE0_BACKEND,
        template=PHASE0_TEMPLATE,
        operator_entry=contract.entry,
        parameters={
            "operation": "add",
            "inputs": list(intent.inputs),
            "output": output,
            "shape": list(shape),
            "dtype": output_tensor.dtype,
            "index_axis": intent.axes[0] if intent.axes else "idx",
            "tolerance": {"atol": 1e-5, "rtol": 1e-5},
            "coverage_cases": [
                {"case": "basic", "n": 1024},
                {"case": "tail", "n": 1025},
                {"case": "small", "n": 1},
                {"case": "zero", "n": 0},
            ],
        },
        optimization_hints={},
        required_assumptions=[assumption.id for assumption in contract.assumptions],
        expected_verification="golden",
        confidence=contract.confidence,
        notes=[
            "Phase 0 deterministic rule path.",
            "Only 1D contiguous float32 vector_add is supported.",
            "Local golden verification is used; no NPU execution is claimed.",
        ],
    )


def validate_phase0_contract(contract: SemanticContract) -> None:
    if contract.confidence < 0.95:
        raise PlanError("Phase 0 requires contract confidence >= 0.95")
    if contract.recommended_path != "rule":
        raise PlanError("Phase 0 requires recommended_path=rule")
    if contract.unknowns:
        raise PlanError("Phase 0 requires no correctness unknowns")
    if contract.rejections:
        raise PlanError("Phase 0 requires no rejected CUDA features")
    if not contract.contract_hash:
        raise PlanError("Phase 0 requires a stable contract_hash")
    if not contract.intents:
        raise PlanError("Phase 0 requires one recognized intent")

    intent = contract.intents[0]
    if intent.kind != "elementwise" or intent.subkind != "add":
        raise PlanError("Phase 0 only supports 1D contiguous elementwise.add")
    if len(intent.inputs) != 2 or len(intent.outputs) != 1:
        raise PlanError("Phase 0 elementwise.add requires two inputs and one output")

    if not contract.effects or contract.effects[0].kind != "pure_store":
        raise PlanError("Phase 0 requires a pure_store effect")
    if not contract.control_flow or contract.control_flow[0].kind not in {"guarded_store", "straight_line"}:
        raise PlanError("Phase 0 requires guarded or straight-line store control flow")

    required_tensors = list(intent.inputs) + list(intent.outputs)
    missing = [name for name in required_tensors if name not in contract.tensors]
    if missing:
        raise PlanError("Phase 0 missing tensor contracts: %s" % ", ".join(missing))

    for name in required_tensors:
        tensor = contract.tensors[name]
        if tensor.dtype != "float32":
            raise PlanError("Phase 0 only supports float32 tensors")
        if len(tensor.shape) != 1:
            raise PlanError("Phase 0 only supports rank-1 tensors")
        if tensor.stride not in (["1"], [1], None) and tensor.layout != "contiguous":
            raise PlanError("Phase 0 only supports contiguous tensors")
        if tensor.stride is None and tensor.layout != "contiguous":
            raise PlanError("Phase 0 requires contiguous stride/layout evidence")

    spaces = {space.name: space.space for space in contract.memory_spaces}
    for name in required_tensors:
        if spaces.get(name) != "global":
            raise PlanError("Phase 0 requires global memory for tensor %s" % name)

