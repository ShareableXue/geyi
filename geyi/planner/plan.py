"""Deterministic rule planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geyi.contract.model import ComputeIntent, SemanticContract, TensorContract, to_jsonable


DETERMINISTIC_BACKEND = "tilelang"
PHASE0_BACKEND = DETERMINISTIC_BACKEND
PHASE0_TEMPLATE = "tilelang.elementwise_binary_1d"
SUPPORTED_BACKENDS = {"tilelang", "ascendc"}
SUPPORTED_TARGETS = {"local_cpu", "scaffold", "cann"}

ELEMENTWISE_UNARY = {"relu", "neg", "exp"}
ELEMENTWISE_BINARY = {"add", "mul"}
COPY_CAST = {"copy", "cast"}
SUPPORTED_DTYPES = {"float32", "float16", "int32", "int64"}


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
    """Build the original Phase 0 rule plan for 1D vector add only."""

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
            "phase": "phase0",
            "pattern": "elementwise",
            "operation": "add",
            "inputs": list(intent.inputs),
            "output": output,
            "rank": 1,
            "shape": list(shape),
            "input_shapes": {name: list(contract.tensors[name].shape) for name in intent.inputs},
            "output_shape": list(output_tensor.shape),
            "dtypes": tensor_dtypes(contract, list(intent.inputs) + [output]),
            "dtype": output_tensor.dtype,
            "index_axis": intent.axes[0] if intent.axes else "idx",
            "tolerance": default_tolerance(),
            "coverage_cases": phase0_coverage_cases(),
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


def create_deterministic_plan(
    contract: SemanticContract,
    backend: str = DETERMINISTIC_BACKEND,
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
) -> TranslationPlan:
    """Build a Phase 1 deterministic rule plan."""

    backend = normalize_backend(backend)
    target = normalize_target(target, backend)
    validate_phase1_contract(contract)
    intent = contract.intents[0]
    operation = operation_for_intent(intent)
    output = intent.outputs[0]
    required_tensors = list(intent.inputs) + [output]

    return TranslationPlan(
        contract_hash=contract.contract_hash,
        strategy="rule",
        backend=backend,
        template=template_for_intent(intent, backend),
        operator_entry=contract.entry,
        parameters={
            "phase": "phase1",
            "target": target,
            "npu_arch": npu_arch,
            "pattern": intent.kind,
            "operation": operation,
            "inputs": list(intent.inputs),
            "output": output,
            "rank": rank_for_intent(contract, intent),
            "shape": list(contract.tensors[output].shape),
            "input_shapes": {name: list(contract.tensors[name].shape) for name in intent.inputs},
            "output_shape": list(contract.tensors[output].shape),
            "dtypes": tensor_dtypes(contract, required_tensors),
            "dtype": contract.tensors[output].dtype,
            "axes": list(intent.axes),
            "reduction_axes": list(intent.reduction_axes),
            "tolerance": default_tolerance(),
            "coverage_cases": coverage_cases_for_intent(intent),
        },
        optimization_hints={},
        required_assumptions=[assumption.id for assumption in contract.assumptions],
        expected_verification="golden",
        confidence=contract.confidence,
        notes=[
            "Phase 1 deterministic rule path.",
            "Supported surface: 1D unary/binary/copy/cast, 2D transpose, row-wise reduce sum.",
            note_for_target(backend, target),
        ],
    )


def validate_phase0_contract(contract: SemanticContract) -> None:
    validate_common_contract(contract, phase="Phase 0")
    intent = contract.intents[0]
    if intent.kind != "elementwise" or intent.subkind != "add":
        raise PlanError("Phase 0 only supports 1D contiguous elementwise.add")
    if len(intent.inputs) != 2 or len(intent.outputs) != 1:
        raise PlanError("Phase 0 elementwise.add requires two inputs and one output")
    validate_required_tensors(contract, intent, rank=1, allowed_dtypes={"float32"}, phase="Phase 0")


def validate_phase1_contract(contract: SemanticContract) -> None:
    validate_common_contract(contract, phase="Phase 1")
    intent = contract.intents[0]

    if intent.kind == "elementwise" and intent.subkind in ELEMENTWISE_BINARY:
        if len(intent.inputs) != 2 or len(intent.outputs) != 1:
            raise PlanError("Phase 1 binary elementwise requires two inputs and one output")
        validate_required_tensors(contract, intent, rank=1, allowed_dtypes={"float32", "float16"}, phase="Phase 1")
        return

    if intent.kind == "elementwise" and intent.subkind in ELEMENTWISE_UNARY:
        if len(intent.inputs) != 1 or len(intent.outputs) != 1:
            raise PlanError("Phase 1 unary elementwise requires one input and one output")
        validate_required_tensors(contract, intent, rank=1, allowed_dtypes={"float32", "float16"}, phase="Phase 1")
        return

    if intent.kind == "copy" and intent.subkind in COPY_CAST:
        if len(intent.inputs) != 1 or len(intent.outputs) != 1:
            raise PlanError("Phase 1 copy/cast requires one input and one output")
        validate_required_tensors(contract, intent, rank=1, allowed_dtypes=SUPPORTED_DTYPES, phase="Phase 1")
        return

    if intent.kind == "transpose" and intent.subkind == "2d_contiguous":
        if len(intent.inputs) != 1 or len(intent.outputs) != 1:
            raise PlanError("Phase 1 transpose2d requires one input and one output")
        validate_required_tensors(contract, intent, rank=2, allowed_dtypes={"float32", "float16"}, phase="Phase 1")
        return

    if intent.kind == "reduce" and intent.subkind == "row_sum":
        if len(intent.inputs) != 1 or len(intent.outputs) != 1:
            raise PlanError("Phase 1 row_reduce_sum requires one input and one output")
        validate_reduce_tensors(contract, intent)
        return

    raise PlanError("Phase 1 unsupported deterministic intent: %s.%s" % (intent.kind, intent.subkind))


def validate_common_contract(contract: SemanticContract, phase: str) -> None:
    if contract.confidence < 0.95:
        raise PlanError("%s requires contract confidence >= 0.95" % phase)
    if contract.recommended_path != "rule":
        raise PlanError("%s requires recommended_path=rule" % phase)
    if contract.unknowns:
        raise PlanError("%s requires no correctness unknowns" % phase)
    if contract.rejections:
        raise PlanError("%s requires no rejected CUDA features" % phase)
    if not contract.contract_hash:
        raise PlanError("%s requires a stable contract_hash" % phase)
    if not contract.intents:
        raise PlanError("%s requires one recognized intent" % phase)
    if not contract.effects or contract.effects[0].kind != "pure_store":
        raise PlanError("%s requires a pure_store effect" % phase)
    if not contract.control_flow or contract.control_flow[0].kind not in {"guarded_store", "straight_line"}:
        raise PlanError("%s requires guarded or straight-line store control flow" % phase)


def validate_required_tensors(
    contract: SemanticContract,
    intent: ComputeIntent,
    rank: int,
    allowed_dtypes: set[str],
    phase: str,
) -> None:
    required_tensors = list(intent.inputs) + list(intent.outputs)
    missing = [name for name in required_tensors if name not in contract.tensors]
    if missing:
        raise PlanError("%s missing tensor contracts: %s" % (phase, ", ".join(missing)))

    for name in required_tensors:
        tensor = contract.tensors[name]
        if tensor.dtype not in allowed_dtypes:
            raise PlanError("%s unsupported dtype for %s: %s" % (phase, name, tensor.dtype))
        if len(tensor.shape) != rank:
            raise PlanError("%s requires rank-%d tensor %s" % (phase, rank, name))
        if not is_contiguous_tensor(tensor, rank):
            raise PlanError("%s requires contiguous tensor %s" % (phase, name))

    spaces = {space.name: space.space for space in contract.memory_spaces}
    for name in required_tensors:
        if spaces.get(name) != "global":
            raise PlanError("%s requires global memory for tensor %s" % (phase, name))


def validate_reduce_tensors(contract: SemanticContract, intent: ComputeIntent) -> None:
    missing = [name for name in list(intent.inputs) + list(intent.outputs) if name not in contract.tensors]
    if missing:
        raise PlanError("Phase 1 missing tensor contracts: %s" % ", ".join(missing))
    input_tensor = contract.tensors[intent.inputs[0]]
    output_tensor = contract.tensors[intent.outputs[0]]
    if len(input_tensor.shape) != 2 or len(output_tensor.shape) != 1:
        raise PlanError("Phase 1 row_reduce_sum requires rank-2 input and rank-1 output")
    for name, tensor, rank in [
        (intent.inputs[0], input_tensor, 2),
        (intent.outputs[0], output_tensor, 1),
    ]:
        if tensor.dtype not in {"float32", "float16"}:
            raise PlanError("Phase 1 unsupported dtype for %s: %s" % (name, tensor.dtype))
        if not is_contiguous_tensor(tensor, rank):
            raise PlanError("Phase 1 requires contiguous tensor %s" % name)
    spaces = {space.name: space.space for space in contract.memory_spaces}
    for name in list(intent.inputs) + list(intent.outputs):
        if spaces.get(name) != "global":
            raise PlanError("Phase 1 requires global memory for tensor %s" % name)


def is_contiguous_tensor(tensor: TensorContract, rank: int) -> bool:
    if tensor.layout in {"contiguous", "row_major", "row_major_contiguous"}:
        return True
    stride = [str(item) for item in tensor.stride] if tensor.stride is not None else None
    if rank == 1:
        return stride == ["1"]
    if rank == 2:
        return bool(stride and len(stride) == 2 and stride[-1] == "1")
    return False


def operation_for_intent(intent: ComputeIntent) -> str:
    if intent.kind in {"elementwise", "copy"}:
        return intent.subkind
    if intent.kind == "transpose":
        return "transpose2d"
    if intent.kind == "reduce":
        return "row_sum"
    raise PlanError("unsupported intent operation: %s.%s" % (intent.kind, intent.subkind))


def template_for_intent(intent: ComputeIntent, backend: str = DETERMINISTIC_BACKEND) -> str:
    prefix = backend
    if intent.kind == "elementwise" and intent.subkind in ELEMENTWISE_BINARY:
        return "%s.elementwise_binary_1d" % prefix
    if intent.kind == "elementwise" and intent.subkind in ELEMENTWISE_UNARY:
        return "%s.elementwise_unary_1d" % prefix
    if intent.kind == "copy":
        return "%s.copy_cast_1d" % prefix
    if intent.kind == "transpose":
        return "%s.transpose2d" % prefix
    if intent.kind == "reduce":
        return "%s.row_reduce_sum" % prefix
    raise PlanError("no template for intent: %s.%s" % (intent.kind, intent.subkind))


def normalize_backend(backend: str) -> str:
    backend = str(backend or DETERMINISTIC_BACKEND)
    if backend == "auto":
        backend = DETERMINISTIC_BACKEND
    if backend not in SUPPORTED_BACKENDS:
        raise PlanError("unsupported backend: %s" % backend)
    return backend


def normalize_target(target: str, backend: str) -> str:
    target = str(target or "auto")
    if target == "auto":
        return "local_cpu" if backend == "tilelang" else "scaffold"
    if target not in SUPPORTED_TARGETS:
        raise PlanError("unsupported target: %s" % target)
    if backend == "tilelang" and target != "local_cpu":
        raise PlanError("tilelang backend currently supports target=local_cpu only in Phase 1")
    if backend == "ascendc" and target == "local_cpu":
        raise PlanError("ascendc backend supports target=scaffold or target=cann")
    return target


def note_for_target(backend: str, target: str) -> str:
    if backend == "tilelang":
        return "Local golden verification is used; no NPU execution is claimed."
    if target == "scaffold":
        return "AscendC project scaffold is generated for hardware validation; no NPU execution is claimed."
    return "AscendC/CANN hardware execution is requested; verification depends on the generated run.sh result."


def rank_for_intent(contract: SemanticContract, intent: ComputeIntent) -> int:
    if intent.kind in {"transpose", "reduce"} and intent.inputs:
        return len(contract.tensors[intent.inputs[0]].shape)
    if intent.outputs:
        return len(contract.tensors[intent.outputs[0]].shape)
    if intent.inputs:
        return len(contract.tensors[intent.inputs[0]].shape)
    return 0


def tensor_dtypes(contract: SemanticContract, names: List[str]) -> Dict[str, str]:
    return {name: contract.tensors[name].dtype for name in names if name in contract.tensors}


def default_tolerance() -> Dict[str, float]:
    return {"atol": 1e-5, "rtol": 1e-5}


def phase0_coverage_cases() -> List[Dict[str, int | str]]:
    return [
        {"case": "basic", "n": 1024},
        {"case": "tail", "n": 1025},
        {"case": "small", "n": 1},
        {"case": "zero", "n": 0},
    ]


def coverage_cases_for_intent(intent: ComputeIntent) -> List[Dict[str, int | str]]:
    if intent.kind in {"elementwise", "copy"}:
        return [
            {"case": "basic", "n": 1024},
            {"case": "tail", "n": 1025},
            {"case": "single", "n": 1},
            {"case": "zero", "n": 0},
            {"case": "tiny", "n": 2},
            {"case": "block_edge", "n": 256},
            {"case": "post_block_tail", "n": 257},
        ]
    if intent.kind == "transpose":
        return [
            {"case": "rect", "rows": 32, "cols": 64},
            {"case": "tail_rect", "rows": 31, "cols": 65},
            {"case": "square", "rows": 8, "cols": 8},
            {"case": "skinny", "rows": 2, "cols": 17},
            {"case": "single", "rows": 1, "cols": 1},
        ]
    if intent.kind == "reduce":
        return [
            {"case": "rect", "rows": 32, "cols": 64},
            {"case": "tail_cols", "rows": 31, "cols": 65},
            {"case": "single", "rows": 1, "cols": 1},
            {"case": "empty_cols", "rows": 4, "cols": 0},
            {"case": "empty_rows", "rows": 0, "cols": 8},
        ]
    raise PlanError("no coverage matrix for intent: %s.%s" % (intent.kind, intent.subkind))
