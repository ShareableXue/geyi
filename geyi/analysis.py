"""Phase -1 analysis pipeline: source + spec -> contract + report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import DEFAULT_SESSION_ROOT
from .contract.confidence import ConfidenceReport, evaluate_confidence
from .contract.model import (
    AccessPattern,
    Assumption,
    ComputeIntent,
    ControlFlowContract,
    EffectContract,
    Evidence,
    LaunchContract,
    MemorySpaceContract,
    Rejection,
    SemanticContract,
    SyncContract,
    TensorContract,
    Unknown,
)
from .contract.schema import GeyiSpec, load_spec
from .evidence.scanner import ScannerResult, scan_cuda_source
from .session import SessionStore


DETERMINISTIC_ELEMENTWISE_OPS = {"add", "mul", "relu", "neg", "exp"}
COMPOSITE_ELEMENTWISE_OPS = {"fused_add_relu"}
ELEMENTWISE_OPS = DETERMINISTIC_ELEMENTWISE_OPS | COMPOSITE_ELEMENTWISE_OPS
COPY_OPS = {"copy", "cast"}
TRANSPOSE_OPS = {"transpose2d"}
REDUCE_OPS = {"row_sum"}
SUPPORTED_RULE_OPS = DETERMINISTIC_ELEMENTWISE_OPS | COPY_OPS | TRANSPOSE_OPS | REDUCE_OPS
PURE_STORE_OPS = ELEMENTWISE_OPS | COPY_OPS | TRANSPOSE_OPS | REDUCE_OPS


@dataclass
class AnalysisResult:
    contract: SemanticContract
    confidence_report: ConfidenceReport
    scanner: ScannerResult
    session: Optional[SessionStore] = None


def analyze(
    source_path: Optional[str],
    spec: str,
    session_root: str = DEFAULT_SESSION_ROOT,
    write_session: bool = True,
) -> AnalysisResult:
    loaded_spec = load_spec(spec)
    entry = loaded_spec.entry
    scanner = scan_cuda_source(source_path, entry=entry, black_box=loaded_spec.black_box)
    contract, report = build_contract(source_path, loaded_spec, scanner)

    session = None
    if write_session:
        session = SessionStore.create(session_root)
        session.snapshot_inputs(source_path, spec)
        session.write_json("evidence/scanner.json", scanner.to_dict())
        session.write_json("contract.json", contract.to_dict())
        session.write_json("confidence_report.json", report.to_dict())
        session.write_log("info.log", "Phase -1 contract analysis completed for %s" % contract.entry)

    return AnalysisResult(contract=contract, confidence_report=report, scanner=scanner, session=session)


def build_contract(
    source_path: Optional[str],
    spec: GeyiSpec,
    scanner: ScannerResult,
) -> tuple[SemanticContract, ConfidenceReport]:
    evidence: List[Evidence] = []
    unknowns: List[Unknown] = []
    rejections = [
        Rejection(
            feature=item.feature,
            reason=item.reason,
            hard=item.hard,
            suggestion=item.suggestion,
        )
        for item in scanner.rejections
    ]

    for item in scanner.evidence:
        evidence.append(
            Evidence(
                id=str(item.get("id")),
                kind="source_ast",
                claim=str(item.get("claim")),
                confidence=float(item.get("confidence", 0.75)),
                details={key: value for key, value in item.items() if key not in {"id", "claim", "confidence"}},
            )
        )

    tensors = build_tensors(spec, scanner, evidence, unknowns)
    launch = build_launch(spec, evidence, unknowns)
    assumptions = build_assumptions(spec)

    source_files = []
    if source_path:
        source_files.append(str(source_path))
    else:
        source_files.extend(spec.source_files)

    if not scanner.source_available:
        unknowns.append(
            Unknown(
                id="black_box_only",
                text="source is unavailable or marked black-box",
                impact="correctness",
                suggested_resolution="provide source-available CUDA and build context",
            )
        )

    intents = build_intents(scanner, evidence, unknowns)
    effects = build_effects(scanner, evidence, unknowns)
    control_flow = build_control_flow(scanner, evidence, unknowns, intents)
    sync = build_sync(scanner, evidence)
    memory_spaces = build_memory_spaces(tensors, intents, effects, evidence)

    rule_covered = is_rule_covered(scanner, intents, effects, control_flow, memory_spaces, rejections)
    report = evaluate_confidence(evidence, unknowns, rejections, rule_covered)

    apply_confidence(report.final_confidence, intents, effects, control_flow, sync, memory_spaces)

    entry = scanner.entry or spec.entry or Path(source_path or "kernel").stem
    contract = SemanticContract(
        name=entry,
        source_files=source_files,
        entry=entry,
        tensors=tensors,
        launch=launch,
        intents=intents,
        effects=effects,
        sync=sync,
        control_flow=control_flow,
        memory_spaces=memory_spaces,
        cuda_features=scanner.cuda_features,
        evidence=evidence,
        assumptions=assumptions,
        unknowns=unknowns,
        rejections=rejections,
        confidence=report.final_confidence,
        confidence_band=report.confidence_band,
        recommended_path=report.recommended_path,
        verification_required=str(spec.verification.get("reference") or "gpu"),
    ).with_hash()
    return contract, report


def build_tensors(
    spec: GeyiSpec,
    scanner: ScannerResult,
    evidence: List[Evidence],
    unknowns: List[Unknown],
) -> Dict[str, TensorContract]:
    tensors: Dict[str, TensorContract] = {}
    if spec.tensors:
        evidence.append(
            Evidence(
                id="spec.tensors",
                kind="user_annotation",
                claim="tensor metadata provided by geyi.yaml",
                confidence=1.0,
                supports=["tensors", "memory_spaces"],
            )
        )

    for name, item in spec.tensors.items():
        dtype = str(item.get("dtype") or "unknown")
        shape = normalize_list(item.get("shape"))
        stride = normalize_optional_list(item.get("stride"))
        access = str(item.get("access") or infer_access(name, scanner))
        layout = str(item.get("layout") or infer_layout(stride))

        if dtype == "unknown":
            unknowns.append(
                Unknown(
                    id="missing_dtype.%s" % name,
                    text="tensor %s has no dtype in geyi.yaml" % name,
                    impact="correctness",
                    suggested_resolution="add tensors.%s.dtype" % name,
                )
            )
        if not shape:
            unknowns.append(
                Unknown(
                    id="missing_shape.%s" % name,
                    text="tensor %s has no shape in geyi.yaml" % name,
                    impact="correctness",
                    suggested_resolution="add tensors.%s.shape" % name,
                )
            )
        if stride is None and layout == "unknown":
            unknowns.append(
                Unknown(
                    id="missing_stride.%s" % name,
                    text="tensor %s has no stride/layout in geyi.yaml" % name,
                    impact="both",
                    suggested_resolution="add tensors.%s.stride or tensors.%s.layout" % (name, name),
                )
            )

        tensors[name] = TensorContract(
            name=name,
            dtype=dtype,
            shape=shape,
            stride=stride,
            access=access,
            layout=layout,
            alias_group=item.get("alias_group"),
            evidence=["spec.tensors"],
        )
    return tensors


def build_launch(spec: GeyiSpec, evidence: List[Evidence], unknowns: List[Unknown]) -> Optional[LaunchContract]:
    if not spec.launch:
        unknowns.append(
            Unknown(
                id="missing_launch",
                text="launch grid/block is missing from geyi.yaml",
                impact="correctness",
                suggested_resolution="add launch.grid and launch.block",
            )
        )
        return None
    evidence.append(
        Evidence(
            id="spec.launch",
            kind="user_annotation",
            claim="launch grid/block provided by geyi.yaml",
            confidence=1.0,
            supports=["launch"],
        )
    )
    grid = normalize_list(spec.launch.get("grid"))
    block = normalize_list(spec.launch.get("block"))
    if not grid or not block:
        unknowns.append(
            Unknown(
                id="missing_launch",
                text="launch grid/block is incomplete in geyi.yaml",
                impact="correctness",
                suggested_resolution="provide both launch.grid and launch.block",
            )
        )
    return LaunchContract(
        grid=grid,
        block=block,
        shared_memory=str(spec.launch.get("shared_memory", 0)),
        stream=spec.launch.get("stream"),
        evidence=["spec.launch"],
    )


def build_assumptions(spec: GeyiSpec) -> List[Assumption]:
    assumptions = []
    for index, item in enumerate(spec.assumptions):
        assumptions.append(
            Assumption(
                id=str(item.get("id") or "assumption_%d" % (index + 1)),
                text=str(item.get("text") or ""),
                required_for=[str(value) for value in item.get("required_for", ["correctness"])],
                source=str(item.get("source") or "user"),
                can_validate=bool(item.get("can_validate", True)),
            )
        )
    return assumptions


def build_intents(
    scanner: ScannerResult,
    evidence: List[Evidence],
    unknowns: List[Unknown],
) -> List[ComputeIntent]:
    if scanner.operation in ELEMENTWISE_OPS and scanner.write_tensor:
        intent_evidence = ["scan.intent", "scan.store"]
        expression = None
        if scanner.expression and scanner.write_tensor and scanner.write_index:
            expression = "%s[%s] = %s" % (scanner.write_tensor, scanner.write_index, scanner.expression)
        if scanner.operation in COMPOSITE_ELEMENTWISE_OPS:
            unknowns.append(
                Unknown(
                    id="template_gap.%s" % scanner.operation,
                    text="recognized %s but deterministic Phase 1 has no direct rule template" % scanner.operation,
                    impact="planning",
                    suggested_resolution="allow LLM planner to choose a constrained composite template",
                )
            )
        return [
            ComputeIntent(
                kind="elementwise",
                subkind=scanner.operation,
                expression=expression,
                axes=[scanner.idx_var or "idx"],
                inputs=scanner.read_tensors,
                outputs=[scanner.write_tensor],
                access_patterns=build_access_patterns(scanner),
                confidence=0.0,
                evidence=intent_evidence,
            )
        ]

    if scanner.operation in COPY_OPS and scanner.write_tensor:
        expression = None
        if scanner.expression and scanner.write_index:
            expression = "%s[%s] = %s" % (scanner.write_tensor, scanner.write_index, scanner.expression)
        return [
            ComputeIntent(
                kind="copy",
                subkind=scanner.operation,
                expression=expression,
                axes=[scanner.idx_var or "idx"],
                inputs=scanner.read_tensors,
                outputs=[scanner.write_tensor],
                access_patterns=build_access_patterns(scanner),
                confidence=0.0,
                evidence=["scan.intent", "scan.store"],
            )
        ]

    if scanner.operation in TRANSPOSE_OPS and scanner.write_tensor:
        return [
            ComputeIntent(
                kind="transpose",
                subkind="2d_contiguous",
                expression="%s[%s] = %s" % (scanner.write_tensor, scanner.write_index, scanner.expression),
                axes=[scanner.index_vars.get("row", "row"), scanner.index_vars.get("col", "col")],
                inputs=scanner.read_tensors,
                outputs=[scanner.write_tensor],
                access_patterns=build_access_patterns(scanner),
                confidence=0.0,
                evidence=["scan.intent", "scan.store"],
            )
        ]

    if scanner.operation in REDUCE_OPS and scanner.write_tensor:
        return [
            ComputeIntent(
                kind="reduce",
                subkind="row_sum",
                expression=scanner.expression,
                axes=[scanner.index_vars.get("linear") or scanner.index_vars.get("x") or "row"],
                reduction_axes=[scanner.reduction_axis or "col"],
                inputs=scanner.read_tensors,
                outputs=[scanner.write_tensor],
                access_patterns=build_access_patterns(scanner),
                confidence=0.0,
                evidence=["scan.intent", "scan.store"],
            )
        ]

    unknowns.append(
        Unknown(
            id="no_supported_intent",
            text="scanner could not classify the source as a Phase 1 supported intent",
            impact="correctness",
            suggested_resolution="provide a supported contiguous elementwise, copy/cast, transpose2d, or row-reduce kernel",
        )
    )
    evidence.append(
        Evidence(
            id="scan.intent_unknown",
            kind="source_ast",
            claim="no supported Phase -1 intent recognized",
            confidence=0.80,
            supports=["unknowns"],
        )
    )
    return [
        ComputeIntent(
            kind="unknown",
            subkind="unknown",
            expression=scanner.expression,
            axes=[],
            confidence=0.0,
            evidence=["scan.intent_unknown"],
        )
    ]


def build_effects(
    scanner: ScannerResult,
    evidence: List[Evidence],
    unknowns: List[Unknown],
) -> List[EffectContract]:
    if scanner.operation in PURE_STORE_OPS and scanner.write_tensor:
        return [
            EffectContract(
                kind="pure_store",
                target=scanner.write_tensor,
                operation=scanner.operation,
                deterministic=True,
                commutative=scanner.operation in {"add", "mul", "row_sum"},
                evidence=["scan.store", "scan.intent"],
            )
        ]
    unknowns.append(
        Unknown(
            id="effect_kind_unknown",
            text="write effect could not be proven as pure_store",
            impact="correctness",
            suggested_resolution="simplify the write pattern or add annotations in a later phase",
        )
    )
    return [
        EffectContract(
            kind="unknown",
            target=scanner.write_tensor or "unknown",
            operation=None,
            deterministic=None,
            commutative=None,
            evidence=["scan.intent_unknown"],
        )
    ]


def build_control_flow(
    scanner: ScannerResult,
    evidence: List[Evidence],
    unknowns: List[Unknown],
    intents: List[ComputeIntent],
) -> List[ControlFlowContract]:
    affected = ["%s.%s" % (intent.kind, intent.subkind) for intent in intents]
    if scanner.guarded:
        return [
            ControlFlowContract(
                kind="guarded_store",
                condition=scanner.guard_condition,
                affected_intents=affected,
                data_dependent=False,
                evidence=["scan.store"],
            )
        ]
    if scanner.write_tensor:
        return [
            ControlFlowContract(
                kind="straight_line",
                condition=None,
                affected_intents=affected,
                data_dependent=False,
                evidence=["scan.store"],
            )
        ]
    unknowns.append(
        Unknown(
            id="control_flow_unknown",
            text="control flow around writes could not be classified",
            impact="correctness",
            suggested_resolution="provide a simple guarded or straight-line store",
        )
    )
    return [
        ControlFlowContract(
            kind="unknown",
            condition=None,
            affected_intents=affected,
            data_dependent=True,
            evidence=["scan.intent_unknown"],
        )
    ]


def build_sync(scanner: ScannerResult, evidence: List[Evidence]) -> List[SyncContract]:
    sync = []
    if "syncthreads" in scanner.cuda_features:
        sync.append(
            SyncContract(
                kind="syncthreads",
                location=None,
                scope="block",
                protects=[],
                required_for_correctness=True,
                evidence=["scan.cuda_features"],
                confidence=0.0,
            )
        )
    return sync


def build_memory_spaces(
    tensors: Dict[str, TensorContract],
    intents: List[ComputeIntent],
    effects: List[EffectContract],
    evidence: List[Evidence],
) -> List[MemorySpaceContract]:
    used_by = ["%s.%s" % (intent.kind, intent.subkind) for intent in intents]
    used_by.extend(["effect.%s.%s" % (effect.kind, effect.target) for effect in effects])
    return [
        MemorySpaceContract(
            name=name,
            space="global",
            shape=tensor.shape,
            layout=tensor.layout,
            lifetime="kernel",
            used_by=used_by,
            evidence=tensor.evidence,
        )
        for name, tensor in tensors.items()
    ]


def build_access_patterns(scanner: ScannerResult) -> List[AccessPattern]:
    patterns: List[AccessPattern] = []
    guards = [scanner.guard_condition] if scanner.guard_condition else []
    for tensor in scanner.read_tensors:
        if not tensor:
            continue
        patterns.append(
            AccessPattern(
                tensor=tensor,
                indices=list(scanner.read_indices.get(tensor) or default_indices(scanner)),
                affine=True,
                contiguous=True,
                guards=guards,
                evidence=["scan.intent", "scan.store"],
            )
        )
    if scanner.write_tensor:
        patterns.append(
            AccessPattern(
                tensor=scanner.write_tensor,
                indices=[scanner.write_index] if scanner.write_index else default_indices(scanner),
                affine=True,
                contiguous=True,
                guards=guards,
                evidence=["scan.intent", "scan.store"],
            )
        )
    return patterns


def default_indices(scanner: ScannerResult) -> List[str]:
    if scanner.rank == 2:
        row = scanner.index_vars.get("row", "row")
        col = scanner.index_vars.get("col", "col")
        return [row, col]
    return [scanner.idx_var or "idx"]


def is_rule_covered(
    scanner: ScannerResult,
    intents: List[ComputeIntent],
    effects: List[EffectContract],
    control_flow: List[ControlFlowContract],
    memory_spaces: List[MemorySpaceContract],
    rejections: List[Rejection],
) -> bool:
    if rejections:
        return False
    if not intents:
        return False
    intent = intents[0]
    supported_intent = (
        (intent.kind == "elementwise" and intent.subkind in DETERMINISTIC_ELEMENTWISE_OPS)
        or (intent.kind == "copy" and intent.subkind in COPY_OPS)
        or (intent.kind == "transpose" and intent.subkind == "2d_contiguous")
        or (intent.kind == "reduce" and intent.subkind == "row_sum")
    )
    if not supported_intent:
        return False
    if not effects or effects[0].kind != "pure_store":
        return False
    if not control_flow or control_flow[0].kind not in {"guarded_store", "straight_line"}:
        return False
    if any(space.space != "global" for space in memory_spaces):
        return False
    if intent.kind in {"elementwise", "copy"}:
        return scanner.idx_var is not None and scanner.write_tensor is not None
    if intent.kind == "transpose":
        return scanner.rank == 2 and bool(scanner.index_vars.get("row")) and bool(scanner.index_vars.get("col"))
    if intent.kind == "reduce":
        return scanner.rank == 2 and scanner.reduction_axis is not None and scanner.write_tensor is not None
    return False


def apply_confidence(score: float, intents: List, effects: List, control_flow: List, sync: List, memory_spaces: List) -> None:
    for collection in [intents, effects, control_flow, sync, memory_spaces]:
        for item in collection:
            if hasattr(item, "confidence"):
                item.confidence = score
            elif isinstance(item, dict):
                item["confidence"] = score


def infer_access(name: str, scanner: ScannerResult) -> str:
    if name == scanner.write_tensor:
        return "write"
    if name in scanner.read_tensors:
        return "read"
    return "unknown"


def infer_layout(stride: Optional[List[str]]) -> str:
    if stride == ["1"]:
        return "contiguous"
    if stride:
        return "strided"
    return "unknown"


def normalize_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def normalize_optional_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    return normalize_list(value)
