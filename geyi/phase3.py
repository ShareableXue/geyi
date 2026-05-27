"""Phase 3 performance hints and autotune search-space reporting."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from geyi.contract.model import SemanticContract, to_jsonable
from geyi.knowledge.cannbot import CannbotKnowledge, load_cannbot_knowledge
from geyi.planner.plan import TranslationPlan
from geyi.profiler import PerformanceReport, no_profile_report
from geyi.session import SessionStore
from geyi.verifier.report import VerificationReport


CANNBOT_TILING_SKILL = "../cannbot-skills/ops/ascendc-tiling-design/SKILL.md"
CANNBOT_PERF_SKILL = "../cannbot-skills/ops/ascendc-performance-best-practices/SKILL.md"


@dataclass
class OptimizationHint:
    id: str
    scope: str
    text: str
    parameters: Dict[str, Any]
    source: str
    confidence: float
    applies_when: List[str] = field(default_factory=list)
    source_digest: Optional[str] = None
    knowledge_claims: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class OptimizationReport:
    opt_level: str
    contract_hash: str
    verification_level: str
    verification_passed: bool
    status: str
    hints: List[OptimizationHint]
    sources: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def apply_phase3_optimization_artifacts(
    session: SessionStore,
    out_path: Path,
    contract: SemanticContract,
    plan: TranslationPlan,
    report: VerificationReport,
    opt_level: str = "none",
) -> Optional[OptimizationReport]:
    if opt_level in {"", "none", None}:
        return None
    if opt_level != "conservative":
        raise ValueError("unsupported opt-level: %s" % opt_level)

    optimization = build_optimization_report(contract, plan, report, opt_level)
    plan.optimization_hints = {
        "opt_level": opt_level,
        "status": optimization.status,
        "hints": [hint.to_dict() for hint in optimization.hints],
    }
    session.write_json("plan.json", plan.to_dict())
    session.write_json("optimization_hints.json", optimization.to_dict())
    session.write_log("optimization.log", render_optimization_log(optimization))
    copy_if_exists(session.path / "optimization_hints.json", out_path / "optimization_hints.json")
    copy_if_exists(session.path / "plan.json", out_path / "plan.json")
    return optimization


def build_optimization_report(
    contract: SemanticContract,
    plan: TranslationPlan,
    report: VerificationReport,
    opt_level: str,
) -> OptimizationReport:
    hints = conservative_hints_for_plan(plan)
    status = "eligible_after_verification" if report.passed else "blocked_until_correctness_passes"
    notes = [
        "Hints are advisory in Phase 3 and do not rewrite generated kernels.",
        "Autotune candidates must carry a passing verification report before selection.",
    ]
    if not report.passed:
        notes.append("Verification did not pass, so no performance change may be applied.")
    return OptimizationReport(
        opt_level=opt_level,
        contract_hash=contract.contract_hash,
        verification_level=report.level.value,
        verification_passed=report.passed,
        status=status,
        hints=hints,
        sources=report_sources_from_hints(hints),
        notes=notes,
    )


def conservative_hints_for_plan(plan: TranslationPlan) -> List[OptimizationHint]:
    operation = str(plan.parameters.get("operation") or "")
    pattern = str(plan.parameters.get("pattern") or "")
    family = family_for_plan(pattern, operation)
    knowledge = load_cannbot_knowledge(family)
    if pattern in {"elementwise", "copy"} or operation in {"add", "mul", "relu", "neg", "copy", "cast", "exp"}:
        rules = knowledge.rules
        return [
            OptimizationHint(
                id="tiling.linear_core_split",
                scope="tiling",
                text="Split contiguous 1D work evenly across AI cores and keep tail handling explicit.",
                parameters={
                    "axis": "n",
                    "tile_lengths": list(rules.get("tile_lengths", [1024, 2048, 4096])),
                    "element_align": rules.get("element_align", 512),
                    "ub_align_bytes": rules.get("ub_align_bytes", 256),
                    "min_bytes_per_core": rules.get("min_bytes_per_core", 4096),
                    "tail_policy": rules.get("tail_policy", "guarded_store"),
                },
                source=primary_source(knowledge),
                confidence=float(rules.get("confidence", 0.82)),
                applies_when=["rank == 1", "layout == contiguous", "verification_passed == true"],
                source_digest=primary_digest(knowledge),
                knowledge_claims=knowledge_claims(knowledge),
                evidence=knowledge_evidence(knowledge),
            ),
            OptimizationHint(
                id="buffer.double_buffer_vector",
                scope="ub_buffer",
                text="Use one input queue per operand, one output queue, and try double buffering under UB budget.",
                parameters={
                    "double_buffer": list(rules.get("double_buffer", [1, 2])),
                    "ub_budget_bytes": ub_budget_bytes(),
                    "ub_align_bytes": rules.get("ub_align_bytes", 256),
                },
                source=primary_source(knowledge),
                confidence=max(0.70, float(rules.get("confidence", 0.76)) - 0.06),
                applies_when=["vector pipeline", "no shared memory dependence"],
                source_digest=primary_digest(knowledge),
                knowledge_claims=knowledge_claims(knowledge),
                evidence=knowledge_evidence(knowledge),
            ),
        ]
    if operation == "transpose2d":
        rules = knowledge.rules
        return [
            OptimizationHint(
                id="tiling.transpose2d_tiles",
                scope="tiling",
                text="Use rectangular 2D tiles and preserve coalesced global reads before considering swizzled stores.",
                parameters={
                    "tile_shapes": list(rules.get("tile_shapes", [[16, 16], [16, 32], [32, 32]])),
                    "align_elements": rules.get("align_elements", 32),
                    "repeat_limit": rules.get("repeat_limit", 255),
                    "tail_policy": rules.get("tail_policy", "2d_guard"),
                },
                source=primary_source(knowledge),
                confidence=float(rules.get("confidence", 0.72)),
                applies_when=["rank == 2", "row_major_contiguous", "verification_passed == true"],
                source_digest=primary_digest(knowledge),
                knowledge_claims=knowledge_claims(knowledge),
                evidence=knowledge_evidence(knowledge),
            )
        ]
    if operation == "row_sum":
        rules = knowledge.rules
        return [
            OptimizationHint(
                id="tiling.row_reduce_chunks",
                scope="tiling",
                text="Split rows across cores and reduce contiguous columns in UB-sized chunks.",
                parameters={
                    "row_split": "core_balanced",
                    "axis_model": rules.get("axis_model", "AR_or_ARA"),
                    "row_reduce_modes": list(rules.get("row_reduce_modes", ["AR-FullLoad", "AR-ColSplit"])),
                    "col_tile_lengths": list(rules.get("col_tile_lengths", [256, 512, 1024])),
                    "alignment": dict(rules.get("alignment", {})),
                },
                source=primary_source(knowledge),
                confidence=float(rules.get("confidence", 0.74)),
                applies_when=["rank == 2", "reduction_axis == columns", "verification_passed == true"],
                source_digest=primary_digest(knowledge),
                knowledge_claims=knowledge_claims(knowledge),
                evidence=knowledge_evidence(knowledge),
            )
        ]
    return [
        OptimizationHint(
            id="phase3.no_conservative_hint",
            scope="none",
            text="No conservative Phase 3 hint is available for this operation yet.",
            parameters={},
            source=CANNBOT_PERF_SKILL,
            confidence=0.0,
            applies_when=[],
            source_digest=None,
            evidence=knowledge_evidence(knowledge),
        )
    ]


def build_search_space(plan: TranslationPlan, search_space: str = "small") -> Dict[str, Any]:
    if search_space != "small":
        raise ValueError("unsupported search-space: %s" % search_space)
    hints = conservative_hints_for_plan(plan)
    candidates = combined_vector_candidates(plan, hints)
    if not candidates:
        candidates = generic_candidates(plan, hints)
    return {
        "name": search_space,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "policy": "correctness_first",
        "measurement": {
            "required_before_selection": True,
            "recommended_profiler": "msprof op --kernel-name=<kernel> <operator command>",
            "generated_operator_profile": "geyi tune --backend ascendc --target cann --profile-generated",
        },
    }


def combined_vector_candidates(plan: TranslationPlan, hints: List[OptimizationHint]) -> List[Dict[str, Any]]:
    linear = next((hint for hint in hints if hint.id == "tiling.linear_core_split"), None)
    buffering = next((hint for hint in hints if hint.id == "buffer.double_buffer_vector"), None)
    if linear is None or buffering is None:
        return []
    candidates = []
    candidate_id = 1
    for tile_length in linear.parameters.get("tile_lengths", []):
        for double_buffer in buffering.parameters.get("double_buffer", [1]):
            candidates.append(
                {
                    "id": "candidate_%02d" % candidate_id,
                    "hint_id": "%s+%s" % (linear.id, buffering.id),
                    "hint_confidence": round(min(linear.confidence, buffering.confidence), 3),
                    "parameters": {"tile_length": int(tile_length), "double_buffer": int(double_buffer)},
                    "constraints": constraints_for_hint(linear, plan),
                    "estimated_ub_bytes": estimate_ub_bytes(plan, int(tile_length), int(double_buffer)),
                    "status": "requires_verified_measurement",
                }
            )
            candidate_id += 1
            if len(candidates) >= 6:
                return candidates
    return candidates


def generic_candidates(plan: TranslationPlan, hints: List[OptimizationHint]) -> List[Dict[str, Any]]:
    candidates = []
    candidate_id = 1
    for hint in hints:
        params = hint.parameters
        for tile_length in params.get("tile_lengths", params.get("col_tile_lengths", [None])):
            double_buffers = params.get("double_buffer", [None])
            for double_buffer in double_buffers:
                candidate = {
                    "id": "candidate_%02d" % candidate_id,
                    "hint_id": hint.id,
                    "hint_confidence": hint.confidence,
                    "parameters": {},
                    "constraints": constraints_for_hint(hint, plan),
                    "estimated_ub_bytes": None,
                    "status": "requires_verified_measurement",
                }
                if tile_length is not None:
                    candidate["parameters"]["tile_length"] = tile_length
                    candidate["estimated_ub_bytes"] = estimate_ub_bytes(plan, int(tile_length), double_buffer)
                if double_buffer is not None:
                    candidate["parameters"]["double_buffer"] = double_buffer
                if params.get("tile_shapes"):
                    candidate["parameters"]["tile_shape"] = params["tile_shapes"][0]
                    candidate["estimated_ub_bytes"] = estimate_tile_shape_ub_bytes(plan, params["tile_shapes"][0])
                candidates.append(candidate)
                candidate_id += 1
                if len(candidates) >= 6:
                    break
            if len(candidates) >= 6:
                break
    return candidates


def build_tuning_report(
    contract: SemanticContract,
    plan: TranslationPlan,
    verification_report: VerificationReport,
    search_space: str,
    performance_report: Optional[PerformanceReport] = None,
) -> Dict[str, Any]:
    space = build_search_space(plan, search_space=search_space)
    performance = performance_report or no_profile_report()
    candidate_reports = []
    for candidate in space["candidates"]:
        candidate_reports.append(
            {
                "candidate": candidate,
                "verification": verification_report.to_dict(),
                "performance": {
                    "measured": performance.status == "measured",
                    "baseline_report_status": performance.status,
                    "reason": "candidate-specific execution is not implemented; baseline msprof data is attached when supplied.",
                },
                "selectable": verification_report.passed,
            }
        )
    return {
        "phase": "phase3b",
        "contract_hash": contract.contract_hash,
        "strategy": plan.strategy,
        "backend": plan.backend,
        "search_space": space,
        "candidate_reports": candidate_reports,
        "selected_candidate": candidate_reports[0]["candidate"]["id"] if candidate_reports and verification_report.passed else None,
        "verification_required_before_selection": True,
        "baseline_verification": verification_report.to_dict(),
        "performance_report": performance.to_dict(),
    }


def write_tuning_report(out_path: Path, session: SessionStore, report: Dict[str, Any]) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (out_path / "tuning_report.json").write_text(payload, encoding="utf-8")
    session.write_json("tuning_report.json", report)
    session.write_json("performance_report.json", report["performance_report"])
    session.write_log("tune.log", "Phase 3b tuning report completed with %d candidates" % report["search_space"]["candidate_count"])


def render_optimization_log(report: OptimizationReport) -> str:
    return "\n".join(
        [
            "Phase 3 conservative optimization hints",
            "opt_level=%s" % report.opt_level,
            "status=%s" % report.status,
            "verification_passed=%s" % report.verification_passed,
            "hint_count=%d" % len(report.hints),
        ]
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def family_for_plan(pattern: str, operation: str) -> str:
    if operation == "transpose2d" or pattern == "transpose":
        return "transpose"
    if operation == "row_sum" or pattern == "reduce":
        return "reduce"
    if pattern in {"elementwise", "copy"}:
        return "elementwise"
    return pattern or operation or "unknown"


def primary_source(knowledge: CannbotKnowledge) -> str:
    return knowledge.sources[0].path if knowledge.sources else CANNBOT_TILING_SKILL


def primary_digest(knowledge: CannbotKnowledge) -> Optional[str]:
    return knowledge.sources[0].sha256 if knowledge.sources else None


def knowledge_claims(knowledge: CannbotKnowledge) -> List[str]:
    claims: List[str] = []
    for source in knowledge.sources:
        claims.extend(source.claims[:3])
    return claims[:8]


def knowledge_evidence(knowledge: CannbotKnowledge) -> List[Dict[str, Any]]:
    return [
        {
            "kind": "cannbot_tiling_knowledge",
            "family": knowledge.family,
            "path": source.path,
            "sha256": source.sha256,
            "anchors": source.anchors,
            "claims": source.claims[:4],
        }
        for source in knowledge.sources
    ]


def report_sources_from_hints(hints: List[OptimizationHint]) -> List[str]:
    sources = []
    for hint in hints:
        if hint.source and hint.source not in sources:
            sources.append(hint.source)
        for evidence in hint.evidence:
            path = str(evidence.get("path") or "")
            if path and path not in sources:
                sources.append(path)
    return sources or [CANNBOT_TILING_SKILL, CANNBOT_PERF_SKILL]


def constraints_for_hint(hint: OptimizationHint, plan: TranslationPlan) -> Dict[str, Any]:
    params = hint.parameters
    return {
        "correctness": "candidate must pass the same verification report coverage before selection",
        "ub_budget_bytes": params.get("ub_budget_bytes", ub_budget_bytes()),
        "tail_policy": params.get("tail_policy"),
        "rank": plan.parameters.get("rank"),
        "dtypes": dict(plan.parameters.get("dtypes") or {}),
    }


def estimate_ub_bytes(plan: TranslationPlan, tile_length: int, double_buffer: Optional[int]) -> int:
    dtypes = dict(plan.parameters.get("dtypes") or {})
    input_count = len(plan.parameters.get("inputs") or [])
    output_count = 1 if plan.parameters.get("output") else 0
    dtype_size = max([dtype_bytes(dtype) for dtype in dtypes.values()] or [4])
    buffers = input_count + output_count
    stages = int(double_buffer or 1)
    return align_up(tile_length * dtype_size * buffers * stages, 256)


def estimate_tile_shape_ub_bytes(plan: TranslationPlan, tile_shape: List[int]) -> int:
    element_count = 1
    for value in tile_shape:
        element_count *= int(value)
    dtypes = dict(plan.parameters.get("dtypes") or {})
    dtype_size = max([dtype_bytes(dtype) for dtype in dtypes.values()] or [4])
    return align_up(element_count * dtype_size * 2, 256)


def dtype_bytes(dtype: str) -> int:
    return {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "int32": 4,
        "int64": 8,
        "int8": 1,
        "int4": 1,
    }.get(str(dtype), 4)


def ub_budget_bytes() -> Dict[str, int]:
    return {"dav-2201": 196608, "dav-3510": 253952}


def align_up(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment
