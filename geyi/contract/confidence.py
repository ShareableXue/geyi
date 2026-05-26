"""Explainable confidence policy for Phase -1 contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .model import Evidence, Rejection, Unknown


EVIDENCE_WEIGHTS: Dict[str, float] = {
    "user_annotation": 1.00,
    "runtime_harness": 0.95,
    "clang_semantic": 0.90,
    "source_ast": 0.75,
    "library_exact": 0.85,
    "library_similar": 0.45,
    "llm_interpretation": 0.35,
    "heuristic": 0.25,
}


@dataclass
class ConfidenceCap:
    id: str
    cap: float
    reason: str
    source: str


@dataclass
class ConfidenceReport:
    weighted_evidence_score: float
    caps: List[ConfidenceCap]
    final_confidence: float
    confidence_band: str
    recommended_path: str
    rule_covered: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "weighted_evidence_score": self.weighted_evidence_score,
            "caps": [cap.__dict__ for cap in self.caps],
            "final_confidence": self.final_confidence,
            "confidence_band": self.confidence_band,
            "recommended_path": self.recommended_path,
            "rule_covered": self.rule_covered,
            "notes": self.notes,
        }


def combine(scores: Sequence[Tuple[float, float]]) -> float:
    if not scores:
        return 0.10
    p_not = 1.0
    for score, weight in scores:
        bounded_score = max(0.0, min(1.0, score))
        bounded_weight = max(0.0, min(1.0, weight))
        p_not *= 1.0 - bounded_score * bounded_weight
    return 1.0 - p_not


def band_for(score: float) -> str:
    if score >= 0.95:
        return "certain"
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.20:
        return "low"
    return "rejected"


def evaluate_confidence(
    evidence: Sequence[Evidence],
    unknowns: Sequence[Unknown],
    rejections: Sequence[Rejection],
    rule_covered: bool,
) -> ConfidenceReport:
    positive_scores = []
    strongest_contradiction = 0.0
    for item in evidence:
        weight = item.weight
        if weight is None:
            weight = EVIDENCE_WEIGHTS.get(item.kind, 0.25)
        if item.contradicts:
            strongest_contradiction = max(strongest_contradiction, item.confidence)
        else:
            positive_scores.append((item.confidence, weight))

    evidence_score = combine(positive_scores)
    if strongest_contradiction:
        evidence_score *= 1.0 - strongest_contradiction

    caps = [ConfidenceCap("calibration.deterministic", 0.99, "deterministic rule calibration cap", "calibration")]
    for unknown in unknowns:
        cap = cap_for_unknown(unknown)
        if cap is not None:
            caps.append(cap)

    for rejection in rejections:
        if rejection.hard:
            caps.append(
                ConfidenceCap(
                    "hard_reject.%s" % rejection.feature,
                    0.10,
                    rejection.reason,
                    "unsupported_feature",
                )
            )
        else:
            caps.append(
                ConfidenceCap(
                    "soft_reject.%s" % rejection.feature,
                    0.45,
                    rejection.reason,
                    "unsupported_feature",
                )
            )

    final = min([evidence_score] + [cap.cap for cap in caps])
    final = round(final, 4)
    band = band_for(final)
    recommended_path = route_for(final, bool(rejections), bool(unknowns), rule_covered)
    notes = []
    if rejections:
        notes.append("rejections take precedence over positive evidence")
    if unknowns:
        notes.append("unknowns apply hard confidence caps")

    return ConfidenceReport(
        weighted_evidence_score=round(evidence_score, 4),
        caps=caps,
        final_confidence=final,
        confidence_band=band,
        recommended_path=recommended_path,
        rule_covered=rule_covered,
        notes=notes,
    )


def cap_for_unknown(unknown: Unknown) -> ConfidenceCap:
    uid = unknown.id
    if uid.startswith("missing_dtype"):
        return ConfidenceCap(uid, 0.60, unknown.text, "missing_context")
    if uid.startswith("missing_shape"):
        return ConfidenceCap(uid, 0.55, unknown.text, "missing_context")
    if uid.startswith("missing_stride") or uid.startswith("missing_layout"):
        return ConfidenceCap(uid, 0.80, unknown.text, "missing_context")
    if uid == "missing_launch":
        return ConfidenceCap(uid, 0.70, unknown.text, "missing_context")
    if uid == "black_box_only":
        return ConfidenceCap(uid, 0.40, unknown.text, "missing_context")
    if uid == "no_supported_intent":
        return ConfidenceCap(uid, 0.40, unknown.text, "unsupported_pattern")
    if uid == "effect_kind_unknown":
        return ConfidenceCap(uid, 0.55, unknown.text, "correctness_unknown")
    if uid == "control_flow_unknown":
        return ConfidenceCap(uid, 0.60, unknown.text, "correctness_unknown")
    if uid.startswith("template_gap"):
        return ConfidenceCap(uid, 0.70, unknown.text, "planning_gap")
    return ConfidenceCap(uid, 0.70, unknown.text, "unknown")


def route_for(score: float, has_rejection: bool, has_unknown: bool, rule_covered: bool) -> str:
    if has_rejection or score < 0.20:
        return "human"
    if score >= 0.95 and rule_covered and not has_unknown:
        return "rule"
    if score >= 0.75:
        return "template"
    if score >= 0.50:
        return "llm_plan"
    if score >= 0.20:
        return "llm_draft"
    return "human"
