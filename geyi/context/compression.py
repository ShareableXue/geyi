"""Structured compaction for Phase 2 context sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geyi.contract.model import SemanticContract


def compact_contract(contract: SemanticContract) -> Dict[str, Any]:
    intent = contract.intents[0] if contract.intents else None
    return {
        "name": contract.name,
        "entry": contract.entry,
        "contract_hash": contract.contract_hash,
        "confidence": contract.confidence,
        "recommended_path": contract.recommended_path,
        "intent": intent and {
            "kind": intent.kind,
            "subkind": intent.subkind,
            "expression": intent.expression,
            "inputs": list(intent.inputs),
            "outputs": list(intent.outputs),
            "axes": list(intent.axes),
            "reduction_axes": list(intent.reduction_axes),
        },
        "tensors": {name: tensor.__dict__ for name, tensor in contract.tensors.items()},
        "assumptions": [assumption.__dict__ for assumption in contract.assumptions],
        "unknowns": [unknown.__dict__ for unknown in contract.unknowns],
        "rejections": [rejection.__dict__ for rejection in contract.rejections],
        "evidence": [
            {
                "id": item.id,
                "kind": item.kind,
                "claim": item.claim,
                "confidence": item.confidence,
                "supports": list(item.supports),
            }
            for item in contract.evidence[:20]
        ],
    }


def source_snippet(path: str | None, max_chars: int = 6000) -> str:
    if not path:
        return ""
    source = Path(path)
    if not source.exists():
        return ""
    text = source.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n/* [GEYI: source snippet truncated] */\n"
