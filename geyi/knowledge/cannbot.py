"""Local cannbot-skills tiling knowledge adapter."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class CannbotSource:
    path: str
    sha256: str
    anchors: List[str]
    claims: List[str]

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "anchors": list(self.anchors),
            "claims": list(self.claims),
        }


@dataclass
class CannbotKnowledge:
    family: str
    sources: List[CannbotSource] = field(default_factory=list)
    rules: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "family": self.family,
            "sources": [source.to_dict() for source in self.sources],
            "rules": dict(self.rules),
        }


def load_cannbot_knowledge(pattern: str, repo_root: Path | None = None) -> CannbotKnowledge:
    """Load a small, auditable slice of local cannbot tiling knowledge."""

    root = repo_root or default_repo_root()
    files = files_for_pattern(pattern, root)
    sources = [source_from_file(path) for path in files if path.exists()]
    return CannbotKnowledge(family=pattern, sources=sources, rules=rules_for_pattern(pattern, sources))


def default_repo_root() -> Path:
    # phase3.py lives in <dev>/geyi/geyi/geyi; parents[2] is <dev>/geyi.
    return Path(__file__).resolve().parents[2].parent


def files_for_pattern(pattern: str, root: Path) -> List[Path]:
    base = root / "cannbot-skills" / "ops"
    tiling = base / "ascendc-tiling-design"
    perf = base / "ascendc-performance-best-practices"
    if pattern in {"elementwise", "copy"}:
        return [
            tiling / "SKILL.md",
            tiling / "references" / "elewise" / "patterns.md",
            tiling / "references" / "elewise" / "tiling.md",
        ]
    if pattern == "reduce":
        return [
            tiling / "SKILL.md",
            tiling / "references" / "reduction" / "patterns.md",
            tiling / "references" / "reduction" / "tiling-fields.md",
            tiling / "references" / "reduction" / "algorithms.md",
        ]
    if pattern == "transpose":
        return [
            tiling / "SKILL.md",
            tiling / "references" / "conversion" / "patterns.md",
        ]
    if pattern == "matmul":
        return [
            tiling / "SKILL.md",
            tiling / "references" / "matmul" / "patterns.md",
            perf / "reference" / "matmul" / "guide.md",
        ]
    return [tiling / "SKILL.md", perf / "SKILL.md"]


def source_from_file(path: Path) -> CannbotSource:
    text = path.read_text(encoding="utf-8")
    anchors = extract_headings(text)
    claims = extract_claims(text)
    return CannbotSource(
        path=str(path),
        sha256=sha256_text(text),
        anchors=anchors[:8],
        claims=claims[:8],
    )


def extract_headings(text: str) -> List[str]:
    headings = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if match:
            headings.append(match.group(2).strip())
    return headings


def extract_claims(text: str) -> List[str]:
    claims = []
    keywords = [
        "多核",
        "UB",
        "对齐",
        "tile",
        "blockDim",
        "FullLoad",
        "ColSplit",
        "RowSplit",
        "double",
        "Buffer",
    ]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|") or len(line) > 180:
            continue
        if any(keyword in line for keyword in keywords):
            claims.append(line.lstrip("- "))
    return claims


def rules_for_pattern(pattern: str, sources: List[CannbotSource]) -> Dict[str, object]:
    if pattern in {"elementwise", "copy"}:
        return {
            "flatten_to_1d": True,
            "element_align": 512,
            "ub_align_bytes": 256,
            "min_bytes_per_core": 4096,
            "tile_lengths": [1024, 2048, 4096, 8192],
            "double_buffer": [1, 2],
            "tail_policy": "guarded_store",
            "confidence": confidence_from_sources(sources, base=0.82),
        }
    if pattern == "reduce":
        return {
            "axis_model": "AR_or_ARA",
            "row_reduce_modes": ["AR-FullLoad", "AR-ColSplit"],
            "col_tile_lengths": [256, 512, 1024, 2048],
            "multi_core": ["row_split", "group_reduce_when_R_large"],
            "alignment": {"rLengthAlign": "buffer_and_ub_offset", "rLength": "datacopy_and_reduce_count"},
            "confidence": confidence_from_sources(sources, base=0.76),
        }
    if pattern == "transpose":
        return {
            "model": "small_channel_or_2d_tile",
            "tile_shapes": [[16, 16], [16, 32], [32, 32]],
            "align_elements": 32,
            "repeat_limit": 255,
            "tail_policy": "2d_guard",
            "confidence": confidence_from_sources(sources, base=0.72),
        }
    return {"confidence": confidence_from_sources(sources, base=0.50)}


def confidence_from_sources(sources: List[CannbotSource], base: float) -> float:
    if not sources:
        return 0.20
    claim_count = sum(len(source.claims) for source in sources)
    source_bonus = min(0.08, 0.02 * len(sources))
    claim_bonus = min(0.06, 0.005 * claim_count)
    return min(0.95, round(base + source_bonus + claim_bonus, 3))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
