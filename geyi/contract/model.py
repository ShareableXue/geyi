"""Semantic Contract data model for the Phase -1 prototype."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SourceSpan:
    file: str
    start_line: int
    start_col: int = 1
    end_line: Optional[int] = None
    end_col: Optional[int] = None


@dataclass
class Evidence:
    id: str
    kind: str
    claim: str
    confidence: float
    span: Optional[SourceSpan] = None
    details: Dict[str, Any] = field(default_factory=dict)
    weight: Optional[float] = None
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)


@dataclass
class Assumption:
    id: str
    text: str
    required_for: List[str]
    source: str
    can_validate: bool = True


@dataclass
class Unknown:
    id: str
    text: str
    impact: str
    suggested_resolution: str


@dataclass
class Rejection:
    feature: str
    reason: str
    hard: bool
    suggestion: str


@dataclass
class TensorContract:
    name: str
    dtype: str
    shape: List[str]
    stride: Optional[List[str]]
    access: str
    layout: str = "unknown"
    alias_group: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class LaunchContract:
    grid: List[str]
    block: List[str]
    shared_memory: str = "0"
    stream: Optional[str] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class AccessPattern:
    tensor: str
    indices: List[str]
    affine: bool
    contiguous: Optional[bool]
    guards: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class ComputeIntent:
    kind: str
    subkind: str
    expression: Optional[str]
    axes: List[str]
    reduction_axes: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    access_patterns: List[AccessPattern] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class EffectContract:
    kind: str
    target: str
    operation: Optional[str]
    deterministic: Optional[bool]
    commutative: Optional[bool]
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SyncContract:
    kind: str
    location: Optional[SourceSpan]
    scope: str
    protects: List[str]
    required_for_correctness: bool
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ControlFlowContract:
    kind: str
    condition: Optional[str]
    affected_intents: List[str]
    data_dependent: bool
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class MemorySpaceContract:
    name: str
    space: str
    shape: Optional[List[str]]
    layout: str
    lifetime: Optional[str]
    used_by: List[str]
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SemanticContract:
    name: str
    source_files: List[str]
    entry: str
    tensors: Dict[str, TensorContract]
    launch: Optional[LaunchContract]
    intents: List[ComputeIntent]
    effects: List[EffectContract]
    sync: List[SyncContract]
    control_flow: List[ControlFlowContract]
    memory_spaces: List[MemorySpaceContract]
    cuda_features: List[str]
    evidence: List[Evidence]
    assumptions: List[Assumption]
    unknowns: List[Unknown]
    rejections: List[Rejection]
    confidence: float
    confidence_band: str
    recommended_path: str
    verification_required: str
    contract_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    def with_hash(self) -> "SemanticContract":
        payload = self.to_dict()
        payload["contract_hash"] = ""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.contract_hash = hashlib.sha256(encoded).hexdigest()[:16]
        return self


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

