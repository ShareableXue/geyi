"""Verification report model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from geyi.contract.model import to_jsonable


class VerificationLevel(str, Enum):
    UNVERIFIED = "unverified"
    COMPILES_ONLY = "compiles_only"
    RUNS_ON_NPU = "runs_on_npu"
    GOLDEN = "golden"
    GPU_REFERENCE = "gpu_reference"
    PROPERTY_BASED = "property_based"
    PRODUCTION_VERIFIED = "production_verified"


@dataclass
class Coverage:
    shapes: List[Dict[str, int]]
    dtypes: List[str]
    strides: Optional[List[List[int]]] = None
    edge_cases: List[str] = field(default_factory=list)
    hardware: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    level: VerificationLevel
    contract_hash: str
    artifact_hash: str
    coverage: Coverage
    tolerance: Dict[str, float]
    max_abs_diff: Optional[float]
    max_rel_diff: Optional[float]
    assumptions: List[str]
    unknowns: List[str]
    strategy: str
    backend: str
    llm_used: bool
    passed: bool
    reproducible_commands: List[str] = field(default_factory=list)
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    cache: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = to_jsonable(self)
        payload["level"] = self.level.value
        return payload

