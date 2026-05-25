"""Backend artifact models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from geyi.contract.model import to_jsonable


@dataclass
class GeneratedProject:
    root: str
    backend: str
    kernel_sources: List[str] = field(default_factory=list)
    host_sources: List[str] = field(default_factory=list)
    bindings: List[str] = field(default_factory=list)
    build_files: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], root: str) -> "GeneratedProject":
        return cls(
            root=root,
            backend=str(payload.get("backend") or ""),
            kernel_sources=[str(item) for item in payload.get("kernel_sources", [])],
            host_sources=[str(item) for item in payload.get("host_sources", [])],
            bindings=[str(item) for item in payload.get("bindings", [])],
            build_files=[str(item) for item in payload.get("build_files", [])],
            tests=[str(item) for item in payload.get("tests", [])],
            metadata=dict(payload.get("metadata") or {}),
            assumptions=[str(item) for item in payload.get("assumptions", [])],
        )


@dataclass
class CompiledArtifact:
    path: str
    artifact_hash: str
    backend: str
    compiler: str
    reused: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], path: str, reused: bool = False) -> "CompiledArtifact":
        return cls(
            path=path,
            artifact_hash=str(payload.get("artifact_hash") or ""),
            backend=str(payload.get("backend") or ""),
            compiler=str(payload.get("compiler") or ""),
            reused=reused,
            metadata=dict(payload.get("metadata") or {}),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]

