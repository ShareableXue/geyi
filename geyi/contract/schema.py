"""Minimal `geyi.yaml` schema loader for Phase -1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class SpecError(ValueError):
    pass


@dataclass
class GeyiSpec:
    path: Path
    version: int
    source: Dict[str, Any] = field(default_factory=dict)
    launch: Optional[Dict[str, Any]] = None
    symbols: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assumptions: List[Dict[str, Any]] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def entry(self) -> str:
        return str(self.source.get("entry") or self.raw.get("entry") or "")

    @property
    def source_files(self) -> List[str]:
        files = self.source.get("files") or []
        if isinstance(files, str):
            return [files]
        return [str(item) for item in files]

    @property
    def black_box(self) -> bool:
        return bool(self.source.get("black_box") or self.raw.get("black_box"))


def load_spec(path: str) -> GeyiSpec:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SpecError("geyi.yaml must contain a mapping at the top level")

    version = int(data.get("version", 1))
    source = data.get("source") or {}
    if not isinstance(source, dict):
        raise SpecError("source must be a mapping")

    launch = data.get("launch")
    if launch is not None and not isinstance(launch, dict):
        raise SpecError("launch must be a mapping when present")

    tensors = data.get("tensors") or {}
    if not isinstance(tensors, dict):
        raise SpecError("tensors must be a mapping")

    assumptions = normalize_assumptions(data.get("assumptions") or [])
    verification = data.get("verification") or {}
    if not isinstance(verification, dict):
        raise SpecError("verification must be a mapping when present")

    symbols = data.get("symbols") or {}
    if not isinstance(symbols, dict):
        raise SpecError("symbols must be a mapping when present")

    normalized_tensors = {}
    for name, tensor in tensors.items():
        if tensor is None:
            tensor = {}
        if not isinstance(tensor, dict):
            raise SpecError("tensor %s must be a mapping" % name)
        normalized_tensors[str(name)] = dict(tensor)

    return GeyiSpec(
        path=spec_path,
        version=version,
        source=source,
        launch=launch,
        symbols=symbols,
        tensors=normalized_tensors,
        assumptions=assumptions,
        verification=verification,
        raw=data,
    )


def normalize_assumptions(items: List[Any]) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        raise SpecError("assumptions must be a list")
    normalized = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            normalized.append({"id": "assumption_%d" % (index + 1), "text": item})
        elif isinstance(item, dict):
            normalized.append(dict(item))
        else:
            raise SpecError("assumption %d must be a string or mapping" % (index + 1))
    return normalized

