"""Minimal SessionStore + ArtifactStore implementation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .contract.model import to_jsonable


@dataclass
class SessionStore:
    root: Path
    session_id: str

    @property
    def path(self) -> Path:
        return self.root / self.session_id

    @classmethod
    def create(cls, root: str = ".geyi/sessions") -> "SessionStore":
        root_path = Path(root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = "geyi_%s_%s" % (stamp, uuid4().hex[:6])
        store = cls(root=root_path, session_id=session_id)
        for relative in [
            "input/source_snapshot",
            "evidence",
            "logs",
        ]:
            (store.path / relative).mkdir(parents=True, exist_ok=True)
        return store

    def snapshot_inputs(self, source_path: Optional[str], spec_path: str) -> None:
        if source_path:
            source = Path(source_path)
            if source.exists():
                shutil.copy2(str(source), str(self.path / "input" / "source_snapshot" / source.name))
        spec = Path(spec_path)
        if spec.exists():
            shutil.copy2(str(spec), str(self.path / "input" / "geyi.yaml"))

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def write_log(self, name: str, text: str) -> Path:
        path = self.path / "logs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        return path

