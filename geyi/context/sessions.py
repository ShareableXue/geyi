"""Independent child session artifacts for Phase 2 tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from geyi.contract.model import to_jsonable


@dataclass
class ChildSession:
    root: Path
    task: str
    session_id: str

    @property
    def path(self) -> Path:
        return self.root / self.session_id

    @classmethod
    def create(cls, root: Path, task: str) -> "ChildSession":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in task)
        child = cls(root=root, task=safe_task, session_id="%s_%s_%s" % (safe_task, stamp, uuid4().hex[:6]))
        for relative in ["input", "prompt", "response", "output", "logs"]:
            (child.path / relative).mkdir(parents=True, exist_ok=True)
        child.write_json("metadata.json", {"task": safe_task, "session_id": child.session_id})
        return child

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_text(self, relative_path: str, text: str) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return path
