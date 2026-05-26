"""Prompt layering for Phase 2 planner and repair sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geyi.context.compression import compact_contract
from geyi.contract.model import SemanticContract


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
STATIC_LAYERS = ["role.md", "contract_schema.md", "backend_constraints.md", "safety.md", "output_schema.md"]


def static_system_prompt() -> str:
    return "\n\n".join((PROMPT_DIR / name).read_text(encoding="utf-8").strip() for name in STATIC_LAYERS)


def planner_messages(
    contract: SemanticContract,
    source_snippet: str,
    available_templates: list[dict[str, Any]],
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    user_payload = {
        "contract": compact_contract(contract),
        "source_snippet": source_snippet,
        "available_templates": available_templates,
        "diagnostics": diagnostics or {},
    }
    return [
        {"role": "system", "content": static_system_prompt()},
        {"role": "user", "content": json.dumps(user_payload, indent=2, sort_keys=True)},
    ]


def schema_repair_messages(raw_output: str, schema_error: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": static_system_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "repair_invalid_planner_json",
                    "schema_error": schema_error,
                    "raw_output": raw_output,
                    "instruction": "Return only a corrected planner JSON object that passes the schema.",
                },
                indent=2,
                sort_keys=True,
            ),
        },
    ]


def compile_repair_messages(
    contract: SemanticContract,
    failed_plan: dict[str, Any],
    diagnostic: dict[str, Any],
    available_templates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": static_system_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "repair_compile_error",
                    "contract": compact_contract(contract),
                    "failed_plan": failed_plan,
                    "diagnostic": diagnostic,
                    "available_templates": available_templates,
                },
                indent=2,
                sort_keys=True,
            ),
        },
    ]
