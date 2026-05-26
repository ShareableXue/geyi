"""Planner JSON schema validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class PlannerSchemaError(ValueError):
    pass


REQUIRED_FIELDS = {
    "intent_confirmation",
    "selected_backend",
    "selected_template",
    "parameter_bindings",
    "required_assumptions",
    "risks",
    "repair_suggestions",
    "cannot_translate",
}
ALLOWED_BACKENDS = {"tilelang", "ascendc"}
ALLOWED_FIELDS = REQUIRED_FIELDS | {"annotation_request", "notes"}


@dataclass
class PlannerOutput:
    intent_confirmation: str
    selected_backend: str
    selected_template: str | None
    parameter_bindings: dict[str, Any]
    required_assumptions: list[str]
    risks: list[str]
    repair_suggestions: list[str]
    cannot_translate: bool
    annotation_request: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_confirmation": self.intent_confirmation,
            "selected_backend": self.selected_backend,
            "selected_template": self.selected_template,
            "parameter_bindings": self.parameter_bindings,
            "required_assumptions": self.required_assumptions,
            "risks": self.risks,
            "repair_suggestions": self.repair_suggestions,
            "cannot_translate": self.cannot_translate,
            "annotation_request": self.annotation_request,
            "notes": self.notes,
        }


def validate_planner_json(text: str) -> PlannerOutput:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerSchemaError("planner output is not valid JSON: %s" % exc) from exc
    return validate_planner_output(payload)


def validate_planner_output(payload: Any) -> PlannerOutput:
    if not isinstance(payload, dict):
        raise PlannerSchemaError("planner output must be a JSON object")
    missing = sorted(REQUIRED_FIELDS - set(payload))
    if missing:
        raise PlannerSchemaError("planner output missing required fields: %s" % ", ".join(missing))
    unknown = sorted(set(payload) - ALLOWED_FIELDS)
    if unknown:
        raise PlannerSchemaError("planner output contains unsupported fields: %s" % ", ".join(unknown))

    intent_confirmation = require_type(payload, "intent_confirmation", str)
    selected_backend = require_type(payload, "selected_backend", str)
    if selected_backend not in ALLOWED_BACKENDS:
        raise PlannerSchemaError("selected_backend must be one of: %s" % ", ".join(sorted(ALLOWED_BACKENDS)))

    selected_template = payload.get("selected_template")
    if selected_template is not None and not isinstance(selected_template, str):
        raise PlannerSchemaError("selected_template must be a string or null")
    parameter_bindings = require_type(payload, "parameter_bindings", dict)
    required_assumptions = require_string_list(payload, "required_assumptions")
    risks = require_string_list(payload, "risks")
    repair_suggestions = require_string_list(payload, "repair_suggestions")
    cannot_translate = require_type(payload, "cannot_translate", bool)
    annotation_request = payload.get("annotation_request")
    if annotation_request is not None and not isinstance(annotation_request, dict):
        raise PlannerSchemaError("annotation_request must be an object when present")
    notes = require_string_list(payload, "notes") if "notes" in payload else []

    if not cannot_translate and not selected_template:
        raise PlannerSchemaError("selected_template is required when cannot_translate is false")
    if cannot_translate and not (annotation_request or risks):
        raise PlannerSchemaError("cannot_translate output must include risks or annotation_request")

    return PlannerOutput(
        intent_confirmation=intent_confirmation,
        selected_backend=selected_backend,
        selected_template=selected_template,
        parameter_bindings=dict(parameter_bindings),
        required_assumptions=required_assumptions,
        risks=risks,
        repair_suggestions=repair_suggestions,
        cannot_translate=cannot_translate,
        annotation_request=annotation_request,
        notes=notes,
    )


def require_type(payload: dict[str, Any], key: str, expected_type):
    value = payload.get(key)
    if not isinstance(value, expected_type):
        raise PlannerSchemaError("%s must be %s" % (key, expected_type.__name__))
    return value


def require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlannerSchemaError("%s must be a list of strings" % key)
    return list(value)
