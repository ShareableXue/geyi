from __future__ import annotations

import pytest

from geyi.llm.schemas import PlannerSchemaError, validate_planner_output


def test_valid_planner_json_schema_passes():
    output = validate_planner_output(
        {
            "intent_confirmation": "elementwise fused add relu",
            "selected_backend": "tilelang",
            "selected_template": "tilelang.fused_add_relu_1d",
            "parameter_bindings": {},
            "required_assumptions": [],
            "risks": [],
            "repair_suggestions": [],
            "cannot_translate": False,
        }
    )

    assert output.selected_backend == "tilelang"
    assert output.selected_template == "tilelang.fused_add_relu_1d"
    assert output.cannot_translate is False


@pytest.mark.parametrize(
    "payload",
    [
        {"selected_backend": "tilelang"},
        {
            "intent_confirmation": "bad backend",
            "selected_backend": "cuda",
            "selected_template": "tilelang.fused_add_relu_1d",
            "parameter_bindings": {},
            "required_assumptions": [],
            "risks": [],
            "repair_suggestions": [],
            "cannot_translate": False,
        },
        {
            "intent_confirmation": "missing selected template",
            "selected_backend": "tilelang",
            "selected_template": None,
            "parameter_bindings": {},
            "required_assumptions": [],
            "risks": [],
            "repair_suggestions": [],
            "cannot_translate": False,
        },
    ],
)
def test_invalid_planner_json_is_rejected(payload):
    with pytest.raises(PlannerSchemaError):
        validate_planner_output(payload)
