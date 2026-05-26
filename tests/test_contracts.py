from __future__ import annotations

from pathlib import Path

import pytest

from geyi.analysis import analyze
from geyi.planner.plan import create_deterministic_plan


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "contracts"


@pytest.mark.parametrize(
    ("fixture", "source", "kind", "subkind", "template"),
    [
        ("vector_add_1d", "vector_add.cu", "elementwise", "add", "tilelang.elementwise_binary_1d"),
        ("vector_mul_1d", "vector_mul.cu", "elementwise", "mul", "tilelang.elementwise_binary_1d"),
        ("elementwise_relu_1d", "relu.cu", "elementwise", "relu", "tilelang.elementwise_unary_1d"),
        ("elementwise_neg_1d", "neg.cu", "elementwise", "neg", "tilelang.elementwise_unary_1d"),
        ("copy_1d", "copy.cu", "copy", "copy", "tilelang.copy_cast_1d"),
        ("cast_1d", "cast.cu", "copy", "cast", "tilelang.copy_cast_1d"),
        ("transpose2d", "transpose.cu", "transpose", "2d_contiguous", "tilelang.transpose2d"),
        ("row_reduce_sum", "row_sum.cu", "reduce", "row_sum", "tilelang.row_reduce_sum"),
    ],
)
def test_phase1_contract_fixtures_route_to_rule(fixture, source, kind, subkind, template):
    result = analyze(
        str(FIXTURES / fixture / source),
        spec=str(FIXTURES / fixture / "geyi.yaml"),
        write_session=False,
    )
    contract = result.contract

    assert contract.confidence >= 0.95
    assert contract.recommended_path == "rule"
    assert contract.unknowns == []
    assert contract.rejections == []
    assert contract.intents[0].kind == kind
    assert contract.intents[0].subkind == subkind
    assert contract.effects[0].kind == "pure_store"
    assert contract.control_flow[0].kind in {"guarded_store", "straight_line"}

    plan = create_deterministic_plan(contract)
    assert plan.strategy == "rule"
    assert plan.backend == "tilelang"
    assert plan.template == template
    assert plan.parameters["operation"] in {"add", "mul", "relu", "neg", "copy", "cast", "transpose2d", "row_sum"}


def test_transpose_contract_records_2d_axes_and_accesses():
    result = analyze(
        str(FIXTURES / "transpose2d" / "transpose.cu"),
        spec=str(FIXTURES / "transpose2d" / "geyi.yaml"),
        write_session=False,
    )
    intent = result.contract.intents[0]
    assert intent.axes == ["row", "col"]
    assert intent.inputs == ["x"]
    assert intent.outputs == ["out"]
    assert any(pattern.tensor == "x" and pattern.contiguous for pattern in intent.access_patterns)


def test_row_reduce_contract_records_reduction_axis():
    result = analyze(
        str(FIXTURES / "row_reduce_sum" / "row_sum.cu"),
        spec=str(FIXTURES / "row_reduce_sum" / "geyi.yaml"),
        write_session=False,
    )
    intent = result.contract.intents[0]
    assert intent.kind == "reduce"
    assert intent.subkind == "row_sum"
    assert intent.reduction_axes == ["col"]
    assert intent.inputs == ["x"]
    assert intent.outputs == ["out"]
