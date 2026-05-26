from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from geyi.phase1 import run_phase1


ROOT = Path(__file__).resolve().parents[1]

E2E_CASES = [
    ("vector_add", "examples/vector_add/vector_add.cu", "examples/vector_add/geyi.yaml", "add"),
    ("elementwise_mul", "examples/elementwise_mul/mul.cu", "examples/elementwise_mul/geyi.yaml", "mul"),
    ("elementwise_relu", "examples/elementwise_relu/relu.cu", "examples/elementwise_relu/geyi.yaml", "relu"),
    ("elementwise_neg", "examples/elementwise_neg/neg.cu", "examples/elementwise_neg/geyi.yaml", "neg"),
    ("copy1d", "examples/copy1d/copy.cu", "examples/copy1d/geyi.yaml", "copy"),
    ("cast1d", "examples/cast1d/cast.cu", "examples/cast1d/geyi.yaml", "cast"),
    ("transpose2d", "examples/transpose2d/transpose.cu", "examples/transpose2d/geyi.yaml", "transpose2d"),
    ("row_reduce_sum", "examples/row_reduce_sum/row_sum.cu", "examples/row_reduce_sum/geyi.yaml", "row_sum"),
]


@pytest.mark.parametrize(("name", "source", "spec", "operation"), E2E_CASES)
def test_phase1_deterministic_e2e_patterns(tmp_path, name, source, spec, operation):
    result = run_phase1(
        source,
        spec=spec,
        out=str(tmp_path / "out" / name),
        session_root=str(tmp_path / "sessions"),
        reproducible_command="geyi run %s --spec %s" % (source, spec),
    )

    report = result.verification_report
    assert result.plan.parameters["phase"] == "phase1"
    assert result.plan.parameters["operation"] == operation
    assert report.level.value == "golden"
    assert report.passed
    assert report.max_abs_diff == 0.0
    assert report.coverage.hardware == ["local_cpu"]
    assert report.llm_used is False
    assert len(report.case_results) >= 5
    assert (result.analysis.session.path / "contract.json").exists()
    assert (result.analysis.session.path / "plan.json").exists()
    assert (result.analysis.session.path / "generated" / "metadata.json").exists()
    assert (result.analysis.session.path / "build" / "artifact_metadata.json").exists()
    assert (result.analysis.session.path / "verification_report.json").exists()


def test_phase1_e2e_matrix_has_at_least_30_passes(tmp_path):
    total = 0
    for name, source, spec, _operation in E2E_CASES:
        result = run_phase1(
            source,
            spec=spec,
            out=str(tmp_path / "out" / name),
            session_root=str(tmp_path / "sessions"),
        )
        assert result.verification_report.passed
        total += len(result.verification_report.case_results)

    assert total >= 30


def test_phase1_binary_coverage_contains_required_tail_cases(tmp_path):
    result = run_phase1(
        "examples/elementwise_mul/mul.cu",
        spec="examples/elementwise_mul/geyi.yaml",
        out=str(tmp_path / "out" / "mul"),
        session_root=str(tmp_path / "sessions"),
    )
    assert {"n": 1024} in result.verification_report.coverage.shapes
    assert {"n": 1025} in result.verification_report.coverage.shapes


def test_phase1_transpose_coverage_contains_required_shapes(tmp_path):
    result = run_phase1(
        "examples/transpose2d/transpose.cu",
        spec="examples/transpose2d/geyi.yaml",
        out=str(tmp_path / "out" / "transpose"),
        session_root=str(tmp_path / "sessions"),
    )
    assert {"rows": 32, "cols": 64} in result.verification_report.coverage.shapes
    assert {"rows": 31, "cols": 65} in result.verification_report.coverage.shapes


@pytest.mark.parametrize(
    ("source", "spec"),
    [
        ("examples/elementwise_mul/mul.cu", "examples/elementwise_mul/geyi.yaml"),
        ("examples/transpose2d/transpose.cu", "examples/transpose2d/geyi.yaml"),
    ],
)
def test_phase1_acceptance_cli_commands(tmp_path, source, spec):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "run",
            source,
            "--spec",
            spec,
            "--out",
            str(tmp_path / "out"),
            "--session-root",
            str(tmp_path / "sessions"),
            "--json",
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["level"] == "golden"
    assert payload["passed"] is True
    assert payload["backend"] == "tilelang"
    assert payload["coverage"]["hardware"] == ["local_cpu"]
