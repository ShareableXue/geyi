from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from geyi.analysis import analyze
from geyi.llm.client import MockLLMProvider, OpenAICompatibleProvider
from geyi.llm.diagnostics import diagnose_precision_mismatch
from geyi.llm.planner import PlannerHandoffRequired, plan_with_llm
from geyi.phase2 import run_phase2
from geyi.verifier.report import Coverage, VerificationLevel, VerificationReport


ROOT = Path(__file__).resolve().parents[1]
FUSED_SOURCE = "examples/template_gap/fused_add_relu.cu"
FUSED_SPEC = "examples/template_gap/geyi.yaml"


def valid_fused_payload(parameter_bindings=None):
    return {
        "intent_confirmation": "The contract is fused add followed by relu.",
        "selected_backend": "tilelang",
        "selected_template": "tilelang.fused_add_relu_1d",
        "parameter_bindings": dict(parameter_bindings or {}),
        "required_assumptions": [],
        "risks": ["golden verification required"],
        "repair_suggestions": [],
        "cannot_translate": False,
    }


def test_llm_planner_mock_closes_template_gap(tmp_path):
    result = run_phase2(
        FUSED_SOURCE,
        spec=FUSED_SPEC,
        out=str(tmp_path / "out"),
        session_root=str(tmp_path / "sessions"),
        allow_llm_plan=True,
        provider=MockLLMProvider(),
    )

    assert result.plan.strategy == "llm_plan"
    assert result.plan.template == "tilelang.fused_add_relu_1d"
    assert result.verification_report.passed
    assert result.verification_report.llm_used is True
    assert result.verification_report.llm_calls[0]["provider"] == "mock"
    assert (result.analysis.session.path / "llm" / "planner_report.json").exists()
    assert any((result.analysis.session.path / "context").glob("planner_*"))


def test_invalid_planner_json_is_rejected_then_repaired(tmp_path):
    analysis = analyze(
        FUSED_SOURCE,
        spec=FUSED_SPEC,
        session_root=str(tmp_path / "sessions"),
        write_session=True,
    )
    provider = MockLLMProvider(responses=["not json", valid_fused_payload()])

    result = plan_with_llm(
        analysis.contract,
        FUSED_SOURCE,
        analysis.session,
        provider,
    )

    assert result.plan.template == "tilelang.fused_add_relu_1d"
    assert len(result.llm_calls) == 2
    rejection_files = list((analysis.session.path / "context").glob("planner_*/output/schema_rejection.json"))
    assert rejection_files


def test_missing_assumption_planner_asks_for_annotation(tmp_path):
    analysis = analyze(
        "tests/fixtures/contracts/missing_shape/vector_add.cu",
        spec="tests/fixtures/contracts/missing_shape/geyi.yaml",
        session_root=str(tmp_path / "sessions"),
        write_session=True,
    )

    with pytest.raises(PlannerHandoffRequired) as raised:
        plan_with_llm(analysis.contract, None, analysis.session, MockLLMProvider())

    assert "planner blocked" in raised.value.handoff["reason"]
    assert (analysis.session.path / "handoff" / "planner.json").exists()


def test_compile_error_fixture_repairs_or_escalates(tmp_path):
    provider = MockLLMProvider(
        responses=[
            valid_fused_payload({"debug_force_syntax_error": True}),
            valid_fused_payload(),
        ]
    )
    result = run_phase2(
        FUSED_SOURCE,
        spec=FUSED_SPEC,
        out=str(tmp_path / "out"),
        session_root=str(tmp_path / "sessions"),
        allow_llm_plan=True,
        provider=provider,
    )

    assert result.verification_report.passed
    assert len(result.verification_report.llm_calls) == 2
    assert (result.analysis.session.path / "llm" / "repair_report.json").exists()
    assert "debug_force_syntax_error" not in result.plan.parameters


def test_precision_mismatch_diagnostic_is_produced():
    report = VerificationReport(
        level=VerificationLevel.UNVERIFIED,
        contract_hash="abc",
        artifact_hash="def",
        coverage=Coverage(shapes=[{"n": 4}], dtypes=["float32"], hardware=["local_cpu"]),
        tolerance={"atol": 1e-5, "rtol": 1e-5},
        max_abs_diff=0.25,
        max_rel_diff=0.5,
        assumptions=[],
        unknowns=[],
        strategy="llm_plan",
        backend="tilelang",
        llm_used=True,
        passed=False,
        case_results=[{"case": "basic", "n": 4, "passed": False, "max_abs_diff": 0.25}],
    )

    diagnostic = diagnose_precision_mismatch(report)
    assert diagnostic["status"] == "mismatch"
    assert diagnostic["failed_cases"][0]["case"] == "basic"


def test_phase2_acceptance_cli_command(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "run",
            FUSED_SOURCE,
            "--spec",
            FUSED_SPEC,
            "--out",
            str(tmp_path / "out"),
            "--session-root",
            str(tmp_path / "sessions"),
            "--allow-llm-plan",
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
    assert payload["strategy"] == "llm_plan"
    assert payload["llm_used"] is True
    assert payload["llm_calls"][0]["provider"] == "mock"


def test_openai_compatible_provider_path_parses_json(monkeypatch):
    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(valid_fused_payload())}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 60
        return FakeHandle()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = OpenAICompatibleProvider(base_url="https://example.invalid/v1/chat/completions").complete(
        [{"role": "user", "content": "return planner json"}]
    )

    assert response.usage.provider == "openai-compatible"
    assert response.usage.total_tokens == 18
    assert json.loads(response.content)["selected_template"] == "tilelang.fused_add_relu_1d"
