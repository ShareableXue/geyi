from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from geyi.llm.client import DeepSeekProvider, OpenAICompatibleProvider, create_provider
from geyi.llm.planner import PlannerHandoffRequired
from geyi.phase2 import run_phase2


FUSED_SOURCE = "examples/template_gap/fused_add_relu.cu"
FUSED_SPEC = "examples/template_gap/geyi.yaml"


def valid_fused_payload():
    return {
        "intent_confirmation": "The contract is fused add followed by relu.",
        "selected_backend": "tilelang",
        "selected_template": "tilelang.fused_add_relu_1d",
        "parameter_bindings": {},
        "required_assumptions": [],
        "risks": ["golden verification required"],
        "repair_suggestions": [],
        "cannot_translate": False,
    }


def test_real_provider_missing_api_key_writes_credential_handoff(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(PlannerHandoffRequired) as raised:
        run_phase2(
            FUSED_SOURCE,
            spec=FUSED_SPEC,
            out=str(tmp_path / "out"),
            session_root=str(tmp_path / "sessions"),
            allow_llm_plan=True,
            provider=OpenAICompatibleProvider(base_url="https://example.invalid/v1/chat/completions"),
        )

    assert raised.value.handoff["reason"] == "LLM provider credentials unavailable"
    handoffs = list((tmp_path / "sessions").glob("*/handoff/credentials.json"))
    assert handoffs
    assert "OPENAI_API_KEY" in handoffs[0].read_text(encoding="utf-8")


def test_deepseek_provider_missing_api_key_writes_credential_handoff(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(PlannerHandoffRequired) as raised:
        run_phase2(
            FUSED_SOURCE,
            spec=FUSED_SPEC,
            out=str(tmp_path / "out"),
            session_root=str(tmp_path / "sessions"),
            allow_llm_plan=True,
            provider=DeepSeekProvider(),
        )

    assert raised.value.handoff["provider"] == "deepseek"
    handoffs = list((tmp_path / "sessions").glob("*/handoff/credentials.json"))
    assert handoffs
    assert "DEEPSEEK_API_KEY" in handoffs[0].read_text(encoding="utf-8")


def test_real_provider_mocked_transport_records_usage_without_leaking_key(tmp_path, monkeypatch):
    secret = "sk-geyi-secret-for-test"

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(valid_fused_payload())}}],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 17, "total_tokens": 30},
                }
            ).encode("utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeHandle())

    result = run_phase2(
        FUSED_SOURCE,
        spec=FUSED_SPEC,
        out=str(tmp_path / "out"),
        session_root=str(tmp_path / "sessions"),
        allow_llm_plan=True,
        provider=OpenAICompatibleProvider(base_url="https://example.invalid/v1/chat/completions"),
    )

    assert result.verification_report.passed
    assert result.verification_report.llm_calls[0]["provider"] == "openai-compatible"
    assert result.verification_report.llm_calls[0]["total_tokens"] == 30
    assert secret not in all_session_text(result.analysis.session.path)


def test_create_provider_deepseek_defaults():
    provider = create_provider("deepseek")
    assert isinstance(provider, DeepSeekProvider)
    assert provider.model == "deepseek-v4-pro"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert provider.base_url == "https://api.deepseek.com/chat/completions"


def test_provider_http_error_writes_actionable_handoff(tmp_path, monkeypatch):
    class ErrorBody:
        def read(self):
            return b'{"error":{"message":"model not found"}}'

        def close(self):
            return None

    def raise_http_error(request, timeout):
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs={},
            fp=ErrorBody(),
        )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-geyi-secret-for-test")
    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)

    with pytest.raises(PlannerHandoffRequired) as raised:
        run_phase2(
            FUSED_SOURCE,
            spec=FUSED_SPEC,
            out=str(tmp_path / "out"),
            session_root=str(tmp_path / "sessions"),
            allow_llm_plan=True,
            provider=DeepSeekProvider(model="bad-model"),
        )

    assert raised.value.handoff["reason"] == "LLM provider request failed"
    assert "model not found" in raised.value.handoff["message"]
    assert list((tmp_path / "sessions").glob("*/handoff/provider_error.json"))


def all_session_text(path: Path) -> str:
    chunks = []
    for item in path.rglob("*"):
        if item.is_file():
            chunks.append(item.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)
