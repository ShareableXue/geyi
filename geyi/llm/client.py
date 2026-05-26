"""LLM provider abstraction for Phase 2."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMUsage:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cached": self.cached,
        }


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage
    raw: dict[str, Any]


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[dict[str, str]], task: str = "planner") -> LLMResponse:
        ...


class LLMProviderError(RuntimeError):
    pass


class LLMCredentialError(LLMProviderError):
    pass


class LLMHTTPError(LLMProviderError):
    pass


class MockLLMProvider:
    """Deterministic local provider for tests and offline Phase 2 demos."""

    name = "mock"

    def __init__(self, responses: list[Any] | None = None, model: str = "mock-phase2-planner"):
        self.responses = list(responses or [])
        self.model = model
        self.calls: list[LLMResponse] = []

    def complete(self, messages: list[dict[str, str]], task: str = "planner") -> LLMResponse:
        if self.responses:
            payload = self.responses.pop(0)
        else:
            payload = self.default_payload(messages, task)
        content = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
        prompt_tokens = rough_tokens(json.dumps(messages, sort_keys=True))
        completion_tokens = rough_tokens(content)
        response = LLMResponse(
            content=content,
            usage=LLMUsage(
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=0.0,
                cached=False,
            ),
            raw={"task": task, "mock": True, "content": content},
        )
        self.calls.append(response)
        return response

    def default_payload(self, messages: list[dict[str, str]], task: str) -> dict[str, Any]:
        joined = "\n".join(message.get("content", "") for message in messages)
        if "fused_add_relu" in joined or "add followed by relu" in joined:
            return {
                "intent_confirmation": "The contract describes elementwise fused add followed by relu.",
                "selected_backend": "tilelang",
                "selected_template": "tilelang.fused_add_relu_1d",
                "parameter_bindings": {},
                "required_assumptions": [],
                "risks": ["Phase 2 planner selected a composite template; golden verification is required."],
                "repair_suggestions": [],
                "cannot_translate": False,
            }
        if task == "repair":
            return {
                "intent_confirmation": "Repair keeps the original fused_add_relu intent and removes invalid debug parameters.",
                "selected_backend": "tilelang",
                "selected_template": "tilelang.fused_add_relu_1d",
                "parameter_bindings": {},
                "required_assumptions": [],
                "risks": ["Re-run compile and golden verification after repair."],
                "repair_suggestions": ["Regenerate the constrained fused_add_relu template without the syntax fault."],
                "cannot_translate": False,
            }
        return {
            "intent_confirmation": "Planner could not safely choose a template from the compact contract.",
            "selected_backend": "tilelang",
            "selected_template": None,
            "parameter_bindings": {},
            "required_assumptions": [],
            "risks": ["No supported Phase 2 template matched."],
            "repair_suggestions": [],
            "cannot_translate": True,
            "annotation_request": {
                "question": "Provide a supported template or add a user annotation for the unresolved intent.",
                "unknown_ids": ["no_supported_intent"],
            },
        }


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible JSON endpoint client.

    This path is intentionally small and dependency-free. Tests use the mock
    provider; real calls require an explicit API key in the environment.
    """

    name = "openai-compatible"

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.name = "openai-compatible"

    def complete(self, messages: list[dict[str, str]], task: str = "planner") -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMCredentialError("missing %s for %s provider" % (self.api_key_env, self.name))
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as handle:
                raw = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMHTTPError(
                "%s provider HTTP %d %s: %s" % (self.name, exc.code, exc.reason, truncate(body))
            ) from exc
        content = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or rough_tokens(json.dumps(messages, sort_keys=True)))
        completion_tokens = int(usage.get("completion_tokens") or rough_tokens(content))
        return LLMResponse(
            content=content,
            usage=LLMUsage(
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
            ),
            raw=redact_api_payload(raw),
        )


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com/chat/completions",
        api_key_env: str = "DEEPSEEK_API_KEY",
    ):
        super().__init__(model=model, base_url=base_url, api_key_env=api_key_env)
        self.name = "deepseek"


def create_provider(name: str = "mock", model: str | None = None, base_url: str | None = None) -> LLMProvider:
    if name == "mock":
        return MockLLMProvider(model=model or "mock-phase2-planner")
    if name in {"openai", "openai-compatible"}:
        return OpenAICompatibleProvider(
            model=model or "gpt-4.1-mini",
            base_url=base_url or "https://api.openai.com/v1/chat/completions",
        )
    if name == "deepseek":
        return DeepSeekProvider(
            model=model or "deepseek-v4-pro",
            base_url=base_url or "https://api.deepseek.com/chat/completions",
        )
    raise ValueError("unsupported LLM provider: %s" % name)


def rough_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def truncate(text: str, limit: int = 1000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def redact_api_payload(raw: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(raw)
    if "headers" in redacted:
        redacted["headers"] = "<redacted>"
    return redacted
