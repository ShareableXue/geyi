"""Small token-budget utilities for independent LLM sessions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenBudget:
    input_tokens: int
    output_tokens: int


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(str(message.get("content") or "")) for message in messages)
