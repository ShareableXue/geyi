"""Phase 2 context orchestrator.

The orchestrator keeps the main run session small and writes every LLM task to
an independent child session with prompts, raw responses, usage, and output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geyi.context.budget import TokenBudget, estimate_messages
from geyi.context.sessions import ChildSession
from geyi.llm.client import LLMProvider, LLMResponse


DEFAULT_BUDGETS = {
    "planner": TokenBudget(input_tokens=6000, output_tokens=2000),
    "repair": TokenBudget(input_tokens=6000, output_tokens=2000),
    "precision": TokenBudget(input_tokens=6000, output_tokens=2000),
}


class ContextOrchestrator:
    def __init__(self, main_session_path: Path):
        self.main_session_path = Path(main_session_path)
        self.root = self.main_session_path / "context"
        self.root.mkdir(parents=True, exist_ok=True)

    def start_child(self, task: str) -> ChildSession:
        return ChildSession.create(self.root, task)

    def complete(
        self,
        child: ChildSession,
        provider: LLMProvider,
        messages: list[dict[str, str]],
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        budget = DEFAULT_BUDGETS.get(task, TokenBudget(input_tokens=6000, output_tokens=2000))
        estimated_tokens = estimate_messages(messages)
        child.write_json(
            "input/budget.json",
            {
                "task": task,
                "estimated_input_tokens": estimated_tokens,
                "input_budget": budget.input_tokens,
                "output_budget": budget.output_tokens,
                "over_budget": estimated_tokens > budget.input_tokens,
            },
        )
        child.write_json("input/metadata.json", metadata or {})
        child.write_json("prompt/messages.json", messages)
        child.write_text(
            "prompt/rendered.md",
            "\n\n".join("## %s\n%s" % (message.get("role", "user"), message.get("content", "")) for message in messages),
        )
        response = provider.complete(messages, task=task)
        child.write_json("response/usage.json", response.usage.to_dict())
        child.write_text("response/raw.txt", response.content)
        child.write_json("response/raw.json", response.raw)
        return response
