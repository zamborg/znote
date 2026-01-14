"""
Thin wrapper around wbal LMs for Black Hole prompts.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from wbal import GPT5MiniTester, LM


class BHLLM:
    """Minimal helper to get assistant text output from wbal LMs."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        lm: Optional[LM] = None,
        event_recorder: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.model = model
        self.lm = lm or GPT5MiniTester(model=model)
        self.event_recorder = event_recorder

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.time()
        response = self.lm.invoke(messages, tools=None, mcp_servers=None)
        text = self._extract_text(response)
        if self.event_recorder:
            try:
                self.event_recorder(
                    {
                        "provider": "wbal",
                        "operation": "llm",
                        "model": self.model,
                        "duration_seconds": time.time() - start,
                        "cost_usd": None,
                        "metadata": {"output_chars": len(text), "output_lines": len(text.splitlines())},
                    }
                )
            except Exception:
                pass
        return text

    @staticmethod
    def _extract_text(response) -> str:
        texts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    texts.append(getattr(part, "text", ""))
        return "\n".join([t for t in texts if t]).strip()
