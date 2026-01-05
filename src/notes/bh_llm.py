"""
Thin wrapper around wbal LMs for Black Hole prompts.
"""

from __future__ import annotations

from typing import Optional

from wbal import GPT5MiniTester, LM


class BHLLM:
    """Minimal helper to get assistant text output from wbal LMs."""

    def __init__(self, model: str = "gpt-5-mini", lm: Optional[LM] = None):
        self.lm = lm or GPT5MiniTester(model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.lm.invoke(messages, tools=None, mcp_servers=None)
        return self._extract_text(response)

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
