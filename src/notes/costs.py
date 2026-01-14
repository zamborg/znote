from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ChatPricing:
    input_per_1k: float
    output_per_1k: float


@dataclass(frozen=True)
class EmbeddingPricing:
    input_per_1k: float


@dataclass(frozen=True)
class AudioPricing:
    per_minute: float


# NOTE: This is a lightweight lookup table. Keep it easy to update.
CHAT_PRICING: dict[str, ChatPricing] = {
    # https://openai.com/pricing (update as needed)
    "gpt-4o-mini": ChatPricing(input_per_1k=0.00015, output_per_1k=0.00060),
}

EMBEDDING_PRICING: dict[str, EmbeddingPricing] = {
    # https://openai.com/pricing (update as needed)
    "text-embedding-3-small": EmbeddingPricing(input_per_1k=0.00002),
    "text-embedding-3-large": EmbeddingPricing(input_per_1k=0.00013),
}

AUDIO_PRICING: dict[str, AudioPricing] = {
    # https://openai.com/pricing (update as needed)
    "whisper-1": AudioPricing(per_minute=0.006),
}


def estimate_cost_usd(
    *,
    operation: str,
    model: str,
    usage: Optional[dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
) -> Optional[float]:
    """
    Estimate USD cost for a single call.

    Returns None when pricing or required metrics are unavailable.
    """
    if not model:
        return None

    if operation == "chat_completions":
        pricing = CHAT_PRICING.get(model)
        if not pricing or not usage:
            return None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is None or completion is None:
            return None
        return (float(prompt) / 1000.0) * pricing.input_per_1k + (float(completion) / 1000.0) * pricing.output_per_1k

    if operation == "embeddings":
        pricing = EMBEDDING_PRICING.get(model)
        if not pricing or not usage:
            return None
        tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
        if tokens is None:
            return None
        return (float(tokens) / 1000.0) * pricing.input_per_1k

    if operation == "transcription":
        pricing = AUDIO_PRICING.get(model)
        if not pricing or duration_seconds is None:
            return None
        return (float(duration_seconds) / 60.0) * pricing.per_minute

    return None

