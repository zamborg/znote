"""
Lightweight wrapper for audio transcription via the Whisper endpoint.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from .costs import estimate_cost_usd


class TranscriptionError(RuntimeError):
    """Raised when transcription fails."""


class WhisperTranscriber:
    """Transcribe audio files using the OpenAI Whisper API."""

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        language: Optional[str] = None,
        event_recorder: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.language = language
        self.event_recorder = event_recorder

    @classmethod
    def from_config(
        cls, config: dict, event_recorder: Optional[Callable[[dict[str, Any]], None]] = None
    ) -> Optional["WhisperTranscriber"]:
        cfg = (config or {}).get("transcriber", {}) or {}
        if not cfg or cfg.get("type") not in {None, "whisper"}:
            return None
        return cls(
            model=cfg.get("model", "whisper-1"),
            api_key=cfg.get("api_key"),
            api_base=cfg.get("api_base"),
            language=cfg.get("language"),
            event_recorder=event_recorder,
        )

    @staticmethod
    def _probe_duration_seconds(path: Path) -> Optional[float]:
        if not shutil.which("ffprobe"):
            return None
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            if not out:
                return None
            return float(out)
        except Exception:
            return None

    def transcribe_file(self, path: Path, prompt: Optional[str] = None, timeout: int = 60) -> str:
        if not self.api_key:
            raise TranscriptionError("OPENAI_API_KEY is required for transcription")

        url = f"{self.api_base.rstrip('/')}/audio/transcriptions"
        with open(path, "rb") as fh:
            files = {
                "file": (path.name, fh, "application/octet-stream"),
            }
            data = {"model": self.model}
            if prompt:
                data["prompt"] = prompt
            if self.language:
                data["language"] = self.language

            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files,
                timeout=timeout,
            )

        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Transcription request failed: {exc}") from exc

        payload = resp.json()
        text = payload.get("text")
        if not text:
            raise TranscriptionError("No transcription text returned.")

        if self.event_recorder:
            duration_seconds = self._probe_duration_seconds(path)
            cost_usd = estimate_cost_usd(
                operation="transcription", model=self.model, duration_seconds=duration_seconds
            )
            try:
                self.event_recorder(
                    {
                        "provider": "openai",
                        "operation": "transcription",
                        "model": self.model,
                        "request_id": resp.headers.get("x-request-id"),
                        "duration_seconds": duration_seconds,
                        "cost_usd": cost_usd,
                        "metadata": {"filename": path.name, "size_bytes": path.stat().st_size},
                    }
                )
            except Exception:
                pass

        return text.strip()
