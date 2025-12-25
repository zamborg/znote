"""
Lightweight config loading for the notes system.
Reads a JSON file in the base path, falling back to defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG = {
    "embedding_provider": {
        "type": "openai",
        "model": "text-embedding-3-small",
    },
    "tagger": {
        "type": "openai",
        "model": "gpt-4o-mini",
    },
}


def load_config(base_path: Path) -> Dict[str, Any]:
    path = Path(base_path) / ".notes_config.json"
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge shallowly with defaults
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        # Fail-safe: use defaults if config is unreadable
        return DEFAULT_CONFIG.copy()
