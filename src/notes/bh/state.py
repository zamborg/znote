from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class BlackHoleState:
    """Persists BH state (last processed timestamps, watched folders)."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.data = {"last_processed": {}, "watches": {}}
        self._load()

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                self.data = json.load(fh)
        except Exception:
            self.data = {"last_processed": {}, "watches": {}}

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    def get_last_processed(self, key: str) -> Optional[datetime]:
        raw = (self.data.get("last_processed") or {}).get(key)
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def update_last_processed(self, key: str, when: datetime):
        self.data.setdefault("last_processed", {})[key] = when.isoformat()
        self._save()

    def get_watch_timestamp(self, path: Path) -> Optional[datetime]:
        raw = (self.data.get("watches") or {}).get(str(path))
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def update_watch_timestamp(self, path: Path, when: datetime):
        self.data.setdefault("watches", {})[str(path)] = when.isoformat()
        self._save()

    def watch_paths(self) -> dict:
        return self.data.get("watches", {}) or {}

