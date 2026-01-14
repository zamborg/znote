from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..datamodel import Attachment


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlackHoleDB:
    """
    BH SQLite backing store for provenance/cost/etc.

    For now, we keep it focused on ground-truth source tracking.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self):
        self.conn.close()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
              source_id TEXT PRIMARY KEY,
              sha256 TEXT NOT NULL,
              filename TEXT NOT NULL,
              media_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              first_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_refs (
              ref_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
              item_id TEXT NOT NULL,
              role TEXT NOT NULL,
              original_path TEXT,
              stored_relpath TEXT,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_source_refs_item_id ON source_refs(item_id);
            CREATE INDEX IF NOT EXISTS idx_source_refs_source_id ON source_refs(source_id);

            CREATE TABLE IF NOT EXISTS cost_events (
              event_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              provider TEXT NOT NULL,
              operation TEXT NOT NULL,
              model TEXT,
              request_id TEXT,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              total_tokens INTEGER,
              duration_seconds REAL,
              cost_usd REAL,
              metadata_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cost_events_created_at ON cost_events(created_at);
            """
        )
        self.conn.commit()

    def upsert_source(self, attachment: Attachment) -> str:
        """
        Ensure a source row exists for a given attachment.
        Uses sha256 as the stable source id when available.
        """
        source_id = attachment.sha256 or str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO sources (source_id, sha256, filename, media_type, size_bytes, first_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
              filename=excluded.filename,
              media_type=excluded.media_type,
              size_bytes=excluded.size_bytes
            """,
            (
                source_id,
                attachment.sha256 or "",
                attachment.filename,
                attachment.media_type.value,
                int(attachment.size_bytes),
                _utc_now_iso(),
            ),
        )
        self.conn.commit()
        return source_id

    def add_source_ref(
        self,
        *,
        item_id: str,
        source_id: str,
        role: str = "attachment",
        original_path: Optional[str] = None,
        stored_relpath: Optional[str] = None,
    ) -> str:
        ref_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO source_refs (ref_id, source_id, item_id, role, original_path, stored_relpath, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ref_id, source_id, item_id, role, original_path, stored_relpath, _utc_now_iso()),
        )
        self.conn.commit()
        return ref_id

    def record_cost_event(
        self,
        *,
        provider: str,
        operation: str,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO cost_events (
              event_id, created_at, provider, operation, model, request_id,
              prompt_tokens, completion_tokens, total_tokens, duration_seconds, cost_usd, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                _utc_now_iso(),
                provider,
                operation,
                model,
                request_id,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                duration_seconds,
                cost_usd,
                json.dumps(metadata) if metadata is not None else None,
            ),
        )
        self.conn.commit()
        return event_id

    def daily_costs(self, days: int = 14) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT
              substr(created_at, 1, 10) AS day,
              COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
              COUNT(*) AS events,
              COALESCE(SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END), 0) AS unknown_cost_events
            FROM cost_events
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (int(days),),
        )
        return [
            {
                "day": row[0],
                "cost_usd": float(row[1] or 0.0),
                "events": int(row[2] or 0),
                "unknown_cost_events": int(row[3] or 0),
            }
            for row in cursor.fetchall()
        ]

    def list_cost_events(self, day: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if day:
            cursor = self.conn.execute(
                """
                SELECT created_at, provider, operation, model, request_id, cost_usd, prompt_tokens, completion_tokens, total_tokens, duration_seconds
                FROM cost_events
                WHERE created_at LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"{day}%", int(limit)),
            )
        else:
            cursor = self.conn.execute(
                """
                SELECT created_at, provider, operation, model, request_id, cost_usd, prompt_tokens, completion_tokens, total_tokens, duration_seconds
                FROM cost_events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            )
        rows = []
        for r in cursor.fetchall():
            rows.append(
                {
                    "created_at": r[0],
                    "provider": r[1],
                    "operation": r[2],
                    "model": r[3],
                    "request_id": r[4],
                    "cost_usd": r[5],
                    "prompt_tokens": r[6],
                    "completion_tokens": r[7],
                    "total_tokens": r[8],
                    "duration_seconds": r[9],
                }
            )
        return rows
