"""
Reintegration/migration helper.
Reindexes notes (search + embeddings) filtered by modified/created date and optionally rebuilds the graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..datamodel import Note
from ..graph import NotesGraph
from ..search import NotesSearch
from ..storage import NotesStorage


def _to_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        return _to_utc(dt)
    except ValueError:
        raise ValueError(f"Invalid datetime format: {value}. Use ISO format, e.g., 2024-01-01T00:00:00")


def _in_window(note: Note, after: Optional[datetime], before: Optional[datetime]) -> bool:
    note_modified = _to_utc(note.modified_at)
    if after and note_modified <= after:
        return False
    if before and note_modified >= before:
        return False
    return True


def reintegrate(
    storage: NotesStorage,
    search: NotesSearch,
    graph: NotesGraph,
    after: Optional[str] = None,
    before: Optional[str] = None,
    dry_run: bool = False,
    rebuild_graph: bool = True,
) -> dict:
    """
    Reprocess notes filtered by modified_at window.
    - Reindexes keyword + semantic search.
    - Optionally rebuilds the graph at the end.
    Returns summary stats.
    """
    after_dt = _parse_dt(after)
    before_dt = _parse_dt(before)

    processed = 0
    skipped = 0

    note_ids = storage.list_notes()

    for note_id in note_ids:
        note = storage.load_note(note_id)
        if not note:
            skipped += 1
            continue

        if not _in_window(note, after_dt, before_dt):
            skipped += 1
            continue

        processed += 1
        if dry_run:
            continue

        # Reindex search (keyword + semantic)
        search.index_note(note)

    # Rebuild graph once at the end for consistency
    if not dry_run and rebuild_graph:
        graph.rebuild_graph()

    return {
        "processed": processed,
        "skipped": skipped,
        "total": len(note_ids),
        "rebuild_graph": rebuild_graph and not dry_run,
    }
