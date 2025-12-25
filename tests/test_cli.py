from datetime import datetime, timedelta, timezone

import numpy as np

from notes.cli import NotesCLI
from notes.migrations.reintegrate import reintegrate


class FakeTagger:
    def suggest_tags(self, title: str, content: str, max_tags: int = 8):
        return ["auto", "ml"]


class FakeEmbeddingProvider:
    name = "fake"
    dimension = 4

    def embed_texts(self, texts):
        return [np.ones(self.dimension, dtype=np.float32) for _ in texts]


def test_create_note_with_auto_tags(tmp_path):
    cli = NotesCLI(base_path=tmp_path)
    cli.tagger = FakeTagger()
    cli.search.semantic_search.provider = FakeEmbeddingProvider()

    note_id = cli.create_note(
        "Hello", "content body", tags=["ml"], attachments=None, auto_tags=True
    )

    note = cli.storage.load_note(note_id)
    assert note is not None
    assert note.id
    assert set(note.tags) >= {"ml", "auto"}


def test_reintegrate_filters_by_date(tmp_path):
    cli = NotesCLI(base_path=tmp_path)
    cli.tagger = None  # avoid network
    cli.search.semantic_search.provider = FakeEmbeddingProvider()

    old_id = cli.create_note("Old", "old content")
    old_note = cli.storage.load_note(old_id)
    old_note.modified_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
    cli.storage.save_note(old_note)

    after_time = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    cli.create_note("New", "fresh content")

    summary = reintegrate(
        cli.storage,
        cli.search,
        cli.graph,
        after=after_time,
        before=None,
        dry_run=False,
        rebuild_graph=False,
    )

    assert summary["processed"] == 1
    assert summary["total"] == 2
    assert summary["rebuild_graph"] is False
