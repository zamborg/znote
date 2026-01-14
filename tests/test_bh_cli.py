from notes.bh_cli import BlackHoleCLI
import numpy as np
import sqlite3


class FakeEmbeddingProvider:
    name = "fake"
    dimension = 4

    def embed_texts(self, texts):
        return [np.ones(self.dimension, dtype=np.float32) for _ in texts]


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs

    def complete(self, system_prompt: str, user_prompt: str):
        if self.outputs:
            return self.outputs.pop(0)
        return ""


class FakeTranscriber:
    def __init__(self, text: str):
        self.text = text

    def transcribe_file(self, path):
        return self.text


def test_bh_add_from_text(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    note_id = cli.add("Quick idea about black holes and data capture")

    note = cli.storage.load_note(note_id)
    assert note is not None
    assert "black holes" in note.content
    assert note.title.startswith("BH")
    cli.close()


def test_bh_todo_scope_all(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()

    todo_1 = cli.create_todo("ship the feature", title="Task 1", due="2024-12-01")
    todo_2 = cli.create_todo("follow up tomorrow", title="Task 2")

    cli.proactive_todo(scope="all")
    todo_note = cli.storage.load_note(BlackHoleCLI.TODO_NOTE_ID)

    assert todo_note is not None
    assert "Task 1" in todo_note.content
    assert todo_1 in todo_note.content
    assert todo_2 in todo_note.content
    assert "2024-12-01" in todo_note.content
    cli.close()


def test_bh_brief_scope_new(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM(
        [
            "## alpha\n- Old note (id-alpha): first content",  # first run
            "## beta\n- New note (id-beta): rocket content",  # second run
        ]
    )

    cli.add("First content for category", title="Old note", tags=["alpha"])
    cli.proactive_brief(scope="all")
    first = cli.storage.load_note(BlackHoleCLI.BRIEF_NOTE_ID)
    assert "Old note" in first.content

    cli.add("Second content about rockets", title="New note", tags=["beta"])
    cli.proactive_brief(scope="new")
    updated = cli.storage.load_note(BlackHoleCLI.BRIEF_NOTE_ID)

    assert "New note" in updated.content
    assert "Old note" not in updated.content
    cli.close()


def test_bh_audio_transcription(tmp_path):
    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"fake audio bytes")

    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.transcriber = FakeTranscriber("transcribed text")
    cli.llm = FakeLLM([])

    note_id = cli.add(str(audio_file))
    note = cli.storage.load_note(note_id)
    assert "transcribed text" in note.content
    cli.close()


def test_bh_records_source_provenance(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world", encoding="utf-8")

    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    note_id = cli.add(str(doc))
    note = cli.storage.load_note(note_id)
    assert note is not None
    assert note.attachments
    assert note.attachments[0].sha256
    assert note.attachments[0].source_id

    db_path = tmp_path / ".notes_db" / "bh.db"
    conn = sqlite3.connect(db_path)
    try:
        sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        refs = conn.execute("SELECT COUNT(*) FROM source_refs WHERE item_id = ?", (note_id,)).fetchone()[0]
    finally:
        conn.close()

    assert sources >= 1
    assert refs >= 1
    cli.close()


def test_bh_todo_lifecycle(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    todo_id = cli.create_todo("Ship v1", due="2026-01-14", stream="Work/BH")
    todo = cli.storage.load_note(todo_id)
    assert todo is not None
    assert todo.kind == "todo"
    assert todo.due_at is not None
    assert todo.stream == "work/bh"
    assert todo.completed_at is None
    assert todo.archived_at is None

    assert cli.mark_todo_done(todo_id) is True
    done = cli.storage.load_note(todo_id)
    assert done is not None
    assert done.completed_at is not None
    assert done.archived_at is None

    assert cli.archive_item(todo_id) is True
    archived = cli.storage.load_note(todo_id)
    assert archived is not None
    assert archived.archived_at is not None
    cli.close()


def test_bh_digest_scope_all(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM(["# Summary\n- item"])

    cli.add("Note one", title="First")
    cli.add("Note two", title="Second")

    cli.digest(scope="all")
    digest = cli.storage.load_note(BlackHoleCLI.DIGEST_NOTE_ID)
    assert digest is not None
    assert "Summary" in digest.content
    cli.close()


def test_bh_retag_scope_all(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM(["alpha, beta"])

    note_id = cli.add("content body", title="Title")
    cli.retag(scope="all")

    note = cli.storage.load_note(note_id)
    assert note.tags == ["alpha", "beta"]
    cli.close()


def test_bh_yank(tmp_path, monkeypatch):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    monkeypatch.setattr(cli, "_read_clipboard", lambda: "clipboard text")
    cli.yank()

    notes = cli.storage.list_notes()
    assert len(notes) == 1
    note = cli.storage.load_note(notes[0])
    assert "clipboard text" in note.content
    cli.close()


def test_bh_link(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    first = cli.add("shared content about rockets", title="A")
    second = cli.add("shared content about rockets", title="B")

    cli.link(first, min_weight=0.0)
    links = cli.graph.get_linked_notes(first, min_weight=0.0)
    assert any(l[0] == second for l in links)
    cli.close()


def test_bh_lint_reports_missing_content(tmp_path):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    note_id = cli.add("body", title="Has content")
    # Remove content file to trigger lint issue
    cli.storage._content_path(note_id).unlink()

    issues = cli.lint()
    assert any("content.md missing" in issue for issue in issues)
    cli.close()


def test_bh_open_uses_editor(tmp_path, monkeypatch):
    cli = BlackHoleCLI(base_path=tmp_path)
    cli.tagger = None
    cli.search.semantic_search.provider = FakeEmbeddingProvider()
    cli.llm = FakeLLM([])

    note_id = cli.add("body", title="To open")
    seen = {}

    monkeypatch.setenv("EDITOR", "unit-editor")
    monkeypatch.setattr("subprocess.run", lambda args: seen.setdefault("args", args))

    cli.open_note(note_id)
    assert seen["args"][0] == "unit-editor"
    cli.close()
