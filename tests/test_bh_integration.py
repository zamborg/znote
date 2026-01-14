import json
import sys
from pathlib import Path

import notes.bh_cli as bh_cli
from notes.bh_cli import BlackHoleCLI


def _write_no_network_config(base_path: Path) -> None:
    cfg = {
        "embedding_provider": {"type": None},
        "tagger": {"type": None},
        "transcriber": {"type": "none"},
        "llm": {"model": "gpt-5-mini"},
    }
    (base_path / ".notes_config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _run_bh(tmp_path: Path, monkeypatch, capsys, args: list[str]) -> str:
    argv = ["bh", "--base-path", str(tmp_path), *args]
    monkeypatch.setattr(sys, "argv", argv)
    bh_cli.main()
    return capsys.readouterr().out


def _extract_created_id(output: str, prefix: str) -> str:
    for line in output.splitlines():
        if line.startswith(prefix):
            rest = line[len(prefix) :].strip()
            if rest.startswith(":"):
                rest = rest[1:].strip()
            if rest:
                return rest
    raise AssertionError(f"Could not find '{prefix}' in output:\n{output}")


def test_bh_cli_add_and_search_stream_filter(tmp_path, monkeypatch, capsys):
    _write_no_network_config(tmp_path)

    out1 = _run_bh(
        tmp_path, monkeypatch, capsys, ["add", "rocket engines", "--stream", "Work/BH"]
    )
    id1 = _extract_created_id(out1, "Created note")

    out2 = _run_bh(tmp_path, monkeypatch, capsys, ["add", "rocket engines", "--stream", "other"])
    id2 = _extract_created_id(out2, "Created note")

    out = _run_bh(
        tmp_path,
        monkeypatch,
        capsys,
        ["search", "rocket", "--mode", "keyword", "--limit", "10", "--stream", "work"],
    )
    assert id1 in out
    assert id2 not in out

    app = BlackHoleCLI(base_path=tmp_path)
    try:
        note = app.get_note(id1)
        assert note is not None
        assert note.stream == "work/bh"
    finally:
        app.close()


def test_bh_cli_new_uses_editor(tmp_path, monkeypatch, capsys):
    _write_no_network_config(tmp_path)

    monkeypatch.setenv("EDITOR", "unit-editor")

    def fake_run(argv):
        path = Path(argv[1])
        path.write_text("# Edited Title\n\nHello from editor", encoding="utf-8")

    import notes.bh.app as bh_app

    monkeypatch.setattr(bh_app.subprocess, "run", fake_run)

    out = _run_bh(tmp_path, monkeypatch, capsys, ["new", "--stream", "Work/BH"])
    note_id = _extract_created_id(out, "Created note")

    app = BlackHoleCLI(base_path=tmp_path)
    try:
        note = app.get_note(note_id)
        assert note is not None
        assert note.title == "Edited Title"
        assert "Hello from editor" in note.content
        assert note.stream == "work/bh"
    finally:
        app.close()


def test_bh_cli_add_directory_ingests_each_file(tmp_path, monkeypatch, capsys):
    _write_no_network_config(tmp_path)

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text("alpha", encoding="utf-8")
    (inbox / "b.bin").write_bytes(b"\x00\x01\x02")

    out = _run_bh(tmp_path, monkeypatch, capsys, ["add", str(inbox), "--stream", "inbox"])
    ids = [line[2:].strip() for line in out.splitlines() if line.startswith("- ")]
    assert len(ids) == 2

    app = BlackHoleCLI(base_path=tmp_path)
    try:
        assert len(app.storage.list_notes()) == 2
    finally:
        app.close()


def test_bh_cli_todo_commands(tmp_path, monkeypatch, capsys):
    _write_no_network_config(tmp_path)

    out = _run_bh(
        tmp_path,
        monkeypatch,
        capsys,
        ["todo", "add", "Ship v1", "--due", "2026-01-14", "--stream", "Work/BH"],
    )
    todo_id = _extract_created_id(out, "Created todo")

    out = _run_bh(tmp_path, monkeypatch, capsys, ["todo", "done", todo_id])
    assert "Marked done" in out

    out = _run_bh(
        tmp_path,
        monkeypatch,
        capsys,
        ["todo", "list", "--status", "done", "--stream", "work", "--limit", "20"],
    )
    assert todo_id in out
    assert "Ship v1" in out

    out = _run_bh(tmp_path, monkeypatch, capsys, ["todo", "archive", todo_id])
    assert "Archived" in out

    out = _run_bh(
        tmp_path,
        monkeypatch,
        capsys,
        ["todo", "list", "--status", "archived", "--stream", "work", "--limit", "20"],
    )
    assert todo_id in out


def test_bh_cli_completion_stream(tmp_path, monkeypatch, capsys):
    _write_no_network_config(tmp_path)

    out_note = _run_bh(tmp_path, monkeypatch, capsys, ["add", "note body", "--stream", "work/bh"])
    note_id = _extract_created_id(out_note, "Created note")
    out_todo = _run_bh(tmp_path, monkeypatch, capsys, ["todo", "add", "todo body", "--stream", "work/bh"])
    todo_id = _extract_created_id(out_todo, "Created todo")

    out = _run_bh(
        tmp_path, monkeypatch, capsys, ["completion", "stream", "work/bh", "--limit", "100"]
    )
    index_id = _extract_created_id(out, "Completion stream index -> note")

    app = BlackHoleCLI(base_path=tmp_path)
    try:
        index = app.get_note(index_id)
        assert index is not None
        assert "Stream Index: work/bh" in index.content
        assert note_id in index.content
        assert todo_id in index.content
        assert index.stream == "work/bh"
    finally:
        app.close()


def test_bh_cli_cost_empty(tmp_path, monkeypatch, capsys):
    _write_no_network_config(tmp_path)

    out = _run_bh(tmp_path, monkeypatch, capsys, ["cost", "daily", "--days", "1"])
    assert "No cost events recorded." in out
