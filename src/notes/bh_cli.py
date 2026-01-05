#!/usr/bin/env python3
"""
Black Hole CLI wrapper around the existing notes system.
Implements a simplified API:
- bh add {content}
- bh search {query}
- bh proactive {todo|brief}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .cli import NotesCLI
from .datamodel import Note
from .bh_llm import BHLLM
from .transcription import TranscriptionError, WhisperTranscriber


TEXT_EXTS = {".txt", ".md", ".rst"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


class BlackHoleState:
    """Persists last processed timestamps for proactive modes."""

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
        raw = self.data.get("last_processed", {}).get(key)
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
        watch = self.data.get("watches", {}).get(str(path))
        if not watch:
            return None
        try:
            dt = datetime.fromisoformat(watch)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def update_watch_timestamp(self, path: Path, when: datetime):
        self.data.setdefault("watches", {})[str(path)] = when.isoformat()
        self._save()

    def watch_paths(self):
        return self.data.get("watches", {})


class BlackHoleCLI(NotesCLI):
    """BH-flavored CLI built atop the NotesCLI primitives."""

    TODO_NOTE_ID = "bh-todo"
    BRIEF_NOTE_ID = "bh-brief"
    DIGEST_NOTE_ID = "bh-digest"

    def __init__(
        self,
        base_path: Optional[Path] = None,
        transcriber: Optional[WhisperTranscriber] = None,
        llm: Optional[BHLLM] = None,
    ):
        super().__init__(base_path=base_path)
        self.state = BlackHoleState(self.storage.db_dir / "bh_state.json")
        self.transcriber = transcriber or WhisperTranscriber.from_config(self.config)
        llm_model = (self.config.get("llm") or {}).get("model", "gpt-5-mini")
        self.llm = llm or BHLLM(model=llm_model)

    def close(self):
        self.search.close()

    def add(
        self,
        content: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_tags: bool = False,
        transcribe_audio: bool = True,
    ):
        """
        Add raw content or ingest a file (text or audio).
        - Text files: content is inlined, file also attached for safekeeping.
        - Audio/other files: stored as attachment with a lightweight stub body.
        """
        path = Path(content)
        attachments: List[str] = []

        if path.exists() and path.is_file():
            attachments.append(str(path))

            if self._is_text_file(path):
                body = path.read_text(encoding="utf-8")
                note_title = title or path.stem
            elif self._is_audio_file(path):
                note_title = title or path.stem
                body = self._ingest_audio_body(path, transcribe_audio)
            else:
                note_title = title or path.stem
                body = f"Attachment ingested via bh add: {path.name}\n\n(No transcription available.)"
        else:
            body = content
            note_title = title or self._title_from_text(content)

        note_id = self.create_note(
            note_title,
            body,
            tags=tags,
            attachments=attachments,
            auto_tags=auto_tags,
        )
        return note_id

    def bh_search(self, query: str, limit: int = 10, mode: str = "semantic"):
        """
        Wrapper for search; defaults to semantic to emphasize conceptual retrieval.
        """
        self.search_notes(query, mode=mode, limit=limit)

    def proactive_todo(self, scope: str = "new"):
        """
        Aggregate TODO items into a superstructure note.
        scope: 'new' uses only notes modified after last run; 'all' scans everything.
        """
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "todo"))
        content = self._render_todo_doc_llm(notes, run_ts, scope)
        note_id = self._upsert_special_note(
            self.TODO_NOTE_ID,
            "BH TODOs",
            content,
            tags=["blackhole", "todo"],
        )
        self.state.update_last_processed("todo", run_ts)
        print(f"Proactive TODO complete -> note {note_id} ({len(notes)} notes scanned)")

    def proactive_brief(self, scope: str = "new"):
        """
        Build context briefings grouped by categories/tags.
        """
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "brief"))
        brief = self._render_brief_doc_llm(notes, run_ts, scope)
        note_id = self._upsert_special_note(
            self.BRIEF_NOTE_ID,
            "BH Briefings",
            brief,
            tags=["blackhole", "brief"],
        )
        self.state.update_last_processed("brief", run_ts)
        print(f"Proactive brief complete -> note {note_id} ({len(notes)} notes processed)")

    def digest(self, scope: str = "new"):
        """
        Generate a compact digest of notes and TODOs.
        """
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "digest"))
        digest = self._render_digest_doc_llm(notes, run_ts, scope)
        note_id = self._upsert_special_note(
            self.DIGEST_NOTE_ID,
            "BH Digest",
            digest,
            tags=["blackhole", "digest"],
        )
        self.state.update_last_processed("digest", run_ts)
        print(f"Digest complete -> note {note_id} ({len(notes)} notes processed)")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _select_notes(self, scope: str, key: str) -> Iterable[Note]:
        last_ts = self.state.get_last_processed(key) if scope == "new" else None
        for note_id in self.storage.list_notes():
            note = self.storage.load_note(note_id)
            if not note:
                continue
            if last_ts and note.modified_at <= last_ts:
                continue
            yield note

    @staticmethod
    def _title_from_text(content: str, max_words: int = 8) -> str:
        words = content.strip().split()
        if not words:
            return "Untitled BH note"
        snippet = " ".join(words[:max_words])
        return f"BH - {snippet}"

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTS

    @staticmethod
    def _is_audio_file(path: Path) -> bool:
        return path.suffix.lower() in AUDIO_EXTS

    def _ingest_audio_body(self, path: Path, transcribe_audio: bool) -> str:
        if not transcribe_audio:
            return f"Attachment ingested via bh add: {path.name}\n\n(Transcription skipped.)"

        if not self.transcriber:
            return f"Attachment ingested via bh add: {path.name}\n\n(No transcriber configured.)"

        try:
            transcript = self.transcriber.transcribe_file(path)
            return f"Transcript for {path.name}\n\n{transcript}"
        except TranscriptionError as exc:
            print(f"Warning: transcription failed for {path}: {exc}")
            return f"Attachment ingested via bh add: {path.name}\n\n(Transcription failed.)"
    def _render_todo_doc_llm(self, notes: List[Note], run_ts: datetime, scope: str) -> str:
        lines = self._render_header("Black Hole TODOs", run_ts, scope)

        if not notes:
            lines.append("_No TODO candidates in scanned notes._")
            return "\n".join(lines)

        prompt = self._format_notes_for_llm(notes, max_chars=1400)
        system = (
            "You are a Black Hole assistant that extracts TODO items from notes. "
            "Return concise markdown bullets. "
            "Each bullet: '- [Title](note-id): task summary (due YYYY-MM-DD or none)'. "
            "Infer due dates if present; keep summaries short."
        )
        user_prompt = f"Notes to scan:\n{prompt}\n\nReturn only markdown bullets."
        llm_output = self.llm.complete(system, user_prompt) or "_LLM returned no content._"

        lines.append(llm_output.strip())
        return "\n".join(lines)

    def _render_brief_doc_llm(self, notes: List[Note], run_ts: datetime, scope: str) -> str:
        lines = self._render_header("Black Hole Briefings", run_ts, scope)
        lines.append(f"Notes processed: {len(notes)}")
        lines.append("")

        if not notes:
            lines.append("_No new notes to brief._")
            return "\n".join(lines)

        prompt = self._format_notes_for_llm(notes, max_chars=1600)
        system = (
            "You are a context summarizer for a Black Hole notes system. "
            "Group notes by categories using their tags (or 'uncategorized'). "
            "Produce markdown sections with headings per category, then bullets of 'Title (note-id): one-line gist'. "
            "Stay concise; no extra prose."
        )
        user_prompt = f"Notes to brief:\n{prompt}\n\nReturn markdown sections."
        llm_output = self.llm.complete(system, user_prompt) or "_LLM returned no content._"

        lines.append(llm_output.strip())
        return "\n".join(lines)

    def _render_digest_doc_llm(self, notes: List[Note], run_ts: datetime, scope: str) -> str:
        lines = self._render_header("Black Hole Digest", run_ts, scope)
        lines.append(f"Notes processed: {len(notes)}")
        lines.append("")

        if not notes:
            lines.append("_No notes to digest._")
            return "\n".join(lines)

        prompt = self._format_notes_for_llm(notes, max_chars=1600)
        system = (
            "You are a digest writer for a Black Hole notes system. "
            "Produce a concise markdown digest with sections: Summary, Highlights (bullets), TODOs (bullets with inferred due dates), "
            "and Notable Links (titles with note-ids). Keep it tight."
        )
        user_prompt = f"Notes to digest:\n{prompt}\n\nReturn markdown with those sections."
        llm_output = self.llm.complete(system, user_prompt) or "_LLM returned no content._"

        lines.append(llm_output.strip())
        return "\n".join(lines)

    @staticmethod
    def _render_header(title: str, run_ts: datetime, scope: str) -> List[str]:
        return [
            f"# {title}",
            f"Run: {run_ts.isoformat()}",
            f"Scope: {scope}",
            "",
        ]

    @staticmethod
    def _format_notes_for_llm(notes: List[Note], max_chars: int = 1200) -> str:
        chunks = []
        for note in notes:
            body = note.content.strip()
            if len(body) > max_chars:
                body = body[: max_chars - 3] + "..."
            tags = ", ".join(note.tags) if note.tags else "uncategorized"
            chunks.append(
                f"Note ID: {note.id}\nTitle: {note.title}\nTags: {tags}\nContent:\n{body}"
            )
        return "\n\n".join(chunks)

    def _llm_tags(self, note: Note, max_tags: int = 8) -> List[str]:
        prompt = (
            "Return comma-separated concise tags (1-3 words, lowercase) for this note. "
            "No explanations."
        )
        user = f"Title: {note.title}\nContent:\n{note.content[:1200]}"
        raw = self.llm.complete(prompt, user) or ""
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return tags[:max_tags]

    @staticmethod
    def _read_clipboard() -> Optional[str]:
        for cmd in [["pbpaste"], ["xclip", "-selection", "clipboard", "-o"], ["wl-paste"]]:
            if shutil.which(cmd[0]):
                try:
                    output = subprocess.check_output(cmd, text=True).strip()
                    if output:
                        return output
                except Exception:
                    continue
        return None

    def status(self):
        last = self.state.data.get("last_processed", {})
        watches = self.state.watch_paths()
        print("Black Hole status")
        print(f"- Notes path: {self.storage.base_path}")
        print(f"- Total notes: {len(self.storage.list_notes())}")
        print(f"- Last TODO run: {last.get('todo', 'never')}")
        print(f"- Last brief run: {last.get('brief', 'never')}")
        if watches:
            print("- Watched paths:")
            for path, ts in watches.items():
                print(f"  - {path} (last seen: {ts})")
        else:
            print("- Watched paths: none")

    def browse(self, limit: int = 15, sort_by: str = "modified"):
        notes = list(self._sorted_notes(sort_by))[:limit]
        if not notes:
            print("No notes available.")
            return
        print(f"Browsing {len(notes)} notes (sorted by {sort_by}):\n")
        for note in notes:
            snippet = note.content.strip().splitlines()
            preview = snippet[0] if snippet else ""
            if len(preview) > 140:
                preview = preview[:137] + "..."
            tags = f" [{', '.join(note.tags)}]" if note.tags else ""
            print(f"- {note.title}{tags}")
            print(f"  ID: {note.id}")
            print(f"  Modified: {note.modified_at.isoformat()}")
            if preview:
                print(f"  {preview}")
            print()

    def show_note(self, note_id: str):
        note = self.storage.load_note(note_id)
        if not note:
            print(f"Note not found: {note_id}")
            return
        print(f"# {note.title} ({note.id})")
        if note.tags:
            print(f"Tags: {', '.join(note.tags)}")
        print(f"Created: {note.created_at.isoformat()}")
        print(f"Modified: {note.modified_at.isoformat()}")
        print("")
        print(note.content.strip())

    def open_note(self, note_id: str):
        note = self.storage.load_note(note_id)
        if not note:
            print(f"Note not found: {note_id}")
            return
        content_path = self.storage._content_path(note_id)
        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, str(content_path)])

    def yank(self):
        text = self._read_clipboard()
        if not text:
            print("Clipboard is empty or unavailable.")
            return
        self.add(text)

    def retag(self, scope: str = "new"):
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "retag"))
        if not notes:
            print("No notes to retag.")
            return
        for note in notes:
            tags = self._llm_tags(note)
            if tags:
                note.tags = tags
                self.storage.save_note(note)
                self.search.index_note(note)
        self.state.update_last_processed("retag", run_ts)
        print(f"Retagged {len(notes)} notes.")

    def lint(self):
        issues = []
        for note_id in self.storage.list_notes():
            note = self.storage.load_note(note_id)
            if not note:
                issues.append(f"{note_id}: metadata missing")
                continue
            content_path = self.storage._content_path(note_id)
            if not content_path.exists():
                issues.append(f"{note_id}: content.md missing")
            for att in note.attachments:
                if not self.storage.get_attachment_path(note_id, att.filename):
                    issues.append(f"{note_id}: attachment missing -> {att.filename}")
        if not issues:
            print("Lint OK: no issues found.")
            return
        print("Lint issues:")
        for issue in issues:
            print(f"- {issue}")

    def link(self, note_id: str, min_weight: float = 0.0):
        note = self.storage.load_note(note_id)
        if not note:
            print(f"Note not found: {note_id}")
            return
        self.graph.update_note_links(note_id)
        links = self.graph.get_linked_notes(note_id, min_weight=min_weight)
        if not links:
            print("No links found.")
            return
        print(f"Links for {note.title}:")
        for target_id, weight in links:
            target = self.storage.load_note(target_id)
            title = target.title if target else target_id
            print(f"- {title} ({target_id}): {weight:.3f}")

    def watch_directory(self, path: Path, interval: int = 15, follow: bool = False):
        target = path.expanduser().resolve()
        if not target.exists() or not target.is_dir():
            print(f"Directory does not exist: {target}")
            return

        print(f"Ingesting from {target} (follow={follow}, interval={interval}s)")
        self._ingest_directory(target, since=self.state.get_watch_timestamp(target))
        self.state.update_watch_timestamp(target, datetime.now(timezone.utc))

        if not follow:
            print("One-shot sync complete.")
            return

        try:
            while True:
                time.sleep(interval)
                self._ingest_directory(target, since=self.state.get_watch_timestamp(target))
                self.state.update_watch_timestamp(target, datetime.now(timezone.utc))
        except KeyboardInterrupt:
            print("Stopped watching.")

    def _sorted_notes(self, sort_by: str) -> Iterable[Note]:
        note_ids = self.storage.list_notes()
        notes = []
        for note_id in note_ids:
            note = self.storage.load_note(note_id)
            if note:
                notes.append(note)

        if sort_by == "created":
            notes.sort(key=lambda n: n.created_at, reverse=True)
        elif sort_by == "title":
            notes.sort(key=lambda n: n.title.lower())
        else:
            notes.sort(key=lambda n: n.modified_at, reverse=True)
        return notes

    def _ingest_directory(self, root: Path, since: Optional[datetime] = None):
        ingested = 0
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            if since and mtime <= since:
                continue
            try:
                self.add(str(file_path))
                ingested += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to ingest {file_path}: {exc}")
        if ingested:
            print(f"Ingested {ingested} new files from {root}")
        else:
            print(f"No new files found in {root}")

    def _upsert_special_note(self, note_id: str, title: str, content: str, tags: List[str]) -> str:
        existing = self.storage.load_note(note_id)
        if existing:
            self.storage.update_note_content(note_id, content, title)
            note = self.storage.load_note(note_id)
        else:
            note = Note(id=note_id, title=title, content=content, tags=tags)
            self.storage.save_note(note)

        if note:
            self.search.index_note(note)
            self.graph.update_note_links(note.id)
        return note_id


def main():
    parser = argparse.ArgumentParser(description="Black Hole CLI")
    parser.add_argument("--base-path", type=Path, help="Base path for storage (defaults to ~/.notes)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add raw content or ingest a file")
    add_parser.add_argument("content", help="Content text or path to a file")
    add_parser.add_argument("-t", "--title", help="Optional title override")
    add_parser.add_argument("--tags", nargs="+", help="Optional tags")
    add_parser.add_argument("--auto-tags", action="store_true", help="Use configured tagger to suggest tags")
    add_parser.add_argument("--no-transcribe", action="store_true", help="Skip audio transcription")

    search_parser = subparsers.add_parser("search", help="Semantic/hybrid search")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("-l", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument(
        "-m",
        "--mode",
        choices=["semantic", "keyword", "hybrid"],
        default="semantic",
        help="Search mode",
    )

    proactive_parser = subparsers.add_parser("proactive", help="Proactive routines (todo, brief)")
    proactive_sub = proactive_parser.add_subparsers(dest="mode", required=True)

    todo_parser = proactive_sub.add_parser("todo", help="Aggregate TODOs into a superstructure note")
    todo_parser.add_argument(
        "--scope",
        choices=["new", "all"],
        default="new",
        help="Process only new notes or all notes",
    )

    brief_parser = proactive_sub.add_parser("brief", help="Generate context briefings grouped by category/tag")
    brief_parser.add_argument(
        "--scope",
        choices=["new", "all"],
        default="new",
        help="Process only new notes or all notes",
    )

    digest_parser = subparsers.add_parser("digest", help="Generate a digest of notes")
    digest_parser.add_argument(
        "--scope",
        choices=["new", "all"],
        default="new",
        help="Process only new notes or all notes",
    )

    browse_parser = subparsers.add_parser("browse", help="Traverse notes in a compact view")
    browse_parser.add_argument("-l", "--limit", type=int, default=15, help="Number of notes to show")
    browse_parser.add_argument(
        "-s",
        "--sort",
        choices=["modified", "created", "title"],
        default="modified",
        help="Sort order",
    )

    show_parser = subparsers.add_parser("show", help="Render a specific note")
    show_parser.add_argument("note_id", help="Note ID to render")

    open_parser = subparsers.add_parser("open", help="Open a note in $EDITOR")
    open_parser.add_argument("note_id", help="Note ID to open")

    subparsers.add_parser("status", help="Show Black Hole status")

    watch_parser = subparsers.add_parser("create-bh", help="Watch a directory and ingest new files")
    watch_parser.add_argument("path", type=Path, help="Directory to watch")
    watch_parser.add_argument("--interval", type=int, default=15, help="Polling interval seconds when following")
    watch_parser.add_argument(
        "--follow",
        action="store_true",
        help="Continue watching instead of a single sync pass",
    )

    retag_parser = subparsers.add_parser("retag", help="LLM-retag notes")
    retag_parser.add_argument(
        "--scope",
        choices=["new", "all"],
        default="new",
        help="Process only new notes or all notes",
    )

    subparsers.add_parser("lint", help="Check storage/index health")

    link_parser = subparsers.add_parser("link", help="Refresh and show links for a note")
    link_parser.add_argument("note_id", help="Note ID to link")
    link_parser.add_argument("--min-weight", type=float, default=0.0, help="Minimum weight to show")

    subparsers.add_parser("yank", help="Create a note from clipboard text")

    args = parser.parse_args()

    cli = BlackHoleCLI(args.base_path)
    try:
        if args.command == "add":
            cli.add(
                args.content,
                title=args.title,
                tags=args.tags,
                auto_tags=args.auto_tags,
                transcribe_audio=not args.no_transcribe,
            )
        elif args.command == "search":
            cli.bh_search(args.query, limit=args.limit, mode=args.mode)
        elif args.command == "proactive":
            if args.mode == "todo":
                cli.proactive_todo(scope=args.scope)
            elif args.mode == "brief":
                cli.proactive_brief(scope=args.scope)
        elif args.command == "digest":
            cli.digest(scope=args.scope)
        elif args.command == "browse":
            cli.browse(limit=args.limit, sort_by=args.sort)
        elif args.command == "show":
            cli.show_note(args.note_id)
        elif args.command == "open":
            cli.open_note(args.note_id)
        elif args.command == "status":
            cli.status()
        elif args.command == "create-bh":
            cli.watch_directory(args.path, interval=args.interval, follow=args.follow)
        elif args.command == "retag":
            cli.retag(scope=args.scope)
        elif args.command == "lint":
            cli.lint()
        elif args.command == "link":
            cli.link(args.note_id, min_weight=args.min_weight)
        elif args.command == "yank":
            cli.yank()
    finally:
        cli.close()


if __name__ == "__main__":
    main()
