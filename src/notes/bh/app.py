from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from ..bh_llm import BHLLM
from ..cli import NotesCLI
from ..datamodel import Attachment, Note
from ..providers_openai import OpenAIEmbeddingProvider, OpenAITagger
from ..streams import normalize_stream, stream_matches
from ..tags import normalize_tags
from ..transcription import TranscriptionError, WhisperTranscriber
from .db import BlackHoleDB
from .ingestion import AUDIO_EXTS, TEXT_EXTS, AddOptions, IngestionPipeline
from .state import BlackHoleState


@dataclass
class LinkResult:
    note_id: str
    title: str
    weight: float


class BlackHoleApp(NotesCLI):
    """
    BH operations built atop storage/search/graph.

    This class avoids argparse/printing where possible so it can be reused by future UIs.
    """

    TODO_NOTE_ID = "bh-todo"
    BRIEF_NOTE_ID = "bh-brief"
    DIGEST_NOTE_ID = "bh-digest"

    def __init__(
        self,
        base_path: Optional[Path] = None,
        transcriber: Optional[WhisperTranscriber] = None,
        llm: Optional[BHLLM] = None,
    ):
        if base_path is None:
            base_path = Path.home() / ".notes"
        self.db = BlackHoleDB(Path(base_path) / ".notes_db" / "bh.db")
        super().__init__(base_path=base_path)
        self.state = BlackHoleState(self.storage.db_dir / "bh_state.json")
        self.ingestion = IngestionPipeline()
        self.transcriber = transcriber or WhisperTranscriber.from_config(
            self.config, event_recorder=self._record_cost_event
        )
        llm_model = (self.config.get("llm") or {}).get("model", "gpt-5-mini")
        self.llm = llm or BHLLM(model=llm_model, event_recorder=self._record_cost_event)

    def close(self):
        self.search.close()
        self.db.close()

    def _record_cost_event(self, event: dict) -> None:
        """
        Record a single cost/usage event (best-effort).
        Expected keys: provider, operation, model, request_id, usage, duration_seconds, cost_usd, metadata.
        """
        try:
            usage = (event.get("usage") or {}) if isinstance(event, dict) else {}
            metadata = (event.get("metadata") or {}) if isinstance(event, dict) else {}
            if usage:
                metadata = {**metadata, "usage": usage}
            self.db.record_cost_event(
                provider=event.get("provider", "unknown"),
                operation=event.get("operation", "unknown"),
                model=event.get("model"),
                request_id=event.get("request_id"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                duration_seconds=event.get("duration_seconds"),
                cost_usd=event.get("cost_usd"),
                metadata=metadata,
            )
        except Exception:
            pass

    def cost_daily(self, days: int = 14) -> list[dict]:
        return self.db.daily_costs(days=days)

    def cost_events(self, day: Optional[str] = None, limit: int = 50) -> list[dict]:
        return self.db.list_cost_events(day=day, limit=limit)

    def _build_embedding_provider(self):
        provider_cfg = self.config.get("embedding_provider", {}) or {}
        provider_type = provider_cfg.get("type")
        if provider_type == "openai":
            return OpenAIEmbeddingProvider(
                model=provider_cfg.get("model", "text-embedding-3-small"),
                event_recorder=self._record_cost_event,
            )
        return None

    def _build_tagger(self):
        tagger_cfg = self.config.get("tagger", {}) or {}
        tagger_type = tagger_cfg.get("type")
        if tagger_type == "openai":
            return OpenAITagger(
                model=tagger_cfg.get("model", "gpt-4o-mini"),
                event_recorder=self._record_cost_event,
            )
        return None

    # -----------------------------
    # Core BH commands
    # -----------------------------
    def add(
        self,
        content: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_tags: bool = False,
        transcribe_audio: bool = True,
        stream: Optional[str] = None,
    ) -> str | List[str]:
        opts = AddOptions(
            title=title,
            tags=tags,
            auto_tags=auto_tags,
            transcribe_audio=transcribe_audio,
            stream=stream,
        )
        return self.ingestion.ingest(self, content, opts)

    def search_bh(
        self, query: str, limit: int = 10, mode: str = "semantic", stream: Optional[str] = None
    ):
        multiplier = 5 if stream else 1
        if mode == "keyword":
            results = self.search.keyword_search_notes(query, limit * multiplier)
        elif mode == "hybrid":
            results = self.search.hybrid_search(query, limit * multiplier)
        else:
            results = self.search.semantic_search_notes(query, limit * multiplier)

        if not stream:
            return results[:limit]

        filtered = []
        for r in results:
            note = self.storage.load_note(r.note_id)
            if note and stream_matches(note.stream, stream):
                filtered.append(r)
            if len(filtered) >= limit:
                break
        return filtered

    def new_note(
        self,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_tags: bool = False,
        stream: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a markdown-first note by opening $EDITOR on a temporary file.
        Returns note_id if created, otherwise None (e.g. empty file).
        """
        editor = os.environ.get("EDITOR", "vim")
        tmp_dir = self.storage.db_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", suffix=".md", dir=str(tmp_dir), delete=False
        ) as fh:
            tmp_path = Path(fh.name)
            if title:
                fh.write(f"# {title}\n\n")
            else:
                fh.write("# \n\n")

        try:
            subprocess.run([editor, str(tmp_path)])
        except FileNotFoundError as exc:
            raise RuntimeError(f"EDITOR not found: {editor}") from exc

        raw = tmp_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        parsed_title, body = self._parse_markdown_note(raw, fallback_title=title)
        return self._create_note_quiet(
            parsed_title,
            body,
            tags=tags,
            attachments=None,
            auto_tags=auto_tags,
            stream=stream,
        )

    def new_todo(
        self,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_tags: bool = False,
        stream: Optional[str] = None,
        due: Optional[str] = None,
    ) -> Optional[str]:
        editor = os.environ.get("EDITOR", "vim")
        tmp_dir = self.storage.db_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", suffix=".md", dir=str(tmp_dir), delete=False
        ) as fh:
            tmp_path = Path(fh.name)
            if title:
                fh.write(f"# {title}\n\n")
            else:
                fh.write("# \n\n")

        try:
            subprocess.run([editor, str(tmp_path)])
        except FileNotFoundError as exc:
            raise RuntimeError(f"EDITOR not found: {editor}") from exc

        raw = tmp_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        parsed_title, body = self._parse_markdown_note(raw, fallback_title=title)
        due_at = self._parse_optional_datetime(due)
        return self._create_note_quiet(
            parsed_title,
            body,
            tags=tags,
            attachments=None,
            auto_tags=auto_tags,
            stream=stream,
            kind="todo",
            due_at=due_at,
        )

    @staticmethod
    def _parse_markdown_note(raw: str, fallback_title: Optional[str] = None) -> tuple[str, str]:
        lines = raw.splitlines()
        title = fallback_title
        body_lines = lines
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            body_lines = lines[2:] if len(lines) > 2 else []
        if not title:
            title = BlackHoleApp._title_from_text(raw)
        return title, "\n".join(body_lines).lstrip("\n")

    @staticmethod
    def _parse_optional_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def _state_key(mode: str, stream: Optional[str]) -> str:
        if not stream:
            return mode
        return f"{mode}:{normalize_stream(stream)}"

    @staticmethod
    def _special_note_id(base_id: str, stream: Optional[str]) -> str:
        if not stream:
            return base_id
        suffix = normalize_stream(stream).replace("/", "__")
        return f"{base_id}--{suffix}"

    def proactive_todo(self, scope: str = "new", stream: Optional[str] = None) -> str:
        run_ts = datetime.now(timezone.utc)
        key = self._state_key("todo", stream)
        last_ts = self.state.get_last_processed(key) if scope == "new" else None
        now = run_ts
        mature_window = timedelta(days=7)
        horizon = now + mature_window

        open_todos: List[Note] = []
        recent_done: List[Note] = []

        for note_id in self.storage.list_notes():
            note = self.storage.load_note(note_id)
            if not note or note.kind != "todo":
                continue
            if note.archived_at is not None:
                continue
            if not stream_matches(note.stream, stream):
                continue

            if note.completed_at is not None:
                if last_ts and note.completed_at > last_ts:
                    recent_done.append(note)
                continue

            if last_ts is None:
                open_todos.append(note)
                continue

            include = note.modified_at > last_ts
            if note.due_at is not None and note.due_at <= horizon:
                include = True
            if include:
                open_todos.append(note)

        content = self._render_todo_dashboard(
            open_todos,
            recent_done,
            run_ts,
            scope,
            stream=stream,
            mature_days=int(mature_window.days),
        )
        note_id = self._upsert_special_note(
            self._special_note_id(self.TODO_NOTE_ID, stream),
            "BH TODOs" if not stream else f"BH TODOs ({normalize_stream(stream)})",
            content,
            tags=["blackhole", "todo"],
            stream=normalize_stream(stream) if stream else "blackhole/system",
        )
        self.state.update_last_processed(key, run_ts)
        return note_id

    def proactive_brief(self, scope: str = "new", stream: Optional[str] = None) -> str:
        run_ts = datetime.now(timezone.utc)
        key = self._state_key("brief", stream)
        notes = list(self._select_notes(scope, key, stream=stream))
        content = self._render_brief_doc_llm(notes, run_ts, scope, stream=stream)
        note_id = self._upsert_special_note(
            self._special_note_id(self.BRIEF_NOTE_ID, stream),
            "BH Briefings" if not stream else f"BH Briefings ({normalize_stream(stream)})",
            content,
            tags=["blackhole", "brief"],
            stream=normalize_stream(stream) if stream else "blackhole/system",
        )
        self.state.update_last_processed(key, run_ts)
        return note_id

    def digest(self, scope: str = "new", stream: Optional[str] = None) -> str:
        run_ts = datetime.now(timezone.utc)
        key = self._state_key("digest", stream)
        notes = list(self._select_notes(scope, key, stream=stream))
        content = self._render_digest_doc_llm(notes, run_ts, scope, stream=stream)
        note_id = self._upsert_special_note(
            self._special_note_id(self.DIGEST_NOTE_ID, stream),
            "BH Digest" if not stream else f"BH Digest ({normalize_stream(stream)})",
            content,
            tags=["blackhole", "digest"],
            stream=normalize_stream(stream) if stream else "blackhole/system",
        )
        self.state.update_last_processed(key, run_ts)
        return note_id

    def retag(self, scope: str = "new", stream: Optional[str] = None) -> int:
        run_ts = datetime.now(timezone.utc)
        key = self._state_key("retag", stream)
        notes = list(self._select_notes(scope, key, stream=stream))
        updated = 0
        for note in notes:
            tags = self._llm_tags(note)
            if tags:
                note.tags = tags
                self.storage.save_note(note)
                self.search.index_note(note)
                updated += 1
        self.state.update_last_processed(key, run_ts)
        return updated

    def lint(self) -> List[str]:
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
        return issues

    def link(self, note_id: str, min_weight: float = 0.0) -> List[LinkResult]:
        note = self.storage.load_note(note_id)
        if not note:
            return []
        self.graph.update_note_links(note_id)
        links = self.graph.get_linked_notes(note_id, min_weight=min_weight)
        out: List[LinkResult] = []
        for target_id, weight in links:
            target = self.storage.load_note(target_id)
            out.append(LinkResult(note_id=target_id, title=(target.title if target else target_id), weight=weight))
        return out

    def status(self) -> dict:
        last = self.state.data.get("last_processed", {}) or {}
        return {
            "notes_path": str(self.storage.base_path),
            "total_notes": len(self.storage.list_notes()),
            "last_processed": last,
            "watches": self.state.watch_paths(),
        }

    def browse(
        self, limit: int = 15, sort_by: str = "modified", stream: Optional[str] = None
    ) -> List[Note]:
        notes = [n for n in self._sorted_notes(sort_by) if stream_matches(n.stream, stream)]
        return notes[:limit]

    def completion_stream(self, stream: str, limit: int = 500) -> str:
        """
        Build a deterministic "completionist" index note for a stream subtree.
        """
        run_ts = datetime.now(timezone.utc)
        norm_stream = normalize_stream(stream)
        note_id = self._special_note_id("bh-index", norm_stream)

        items = [
            n
            for n in self._sorted_notes("modified")
            if stream_matches(n.stream, norm_stream) and n.id != note_id
        ][:limit]

        lines = [
            f"# Stream Index: {norm_stream}",
            f"Run: {run_ts.isoformat()}",
            f"Stream: {norm_stream}",
            f"Items: {len(items)}",
            "",
        ]

        for item in items:
            tags = f" [{', '.join(item.tags)}]" if item.tags else ""
            if item.kind == "todo":
                state = "archived" if item.archived_at else ("done" if item.completed_at else "open")
                due = f" due {item.due_at.date().isoformat()}" if item.due_at else ""
                lines.append(f"- todo [{state}] {item.title}{tags}{due} ({item.id})")
            else:
                lines.append(f"- note {item.title}{tags} ({item.id})")

        content = "\n".join(lines)
        return self._upsert_special_note(
            note_id,
            f"BH Stream Index ({norm_stream})",
            content,
            tags=["blackhole", "index"],
            stream=norm_stream,
        )

    def create_todo(
        self,
        text: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_tags: bool = False,
        due: Optional[str] = None,
        stream: Optional[str] = None,
    ) -> str:
        todo_title = title or self._todo_title_from_text(text)
        due_at = self._parse_optional_datetime(due)
        return self._create_note_quiet(
            todo_title,
            text,
            tags=tags,
            attachments=None,
            auto_tags=auto_tags,
            stream=stream,
            kind="todo",
            due_at=due_at,
        )

    def list_todos(
        self,
        status: str = "open",
        stream: Optional[str] = None,
        limit: int = 25,
        sort_by: str = "due",
    ) -> List[Note]:
        todos = [
            n
            for n in self._sorted_notes("modified")
            if n.kind == "todo" and stream_matches(n.stream, stream)
        ]

        if status == "open":
            todos = [t for t in todos if t.archived_at is None and t.completed_at is None]
        elif status == "done":
            todos = [t for t in todos if t.archived_at is None and t.completed_at is not None]
        elif status == "archived":
            todos = [t for t in todos if t.archived_at is not None]

        if sort_by == "created":
            todos.sort(key=lambda t: t.created_at, reverse=True)
        elif sort_by == "title":
            todos.sort(key=lambda t: t.title.lower())
        elif sort_by == "modified":
            todos.sort(key=lambda t: t.modified_at, reverse=True)
        else:  # due
            far_future = datetime.max.replace(tzinfo=timezone.utc)
            todos.sort(key=lambda t: (t.due_at or far_future, t.modified_at), reverse=False)

        return todos[:limit]

    def mark_todo_done(self, todo_id: str) -> bool:
        todo = self.storage.load_note(todo_id)
        if not todo or todo.kind != "todo":
            return False
        todo.completed_at = datetime.now(timezone.utc)
        todo.modified_at = datetime.now(timezone.utc)
        self.storage.save_note(todo)
        self.search.index_note(todo)
        self.graph.update_note_links(todo.id)
        return True

    def archive_item(self, item_id: str) -> bool:
        note = self.storage.load_note(item_id)
        if not note:
            return False
        note.archived_at = datetime.now(timezone.utc)
        note.modified_at = datetime.now(timezone.utc)
        self.storage.save_note(note)
        self.search.index_note(note)
        self.graph.update_note_links(note.id)
        return True

    @staticmethod
    def _todo_title_from_text(text: str, max_len: int = 80) -> str:
        first = (text.strip().splitlines() or [""])[0].strip()
        if not first:
            return "Untitled TODO"
        return first if len(first) <= max_len else first[: max_len - 3] + "..."

    def get_note(self, note_id: str) -> Optional[Note]:
        return self.storage.load_note(note_id)

    def open_note(self, note_id: str) -> bool:
        note = self.storage.load_note(note_id)
        if not note:
            return False
        content_path = self.storage._content_path(note_id)
        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, str(content_path)])
        return True

    def yank(self, stream: Optional[str] = None) -> Optional[str]:
        text = self._read_clipboard()
        if not text:
            return None
        self.add(text, stream=stream)
        return text

    def ingest_directory_once(self, path: Path, stream: Optional[str] = None) -> List[str]:
        target = path.expanduser().resolve()
        if not target.exists() or not target.is_dir():
            return []
        since = self.state.get_watch_timestamp(target)
        note_ids = self._ingest_directory(target, since=since, stream=stream)
        self.state.update_watch_timestamp(target, datetime.now(timezone.utc))
        return note_ids

    def ingest_directory_all(self, path: Path, stream: Optional[str] = None) -> List[str]:
        target = path.expanduser().resolve()
        if not target.exists() or not target.is_dir():
            return []
        return self._ingest_directory(target, since=None, stream=stream)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _select_notes(self, scope: str, key: str, stream: Optional[str] = None) -> Iterable[Note]:
        last_ts = self.state.get_last_processed(key) if scope == "new" else None
        for note_id in self.storage.list_notes():
            note = self.storage.load_note(note_id)
            if not note:
                continue
            if last_ts and note.modified_at <= last_ts:
                continue
            if not stream_matches(note.stream, stream):
                continue
            yield note

    def _create_note_quiet(
        self,
        title: str,
        content: str = "",
        tags: list = None,
        attachments: list = None,
        auto_tags: bool = False,
        stream: Optional[str] = None,
        kind: str = "note",
        due_at: Optional[datetime] = None,
    ) -> str:
        final_tags = normalize_tags(tags or [])
        if auto_tags and self.tagger:
            try:
                suggested = self.tagger.suggest_tags(title, content)
                final_tags = self._merge_tags(final_tags, suggested)
            except Exception:
                pass

        note = Note(
            kind=kind,
            title=title,
            content=content,
            tags=final_tags,
            stream=normalize_stream(stream),
            due_at=due_at,
        )
        self.storage.save_note(note)

        added_attachments: List[Attachment] = []
        if attachments:
            for attachment_path in attachments:
                file_path = Path(attachment_path)
                if file_path.exists():
                    attachment = self.storage.add_attachment(note.id, file_path)
                    if attachment:
                        added_attachments.append(attachment)

        self._record_sources(note.id, added_attachments)

        self.search.index_note(note)
        self.graph.update_note_links(note.id)
        return note.id

    def _record_sources(self, note_id: str, attachments: List[Attachment]) -> None:
        if not attachments:
            return

        mapping: dict[tuple[str, str], str] = {}
        for attachment in attachments:
            if not attachment.sha256:
                continue
            source_id = self.db.upsert_source(attachment)
            stored_path = self.storage.get_attachment_path(note_id, attachment.filename)
            stored_relpath = None
            if stored_path:
                try:
                    stored_relpath = str(stored_path.relative_to(self.storage.base_path))
                except Exception:
                    stored_relpath = str(stored_path)
            self.db.add_source_ref(
                item_id=note_id,
                source_id=source_id,
                role="attachment",
                original_path=attachment.original_path,
                stored_relpath=stored_relpath,
            )
            mapping[(attachment.filename, attachment.sha256)] = source_id

        note = self.storage.load_note(note_id)
        if not note:
            return

        changed = False
        for att in note.attachments:
            if not att.sha256:
                continue
            source_id = mapping.get((att.filename, att.sha256))
            if source_id and att.source_id != source_id:
                att.source_id = source_id
                changed = True

        if changed:
            self.storage.save_note(note)

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
        except TranscriptionError:
            return f"Attachment ingested via bh add: {path.name}\n\n(Transcription failed.)"

    def _render_todo_dashboard(
        self,
        open_todos: List[Note],
        recent_done: List[Note],
        run_ts: datetime,
        scope: str,
        stream: Optional[str] = None,
        mature_days: int = 7,
    ) -> str:
        now = run_ts
        horizon = now + timedelta(days=mature_days)
        show_stream = stream is None

        overdue = [t for t in open_todos if t.due_at is not None and t.due_at < now]
        due_soon = [
            t for t in open_todos if t.due_at is not None and now <= t.due_at <= horizon
        ]
        later = [t for t in open_todos if t.due_at is not None and t.due_at > horizon]
        no_due = [t for t in open_todos if t.due_at is None]

        overdue.sort(key=lambda t: t.due_at or datetime.max.replace(tzinfo=timezone.utc))
        due_soon.sort(key=lambda t: t.due_at or datetime.max.replace(tzinfo=timezone.utc))
        later.sort(key=lambda t: t.due_at or datetime.max.replace(tzinfo=timezone.utc))
        no_due.sort(key=lambda t: t.modified_at, reverse=True)
        recent_done.sort(key=lambda t: t.completed_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        lines = self._render_header("Black Hole TODOs", run_ts, scope, stream=stream)
        lines.append(
            f"Open: {len(open_todos)} (overdue {len(overdue)}, due-soon {len(due_soon)}, later {len(later)}, no-due {len(no_due)})"
        )
        if scope == "new":
            lines.append(f"Maturity window: {mature_days} days")
        if recent_done:
            lines.append(f"Recently completed: {len(recent_done)}")
        lines.append("")

        if not open_todos and not recent_done:
            lines.append("_No todos in scope._")
            return "\n".join(lines)

        def fmt_open(todo: Note) -> str:
            parts = [f"- [ ] {todo.title} ({todo.id})"]
            if todo.due_at:
                parts.append(f"due={todo.due_at.date().isoformat()}")
            if show_stream:
                parts.append(f"stream={todo.stream}")
            if todo.tags:
                parts.append(f"tags={', '.join(todo.tags)}")
            return " ".join(parts)

        def fmt_done(todo: Note) -> str:
            parts = [f"- [x] {todo.title} ({todo.id})"]
            if todo.completed_at:
                parts.append(f"completed={todo.completed_at.date().isoformat()}")
            if show_stream:
                parts.append(f"stream={todo.stream}")
            if todo.tags:
                parts.append(f"tags={', '.join(todo.tags)}")
            return " ".join(parts)

        if overdue:
            lines.append("## Overdue")
            lines.extend([fmt_open(t) for t in overdue])
            lines.append("")
        if due_soon:
            lines.append("## Due Soon")
            lines.extend([fmt_open(t) for t in due_soon])
            lines.append("")
        if later:
            lines.append("## Later")
            lines.extend([fmt_open(t) for t in later])
            lines.append("")
        if no_due:
            lines.append("## No Due Date")
            lines.extend([fmt_open(t) for t in no_due])
            lines.append("")
        if recent_done:
            lines.append("## Recently Completed")
            lines.extend([fmt_done(t) for t in recent_done])
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_brief_doc_llm(
        self, notes: List[Note], run_ts: datetime, scope: str, stream: Optional[str] = None
    ) -> str:
        lines = self._render_header("Black Hole Briefings", run_ts, scope, stream=stream)
        lines.append(f"Notes processed: {len(notes)}")
        lines.append("")
        if not notes:
            lines.append("_No new notes to brief._")
            return "\n".join(lines)
        prompt = self._format_notes_for_llm(notes, max_chars=1600)
        system = (
            "Group notes by categories using their tags (or 'uncategorized'). "
            "Produce markdown headings per category, then bullets: 'Title (note-id): one-line gist'. "
            "No extra prose."
        )
        user_prompt = f"Notes to brief:\n{prompt}\n\nReturn markdown sections."
        llm_output = self.llm.complete(system, user_prompt) or "_LLM returned no content._"
        lines.append(llm_output.strip())
        return "\n".join(lines)

    def _render_digest_doc_llm(
        self, notes: List[Note], run_ts: datetime, scope: str, stream: Optional[str] = None
    ) -> str:
        lines = self._render_header("Black Hole Digest", run_ts, scope, stream=stream)
        lines.append(f"Notes processed: {len(notes)}")
        lines.append("")
        if not notes:
            lines.append("_No notes to digest._")
            return "\n".join(lines)
        prompt = self._format_notes_for_llm(notes, max_chars=1600)
        system = (
            "Produce a concise markdown digest with sections: Summary, Highlights (bullets), TODOs (bullets with inferred due dates), "
            "and Notable Links (titles with note-ids)."
        )
        user_prompt = f"Notes to digest:\n{prompt}\n\nReturn markdown with those sections."
        llm_output = self.llm.complete(system, user_prompt) or "_LLM returned no content._"
        lines.append(llm_output.strip())
        return "\n".join(lines)

    @staticmethod
    def _render_header(title: str, run_ts: datetime, scope: str, stream: Optional[str] = None) -> List[str]:
        lines = [f"# {title}", f"Run: {run_ts.isoformat()}", f"Scope: {scope}"]
        if stream:
            lines.append(f"Stream: {normalize_stream(stream)}")
        lines.append("")
        return lines

    @staticmethod
    def _format_notes_for_llm(notes: List[Note], max_chars: int = 1200) -> str:
        chunks = []
        for note in notes:
            body = note.content.strip()
            if len(body) > max_chars:
                body = body[: max_chars - 3] + "..."
            tags = ", ".join(note.tags) if note.tags else "uncategorized"
            chunks.append(f"Note ID: {note.id}\nTitle: {note.title}\nTags: {tags}\nContent:\n{body}")
        return "\n\n".join(chunks)

    def _llm_tags(self, note: Note, max_tags: int = 8) -> List[str]:
        system = "Return comma-separated concise tags (1-3 words, lowercase). No explanations."
        user = f"Title: {note.title}\nContent:\n{note.content[:1200]}"
        raw = self.llm.complete(system, user) or ""
        tags = normalize_tags([t.strip() for t in raw.split(",") if t.strip()])
        return tags[:max_tags]

    def _sorted_notes(self, sort_by: str) -> Iterable[Note]:
        note_ids = self.storage.list_notes()
        notes: List[Note] = []
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

    def _ingest_directory(
        self, root: Path, since: Optional[datetime] = None, stream: Optional[str] = None
    ) -> List[str]:
        note_ids: List[str] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            if since and mtime <= since:
                continue
            try:
                created = self.add(str(file_path), stream=stream)
                if isinstance(created, list):
                    note_ids.extend(created)
                else:
                    note_ids.append(created)
            except Exception:
                continue
        return note_ids

    def _upsert_special_note(
        self,
        note_id: str,
        title: str,
        content: str,
        tags: List[str],
        stream: Optional[str] = "blackhole/system",
    ) -> str:
        existing = self.storage.load_note(note_id)
        if existing:
            self.storage.update_note_content(note_id, content, title)
            note = self.storage.load_note(note_id)
        else:
            note = Note(id=note_id, title=title, content=content, tags=tags, stream=normalize_stream(stream))
            self.storage.save_note(note)
        if note:
            self.search.index_note(note)
            self.graph.update_note_links(note.id)
        return note_id

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
