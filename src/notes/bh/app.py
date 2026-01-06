from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from ..bh_llm import BHLLM
from ..cli import NotesCLI
from ..datamodel import Note
from ..transcription import TranscriptionError, WhisperTranscriber
from .state import BlackHoleState


TEXT_EXTS = {".txt", ".md", ".rst"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


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
        super().__init__(base_path=base_path)
        self.state = BlackHoleState(self.storage.db_dir / "bh_state.json")
        self.transcriber = transcriber or WhisperTranscriber.from_config(self.config)
        llm_model = (self.config.get("llm") or {}).get("model", "gpt-5-mini")
        self.llm = llm or BHLLM(model=llm_model)

    def close(self):
        self.search.close()

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
    ) -> str:
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

        note_id = self._create_note_quiet(
            note_title,
            body,
            tags=tags,
            attachments=attachments,
            auto_tags=auto_tags,
        )
        return note_id

    def search_bh(self, query: str, limit: int = 10, mode: str = "semantic"):
        if mode == "keyword":
            return self.search.keyword_search_notes(query, limit)
        if mode == "hybrid":
            return self.search.hybrid_search(query, limit)
        return self.search.semantic_search_notes(query, limit)

    def proactive_todo(self, scope: str = "new") -> str:
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "todo"))
        content = self._render_todo_doc_llm(notes, run_ts, scope)
        note_id = self._upsert_special_note(
            self.TODO_NOTE_ID, "BH TODOs", content, tags=["blackhole", "todo"]
        )
        self.state.update_last_processed("todo", run_ts)
        return note_id

    def proactive_brief(self, scope: str = "new") -> str:
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "brief"))
        content = self._render_brief_doc_llm(notes, run_ts, scope)
        note_id = self._upsert_special_note(
            self.BRIEF_NOTE_ID, "BH Briefings", content, tags=["blackhole", "brief"]
        )
        self.state.update_last_processed("brief", run_ts)
        return note_id

    def digest(self, scope: str = "new") -> str:
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "digest"))
        content = self._render_digest_doc_llm(notes, run_ts, scope)
        note_id = self._upsert_special_note(
            self.DIGEST_NOTE_ID, "BH Digest", content, tags=["blackhole", "digest"]
        )
        self.state.update_last_processed("digest", run_ts)
        return note_id

    def retag(self, scope: str = "new") -> int:
        run_ts = datetime.now(timezone.utc)
        notes = list(self._select_notes(scope, "retag"))
        updated = 0
        for note in notes:
            tags = self._llm_tags(note)
            if tags:
                note.tags = tags
                self.storage.save_note(note)
                self.search.index_note(note)
                updated += 1
        self.state.update_last_processed("retag", run_ts)
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

    def browse(self, limit: int = 15, sort_by: str = "modified") -> List[Note]:
        return list(self._sorted_notes(sort_by))[:limit]

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

    def yank(self) -> Optional[str]:
        text = self._read_clipboard()
        if not text:
            return None
        self.add(text)
        return text

    def ingest_directory_once(self, path: Path) -> List[str]:
        target = path.expanduser().resolve()
        if not target.exists() or not target.is_dir():
            return []
        since = self.state.get_watch_timestamp(target)
        note_ids = self._ingest_directory(target, since=since)
        self.state.update_watch_timestamp(target, datetime.now(timezone.utc))
        return note_ids

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

    def _create_note_quiet(
        self,
        title: str,
        content: str = "",
        tags: list = None,
        attachments: list = None,
        auto_tags: bool = False,
    ) -> str:
        final_tags = tags or []
        if auto_tags and self.tagger:
            try:
                suggested = self.tagger.suggest_tags(title, content)
                final_tags = self._merge_tags(final_tags, suggested)
            except Exception:
                pass

        note = Note(title=title, content=content, tags=final_tags)
        self.storage.save_note(note)

        if attachments:
            for attachment_path in attachments:
                file_path = Path(attachment_path)
                if file_path.exists():
                    self.storage.add_attachment(note.id, file_path)

        self.search.index_note(note)
        self.graph.update_note_links(note.id)
        return note.id

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

    def _render_todo_doc_llm(self, notes: List[Note], run_ts: datetime, scope: str) -> str:
        lines = self._render_header("Black Hole TODOs", run_ts, scope)
        if not notes:
            lines.append("_No TODO candidates in scanned notes._")
            return "\n".join(lines)
        prompt = self._format_notes_for_llm(notes, max_chars=1400)
        system = (
            "You extract TODO items from notes. Return concise markdown bullets. "
            "Each bullet: '- [Title](note-id): task summary (due YYYY-MM-DD or none)'."
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
            "Group notes by categories using their tags (or 'uncategorized'). "
            "Produce markdown headings per category, then bullets: 'Title (note-id): one-line gist'. "
            "No extra prose."
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
            "Produce a concise markdown digest with sections: Summary, Highlights (bullets), TODOs (bullets with inferred due dates), "
            "and Notable Links (titles with note-ids)."
        )
        user_prompt = f"Notes to digest:\n{prompt}\n\nReturn markdown with those sections."
        llm_output = self.llm.complete(system, user_prompt) or "_LLM returned no content._"
        lines.append(llm_output.strip())
        return "\n".join(lines)

    @staticmethod
    def _render_header(title: str, run_ts: datetime, scope: str) -> List[str]:
        return [f"# {title}", f"Run: {run_ts.isoformat()}", f"Scope: {scope}", ""]

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
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
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

    def _ingest_directory(self, root: Path, since: Optional[datetime] = None) -> List[str]:
        note_ids: List[str] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            if since and mtime <= since:
                continue
            try:
                note_ids.append(self.add(str(file_path)))
            except Exception:
                continue
        return note_ids

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

