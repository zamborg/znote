"""
Storage layer for the notes system.
Handles file system operations and persistence.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .datamodel import Attachment, MediaType, Note


class NotesStorage:
    """Manages file system storage for notes."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.notes_dir = self.base_path / "notes"
        self.db_dir = self.base_path / ".notes_db"
        self.index_dir = self.db_dir / "index"

        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _note_path(self, note_id: str) -> Path:
        """Get the directory path for a note."""
        return self.notes_dir / note_id

    def _metadata_path(self, note_id: str) -> Path:
        """Get the metadata file path for a note."""
        return self._note_path(note_id) / "metadata.json"

    def _content_path(self, note_id: str) -> Path:
        """Get the content file path for a note."""
        return self._note_path(note_id) / "content.md"

    def _attachments_path(self, note_id: str) -> Path:
        """Get the attachments directory for a note."""
        return self._note_path(note_id) / "attachments"

    def save_note(self, note: Note) -> Path:
        """
        Save a note to the file system.
        Returns the path to the note directory.
        """
        note_path = self._note_path(note.id)
        note_path.mkdir(parents=True, exist_ok=True)

        # Save content
        content_path = self._content_path(note.id)
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(f"# {note.title}\n\n{note.content}")

        # Save metadata
        metadata_path = self._metadata_path(note.id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(note.model_dump_json(indent=2))

        return note_path

    def load_note(self, note_id: str) -> Optional[Note]:
        """Load a note from the file system."""
        metadata_path = self._metadata_path(note_id)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        note = Note.model_validate(data)

        # Normalize datetimes to be timezone-aware for safe comparisons/sorting.
        def _aware(dt: datetime) -> datetime:
            if dt is None:
                return dt
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        note.created_at = _aware(note.created_at)
        note.modified_at = _aware(note.modified_at)
        for att in note.attachments:
            att.created_at = _aware(att.created_at)

        return note

    def delete_note(self, note_id: str) -> bool:
        """Delete a note and all its attachments."""
        note_path = self._note_path(note_id)

        if not note_path.exists():
            return False

        shutil.rmtree(note_path)
        return True

    def list_notes(self) -> List[str]:
        """List all note IDs."""
        if not self.notes_dir.exists():
            return []

        return [d.name for d in self.notes_dir.iterdir() if d.is_dir()]

    def add_attachment(
        self, note_id: str, file_path: Path, media_type: Optional[MediaType] = None
    ) -> Optional[Attachment]:
        """
        Add an attachment to a note.
        Copies the file into the note's attachments directory.
        """
        if not file_path.exists():
            return None

        note = self.load_note(note_id)
        if not note:
            return None

        # Create attachments directory
        attachments_dir = self._attachments_path(note_id)
        attachments_dir.mkdir(parents=True, exist_ok=True)

        # Determine media type
        if media_type is None:
            media_type = self._infer_media_type(file_path)

        # Copy file
        dest_path = attachments_dir / file_path.name
        shutil.copy2(file_path, dest_path)

        # Create attachment record
        attachment = Attachment(
            filename=file_path.name,
            media_type=media_type,
            size_bytes=dest_path.stat().st_size,
            created_at=datetime.now(timezone.utc),
        )

        # Update note
        note.add_attachment(attachment)
        self.save_note(note)

        return attachment

    def _infer_media_type(self, file_path: Path) -> MediaType:
        """Infer media type from file extension."""
        ext = file_path.suffix.lower()

        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}
        audio_exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        text_exts = {".txt", ".md", ".pdf", ".doc", ".docx"}

        if ext in image_exts:
            return MediaType.IMAGE
        elif ext in audio_exts:
            return MediaType.AUDIO
        elif ext in video_exts:
            return MediaType.VIDEO
        elif ext in text_exts:
            return MediaType.TEXT
        else:
            return MediaType.OTHER

    def get_attachment_path(self, note_id: str, filename: str) -> Optional[Path]:
        """Get the full path to an attachment file."""
        path = self._attachments_path(note_id) / filename
        return path if path.exists() else None

    def update_note_content(
        self, note_id: str, content: str, title: Optional[str] = None
    ) -> bool:
        """Update note content."""
        note = self.load_note(note_id)
        if not note:
            return False

        note.update_content(content, title)
        self.save_note(note)
        return True
