"""
Core datamodel for the multimodal notes system.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class Attachment(BaseModel):
    """Represents a single attachment in a note."""

    filename: str
    media_type: MediaType
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Note(BaseModel):
    """Core note datamodel."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str = ""
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attachments: List[Attachment] = Field(default_factory=list)
    linked_notes: List[str] = Field(default_factory=list)

    def update_content(self, content: str, title: Optional[str] = None):
        """Update note content and refresh modified timestamp."""
        self.content = content
        if title:
            self.title = title
        self.modified_at = datetime.now(timezone.utc)

    def add_attachment(self, attachment: Attachment):
        """Add an attachment to the note."""
        self.attachments.append(attachment)
        self.modified_at = datetime.now(timezone.utc)

    def add_tag(self, tag: str):
        """Add a tag to the note."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.modified_at = datetime.now(timezone.utc)

    def link_to(self, note_id: str):
        """Create a link to another note."""
        if note_id not in self.linked_notes:
            self.linked_notes.append(note_id)
            self.modified_at = datetime.now(timezone.utc)


class NoteLink(BaseModel):
    """Represents a link between two notes with weight."""

    source_id: str
    target_id: str
    weight: float = Field(ge=0, le=1)  # Similarity score 0-1
    link_type: str = "auto"  # "auto" for automatic, "manual" for user-created
