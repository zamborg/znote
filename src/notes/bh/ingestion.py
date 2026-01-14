from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    from .app import BlackHoleApp


TEXT_EXTS = {".txt", ".md", ".rst"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


@dataclass(frozen=True)
class AddOptions:
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    auto_tags: bool = False
    transcribe_audio: bool = True
    stream: Optional[str] = None


class IngestAdapter(Protocol):
    def can_handle(self, raw: str) -> bool: ...

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> str | List[str]: ...


class DirectoryAdapter:
    def can_handle(self, raw: str) -> bool:
        path = Path(raw)
        return path.exists() and path.is_dir()

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> List[str]:
        return app.ingest_directory_all(Path(raw), stream=opts.stream)


class TextFileAdapter:
    def can_handle(self, raw: str) -> bool:
        path = Path(raw)
        return path.exists() and path.is_file() and path.suffix.lower() in TEXT_EXTS

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> str:
        path = Path(raw)
        body = path.read_text(encoding="utf-8")
        title = opts.title or path.stem
        return app._create_note_quiet(
            title,
            body,
            tags=opts.tags,
            attachments=[str(path)],
            auto_tags=opts.auto_tags,
            stream=opts.stream,
        )


class AudioFileAdapter:
    def can_handle(self, raw: str) -> bool:
        path = Path(raw)
        return path.exists() and path.is_file() and path.suffix.lower() in AUDIO_EXTS

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> str:
        path = Path(raw)
        title = opts.title or path.stem
        body = app._ingest_audio_body(path, opts.transcribe_audio)
        return app._create_note_quiet(
            title,
            body,
            tags=opts.tags,
            attachments=[str(path)],
            auto_tags=opts.auto_tags,
            stream=opts.stream,
        )


class GenericFileAdapter:
    def can_handle(self, raw: str) -> bool:
        path = Path(raw)
        return path.exists() and path.is_file()

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> str:
        path = Path(raw)
        title = opts.title or path.stem
        body = f"Attachment ingested via bh add: {path.name}\n\n(No transcription available.)"
        return app._create_note_quiet(
            title,
            body,
            tags=opts.tags,
            attachments=[str(path)],
            auto_tags=opts.auto_tags,
            stream=opts.stream,
        )


class RawTextAdapter:
    def can_handle(self, raw: str) -> bool:
        path = Path(raw)
        return not (path.exists() and (path.is_file() or path.is_dir()))

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> str:
        title = opts.title or app._title_from_text(raw)
        return app._create_note_quiet(
            title,
            raw,
            tags=opts.tags,
            attachments=None,
            auto_tags=opts.auto_tags,
            stream=opts.stream,
        )


class IngestionPipeline:
    def __init__(self, adapters: Optional[List[IngestAdapter]] = None):
        self.adapters: List[IngestAdapter] = adapters or [
            DirectoryAdapter(),
            TextFileAdapter(),
            AudioFileAdapter(),
            GenericFileAdapter(),
            RawTextAdapter(),
        ]

    def ingest(self, app: "BlackHoleApp", raw: str, opts: AddOptions) -> str | List[str]:
        for adapter in self.adapters:
            if adapter.can_handle(raw):
                return adapter.ingest(app, raw, opts)
        raise RuntimeError(f"No ingestion adapter matched input: {raw!r}")

