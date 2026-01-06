# Black Hole (BH)

BH is a simplified capture-first interface on top of the notes system.

## Commands

Capture:
- `bh add <text-or-path>`: create a note from raw text or a file
  - Text files are inlined; all files are attached to the note.
  - Audio files (`.m4a/.mp3/.wav/...`) are transcribed via Whisper by default.
  - Use `--no-transcribe` to ingest audio as an attachment only.
- `bh yank`: create a note from clipboard text (uses `pbpaste` on macOS).

Search:
- `bh search <query> [--mode semantic|keyword|hybrid] [--limit N]`

Traversal:
- `bh browse [--limit N] [--sort modified|created|title]`
- `bh show <note-id>`
- `bh open <note-id>`: opens the note in `$EDITOR`
- `bh link <note-id> [--min-weight W]`: refreshes and prints the top auto-links

Proactive:
- `bh proactive todo [--scope new|all]`
- `bh proactive brief [--scope new|all]`
- `bh digest [--scope new|all]`
- `bh retag [--scope new|all]`

Health:
- `bh lint`: reports missing `content.md`/metadata/attachments

Watching:
- `bh create-bh <dir> [--follow] [--interval seconds]`
  - `--follow` runs an infinite poll loop; `Ctrl+C` stops it.

## Configuration

Default base path is `~/.notes`.

Config file location:
- `~/.notes/.notes_config.json`

Example:
```json
{
  "embedding_provider": {"type": "openai", "model": "text-embedding-3-small"},
  "tagger": {"type": "openai", "model": "gpt-5-mini"},
  "transcriber": {"type": "whisper", "model": "whisper-1"},
  "llm": {"model": "gpt-5-mini"}
}
```

Environment:
- `OPENAI_API_KEY`: required for embeddings/transcription and for wbal-backed LLM calls.
- `OPENAI_API_BASE` (optional): defaults to `https://api.openai.com/v1`.

## State and indexes

BH state:
- `~/.notes/.notes_db/bh_state.json`
  - `last_processed`: timestamps per mode (`todo`, `brief`, `digest`, `retag`)
  - `watches`: per-directory last-seen timestamps for `create-bh`

To force a watched directory to rescan from scratch:
- remove its entry under `watches` (or delete `bh_state.json` entirely), then rerun `bh create-bh <dir>`

Semantic embeddings index:
- `~/.notes/.notes_db/embeddings.npy`
  - stores `meta` (`provider`, `dimension`)
  - auto-rebuilds if the provider or embedding dimension changes

## Architecture / extending BH

Business logic (reusable across future UIs):
- `src/notes/bh/app.py` (`BlackHoleApp`)

Argparse / printing layer:
- `src/notes/bh_cli.py`

LLM wrapper (wbal):
- `src/notes/bh_llm.py`

Whisper transcription wrapper:
- `src/notes/transcription.py`

When adding new BH features:
- implement the behavior in `BlackHoleApp` first (return values, no printing)
- add an argparse subcommand in `src/notes/bh_cli.py`
- add unit tests under `tests/`

