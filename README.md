# BH (Black Hole)

BH is the primary CLI for this repo: a capture-first, markdown-first notes + todos system.

Core ideas:
- **Streams**: hierarchical categories like `inbox`, `work/bh`, `clients/acme`.
- **Items**: notes and todos are stored as folders with `content.md`, `metadata.json`, and `attachments/`.
- **Multimodal ingest**: text, files, directories; voice notes can be transcribed.
- **Access**: browse/search by stream, generate briefs/digests, and build completionist index docs.
- **Provenance + cost tracking**: attachments are hashed and referenced; external calls are logged.

## Install / Run (uv)

```bash
uv sync
uv run bh status
```

If you prefer the venv:

```bash
source .venv/bin/activate
bh status
```

## Configuration

BH reads config from `<base-path>/.notes_config.json` (default `<base-path>` is `~/.notes`).

Example:
```json
{
  "embedding_provider": {"type": "openai", "model": "text-embedding-3-small"},
  "tagger": {"type": "openai", "model": "gpt-4o-mini"},
  "transcriber": {"type": "whisper", "model": "whisper-1"},
  "llm": {"model": "gpt-5-mini"}
}
```

Environment:
- `OPENAI_API_KEY` (required for OpenAI embeddings/tagging/transcription)
- `OPENAI_API_BASE` (optional; defaults to `https://api.openai.com/v1`)

## CLI Cheatsheet

All commands support `--base-path` (defaults to `~/.notes`).

Capture:
```bash
uv run bh add "stray idea about energy markets" --stream inbox
uv run bh new --stream work/bh                 # open $EDITOR to write markdown

uv run bh add path/to/voice.m4a --stream work/bh
uv run bh add path/to/voice.m4a --no-transcribe

uv run bh add path/to/folder --stream inbox    # ingests one item per file
uv run bh create-bh path/to/folder --follow --interval 30 --stream inbox
```

Access:
```bash
uv run bh search "fusion breakthroughs" --mode semantic --stream work
uv run bh browse --sort modified --stream work/bh
uv run bh show <item-id>
uv run bh open <item-id>
uv run bh link <item-id>
```

Todos:
```bash
uv run bh todo add "Ship v1" --due 2026-01-14 --stream work/bh
uv run bh todo list --status open --stream work
uv run bh todo done <todo-id>
uv run bh todo archive <todo-id>
```

Proactive (writes notes under the hood):
```bash
uv run bh proactive todo --scope all --stream work/bh
uv run bh proactive brief --scope new --stream work/bh
uv run bh digest --scope new --stream work/bh
```

Completionist docs:
```bash
uv run bh completion stream work/bh
```

Costs:
```bash
uv run bh cost daily --days 7
uv run bh cost events --day 2026-01-14 --limit 50
```

## Storage layout

Default base path is `~/.notes` (override via `--base-path`):

```
<base-path>/
  .notes_config.json
  notes/
    <item-id>/
      content.md
      metadata.json
      attachments/
  .notes_db/
    search.db       # SQLite FTS5
    embeddings.npy  # semantic index
    graph.json      # auto-link graph
    bh_state.json   # watched folders + last runs
    bh.db           # provenance + cost ledger (SQLite)
    tmp/            # editor scratch files
```

More details: `docs/black-hole.md`.

## Development

```bash
uv run pytest -q
```

Notes:
- `bh` is the supported UX. `znote` still exists but is considered legacy.
- Some cost entries may be `unknown` if usage metadata isn’t available; pricing lookup lives in `src/notes/costs.py`.
