# Black Hole (BH)

BH is a capture-first CLI on top of a local-first notes system.

It is:
- **Markdown-first**: items live on disk as `content.md` + `metadata.json`.
- **Streams-first**: everything belongs to a stream (`work/bh`, `inbox`, ...).
- **Multimodal**: files become attachments; audio can be transcribed.
- **Auditable**: attachments are hashed and referenced in a provenance table; external calls are logged for cost tracking.

## Concepts

### Streams

A **stream** is a hierarchical category, represented as a path-like string.

Normalization rules:
- lowercased
- whitespace → `-`
- separators → `/`

Examples:
- `Work/BH` → `work/bh`
- `clients acme / finance` → `clients-acme/finance`

Filtering behavior:
- `--stream work` matches `work`, `work/bh`, `work/anything/else`

### Items (notes + todos)

Everything is an **item**; items are either:
- `kind="note"`
- `kind="todo"`

Todos add lifecycle fields:
- `due_at` (optional)
- `completed_at` (set by `bh todo done`)
- `archived_at` (set by `bh todo archive`)

### Sources (ground truth) vs derived content

The system treats attachments as **ground truth**.

When you `bh add /path/to/file`:
- the file is copied into the item’s `attachments/`
- its SHA-256 is computed and stored
- a provenance record is written to `bh.db` linking the item ⇄ the source

Derived content (transcripts, tags, briefs/digests) is reproducible and should always be understood as “generated from sources”.

### “Access” (ways to get to things)

BH supports multiple ways to access items:
- **Search**: `bh search ...` (semantic/keyword/hybrid), optionally scoped to a stream subtree.
- **Browse**: `bh browse ... --stream ...` to leaf through recent items.
- **Proactive docs**: `bh proactive brief`, `bh digest` produce AI-authored documents.
- **Completionist docs**: `bh completion stream <stream>` builds a deterministic index note for a stream.

## Running (uv)

All examples assume `uv`:

```bash
uv sync
uv run bh status
```

## Configuration

Config file: `<base-path>/.notes_config.json` (default `<base-path>` is `~/.notes`).

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

## Commands

### Capture

Add raw text:
```bash
uv run bh add "quick idea about rockets" --stream work/bh
```

Open `$EDITOR` to create a markdown note:
```bash
uv run bh new --stream work/bh
```

Ingest a file (copied into `attachments/`):
```bash
uv run bh add path/to/file.pdf --stream inbox
```

Voice notes (audio):
```bash
uv run bh add path/to/voice.m4a --stream work/bh
uv run bh add path/to/voice.m4a --no-transcribe
```

Directory ingest:
- `bh add <dir>` ingests **every file** (one item per file) each time you run it.
- `bh create-bh <dir>` ingests **only new files** based on persistent watch state.

```bash
uv run bh add /path/to/inbox --stream inbox
uv run bh create-bh /path/to/inbox --stream inbox
uv run bh create-bh /path/to/inbox --follow --interval 30 --stream inbox
```

Clipboard capture:
```bash
uv run bh yank --stream inbox
```

### Search

```bash
uv run bh search "fusion" --mode semantic --limit 10
uv run bh search "fusion" --mode keyword --stream work
uv run bh search "fusion" --mode hybrid --stream work/bh
```

### Browse / show / open

```bash
uv run bh browse --sort modified --limit 20 --stream work
uv run bh show <item-id>
uv run bh open <item-id>
```

### Links

BH maintains an auto-link graph (semantic + keyword overlap + tag overlap):
```bash
uv run bh link <item-id> --min-weight 0.3
```

### Todos

Create:
```bash
uv run bh todo add "Ship v1" --due 2026-01-14 --stream work/bh
uv run bh todo new --due 2026-01-14 --stream work/bh   # opens $EDITOR
```

List / mutate:
```bash
uv run bh todo list --status open --stream work
uv run bh todo done <todo-id>
uv run bh todo archive <todo-id>
```

### Proactive documents

All proactive routines write (or update) a note under the hood. When you pass `--stream`, BH writes a stream-specific doc note id like:
- `bh-todo--work__bh`
- `bh-brief--work__bh`
- `bh-digest--work__bh`

TODO dashboard (deterministic; no LLM):
```bash
uv run bh proactive todo --scope all --stream work/bh
uv run bh proactive todo --scope new --stream work/bh
```

Notes:
- `--scope all` includes all open todos in the stream.
- `--scope new` includes todos modified since the last run, plus todos due within the next 7 days.

Briefings (LLM-authored):
```bash
uv run bh proactive brief --scope new --stream work/bh
```

Digest (LLM-authored):
```bash
uv run bh digest --scope new --stream work/bh
```

### Tags

Tags are normalized (lowercase, slug-like) and can be regenerated via LLM:
```bash
uv run bh retag --scope all --stream work
```

### Completionist docs

Build a deterministic index note for a stream subtree:
```bash
uv run bh completion stream work/bh
```

### Health

```bash
uv run bh status
uv run bh lint
```

## Provenance + cost tracking

BH stores provenance and cost data in `<base-path>/.notes_db/bh.db`.

### Provenance tables

- `sources`: unique ground-truth blobs (keyed by SHA-256 when available)
- `source_refs`: links an item id ⇄ a source id (role, original path, stored path)

Each attachment in an item’s `metadata.json` stores:
- `sha256`
- `original_path` (at ingest time)
- `source_id` (back-reference into `sources`)

### Cost ledger

Each external call is logged as a row in `cost_events`:
- provider (`openai` / `wbal` / ...)
- operation (`embeddings` / `chat_completions` / `transcription` / `llm`)
- model, request id, token usage (if available), duration, and estimated USD cost

CLI:
```bash
uv run bh cost daily --days 7
uv run bh cost events --day 2026-01-14 --limit 50
```

Notes:
- Costs are **best-effort**. Some providers don’t expose enough metadata; those rows show `unknown`.
- Transcription cost estimation uses audio duration. If `ffprobe` is available, BH records duration; otherwise duration may be missing.
- Pricing lookup tables live in `src/notes/costs.py`.

## Storage layout

Default base path is `~/.notes`:

```
<base-path>/
  .notes_config.json
  notes/
    <item-id>/
      content.md
      metadata.json
      attachments/
  .notes_db/
    bh.db
    bh_state.json
    search.db
    embeddings.npy
    graph.json
    tmp/
```

## Extending ingestion

Ingestion is adapter-based (see `src/notes/bh/ingestion.py`).

To add a new ingest behavior (e.g. PDFs → OCR, images → captioning):
1. Implement an adapter with `can_handle()` + `ingest()`.
2. Insert it near the top of `IngestionPipeline.adapters`.
3. Write tests under `tests/`.

## Development

Run tests:
```bash
uv run pytest -q
```
