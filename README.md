# Notes - Multimodal Research Notes System

A clean, efficient notes management system designed for research teams. Supports multimodal content (text, images, audio, video), semantic search, and automatic note linking.

## Features

1. **Multimodal Notes**: Each note is a folder that can contain text, images, audio, video, and other attachments
2. **Dual Search**: Keyword search (SQLite FTS5) + Semantic search (embeddings) + Hybrid mode
3. **Easy Updates**: Edit notes directly in your editor or through the CLI
4. **Automatic Linking**: Graph structure with edge weights based on semantic similarity and keyword co-occurrence
5. **Simple CLI**: Command-line interface for all operations

## Installation

```bash
# From the project directory (package name: znote)
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e .
# or
pip install -e .
```

Configure providers in `~/.notes_config.json` (example for OpenAI):
```json
{
  "embedding_provider": {"type": "openai", "model": "text-embedding-3-small"},
  "tagger": {"type": "openai", "model": "gpt-5-mini"}
}
```

## Quick Start

```bash
# Create a note
znote create "My First Note" -c "This is the content" -t research ml

# Search for notes
znote search "machine learning"

# List all notes
znote list

# View a specific note
znote show <note-id>

# Edit a note
znote edit <note-id>

# Add an attachment
znote attach <note-id> /path/to/file.png

# View linked notes
znote links <note-id>

# Auto-tag on create (uses configured tagger)
znote create "Planning" -c "Discussed LLM benchmarks" --auto-tags

# Reindex notes modified after a date and rebuild graph
znote reintegrate --after 2024-01-01T00:00:00
```

## Usage

### Creating Notes

```bash
# Simple note
znote create "Meeting Notes"

# With content and tags
znote create "Research Ideas" -c "Initial thoughts..." -t research ideas

# With attachments
znote create "Experiment Results" -a plot.png data.csv recording.mp3
```

### Searching Notes

```bash
# Hybrid search (default - combines keyword + semantic)
znote search "neural networks"

# Keyword-only search (faster)
znote search "neural networks" --mode keyword

# Semantic search (finds conceptually similar notes)
znote search "deep learning architectures" --mode semantic

# Limit results
znote search "transformer" --limit 5
```

### Managing Notes

```bash
# List all notes (sorted by modified date)
znote list

# Sort by creation date
znote list --sort created

# Sort by title
znote list --sort title

# View a note
znote show abc123

# Edit a note (opens in $EDITOR or vim)
znote edit abc123

# Delete a note
znote delete abc123
```

### Attachments

```bash
# Add an attachment
znote attach abc123 /path/to/image.png

# Attachments are stored in the note's attachments/ folder
# You can also manually add files to: ~/.notes/notes/<note-id>/attachments/
```

### Automatic Linking

Notes are automatically linked based on:
- Semantic similarity (via embeddings)
- Keyword co-occurrence
- Tag overlap

```bash
# View links for a note
znote links abc123

# Rebuild the entire graph (run after bulk imports)
znote rebuild-graph
```

## Architecture

### Directory Structure

```
~/.notes/                    # Default base directory
├── notes/                   # All notes
│   └── <note-id>/          # Each note is a directory
│       ├── content.md      # Main content
│       ├── metadata.json   # Note metadata
│       └── attachments/    # Media files
└── .notes_db/              # Indices and metadata
    ├── search.db           # Keyword search index (SQLite FTS5)
    ├── embeddings.npy      # Semantic embeddings
    ├── graph.json          # Note link graph
    └── index/              # Additional indices
```

### Data Model

**Note Structure:**
- `id`: Unique identifier (UUID)
- `title`: Note title
- `content`: Main text content (markdown)
- `tags`: List of tags
- `created_at`: Creation timestamp
- `modified_at`: Last modification timestamp
- `attachments`: List of attached files
- `linked_notes`: IDs of manually linked notes

**Graph Structure:**
- Directed graph with weighted edges
- Edge weight = 0.5 × semantic_similarity + 0.3 × keyword_overlap + 0.2 × tag_overlap
- Default threshold: 0.3 (configurable)
- Max links per note: 10 (configurable)

### Search Strategy

**Keyword Search:**
- SQLite FTS5 full-text search
- Searches title, content, and tags
- Returns ranked results with snippets

**Semantic Search:**
- Embeddings-based similarity search
- Cosine similarity between query and note embeddings
- Currently uses simple TF-IDF embeddings (placeholder)
- Can be upgraded to sentence-transformers or OpenAI embeddings

**Hybrid Search:**
- Combines keyword and semantic scores
- Default weight: 50% keyword, 50% semantic
- Configurable via `keyword_weight` parameter

## Customization

### Change Storage Location

```bash
znote --base-path /path/to/znote create "My Note"
```

### Set Default Editor

```bash
export EDITOR=nano  # or code, emacs, etc.
znote edit <note-id>
```

### Adjust Link Threshold

Edit `cli.py` and modify the `threshold` parameter in `update_note_links()`:

```python
self.graph.update_note_links(note.id, threshold=0.4)  # Higher = fewer links
```

### Reintegrate / Migrations

Use `znote reintegrate` to reindex notes filtered by modified date, with an optional graph rebuild:

```bash
znote reintegrate --after 2024-01-01T00:00:00 --before 2024-06-01T00:00:00
znote reintegrate --after 2024-01-01T00:00:00 --dry-run  # preview only
```

### Development

- Layout: `src/notes/` package; migrations live in `notes.migrations`.
- Install for hacking: `uv pip install -e .` (or `pip install -e .`).
- Config: `.notes_config.json` controls embedding/tagging providers; requires `OPENAI_API_KEY` for OpenAI adapters.
- Tests: `pytest` (uses fake providers; no network needed).

## Future Enhancements

- [ ] Upgrade to proper embeddings (sentence-transformers)
- [ ] LLM integration for note summarization
- [ ] Web interface
- [ ] Export to various formats (PDF, HTML)
- [ ] Note versioning
- [ ] Collaborative features
- [ ] Mobile app

## File Organization

```
src/notes/datamodel.py          - Core data structures (Note, Attachment, NoteLink)
src/notes/storage.py            - File system operations and persistence
src/notes/search.py             - Keyword, semantic, and hybrid search
src/notes/graph.py              - Automatic linking and graph management
src/notes/cli.py                - Command-line interface
src/notes/adapters.py           - Interfaces for embeddings, tagging, vector indexes
src/notes/providers_openai.py   - OpenAI adapters for embeddings/tagging
src/notes/migrations/reintegrate.py - Reindexing helper for migrations
```

## Tips

1. **Tags are powerful**: Use consistent tags for better organization and linking
2. **Edit directly**: You can edit `~/.notes/notes/<note-id>/content.md` directly
3. **Bulk import**: Drop folders into `~/.notes/notes/` then run `znote rebuild-graph`
4. **Semantic search**: Works better with longer, descriptive content
5. **Graph rebuilds**: Run after importing many notes or changing link thresholds

## License

MIT
