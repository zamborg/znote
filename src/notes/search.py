"""
Search functionality for notes: keyword and semantic search.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import sqlite3
import numpy as np
from dataclasses import dataclass

from .datamodel import Note
from .adapters import EmbeddingProvider
from .storage import NotesStorage


@dataclass
class SearchResult:
    """Represents a search result."""

    note_id: str
    score: float
    title: str
    snippet: str


class KeywordSearch:
    """Full-text search using SQLite FTS5."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the FTS5 table."""
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                note_id,
                title,
                content,
                tags
            )
        """)
        self.conn.commit()

    def index_note(self, note: Note):
        """Index a note for keyword search."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO notes_fts (note_id, title, content, tags)
            VALUES (?, ?, ?, ?)
        """,
            (note.id, note.title, note.content, " ".join(note.tags)),
        )
        self.conn.commit()

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for notes by keyword."""
        cursor = self.conn.execute(
            """
            SELECT note_id, title, snippet(notes_fts, 2, '<b>', '</b>', '...', 32) as snippet,
                   rank
            FROM notes_fts
            WHERE notes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """,
            (query, limit),
        )

        results = []
        for row in cursor:
            results.append(
                SearchResult(
                    note_id=row[0],
                    title=row[1],
                    snippet=row[2],
                    score=-row[3],  # FTS5 rank is negative
                )
            )

        return results

    def delete_note(self, note_id: str):
        """Remove a note from the index."""
        self.conn.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()


class SemanticSearch:
    """Semantic search using embeddings."""

    def __init__(
        self,
        embeddings_path: Path,
        index_path: Path,
        provider: EmbeddingProvider | None = None,
    ):
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.provider = provider
        self.embeddings_cache = {}  # note_id -> embedding vector
        self.note_ids = []
        self.meta: dict = {}
        self.dimension = getattr(provider, "dimension", 384) if provider else 384

        self._load_index()

    def _load_index(self):
        """Load existing embeddings and index."""
        if self.embeddings_path.exists():
            data = np.load(self.embeddings_path, allow_pickle=True).item()
            self.embeddings_cache = data.get("embeddings", {})
            self.note_ids = data.get("note_ids", [])
            self.meta = data.get("meta", {}) or {}

    def _save_index(self):
        """Save embeddings and index to disk."""
        np.save(
            self.embeddings_path,
            {
                "embeddings": self.embeddings_cache,
                "note_ids": self.note_ids,
                "meta": self.meta,
            },
        )

    def _expected_meta(self) -> dict:
        provider_name = getattr(self.provider, "name", None) if self.provider else None
        return {
            "provider": provider_name or "local",
            "dimension": int(self.dimension),
        }

    def _infer_stored_dimension(self) -> Optional[int]:
        if isinstance(self.meta, dict) and self.meta.get("dimension"):
            try:
                return int(self.meta["dimension"])
            except Exception:
                pass
        for vec in (self.embeddings_cache or {}).values():
            try:
                if vec is not None and getattr(vec, "shape", None):
                    return int(vec.shape[0])
            except Exception:
                continue
        return None

    def ensure_compatible(self, storage: NotesStorage) -> bool:
        """
        Ensure the on-disk index is compatible with the configured provider.
        Rebuilds the index when provider/dimension change to prevent shape mismatches.

        Returns True if a rebuild was performed.
        """
        expected = self._expected_meta()
        stored_dim = self._infer_stored_dimension()
        stored_provider = (self.meta or {}).get("provider")

        compatible = True
        if stored_dim is not None and int(stored_dim) != int(expected["dimension"]):
            compatible = False
        if stored_provider and stored_provider != expected["provider"]:
            compatible = False

        if compatible:
            # If metadata is missing or incomplete, backfill it.
            if self.meta != expected:
                self.meta = expected
                self._save_index()
            return False

        # Rebuild
        self.embeddings_cache = {}
        self.note_ids = []
        self.meta = expected
        for note_id in storage.list_notes():
            note = storage.load_note(note_id)
            if note:
                self._index_note_in_memory(note)
        self._save_index()
        return True

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for text.
        Using simple TF-IDF style embedding as placeholder.
        In production, you'd use sentence-transformers or OpenAI embeddings.
        """
        if self.provider:
            try:
                vecs = self.provider.embed_texts([text])
                if vecs:
                    vec = vecs[0]
                    try:
                        if getattr(vec, "shape", None):
                            self.dimension = int(vec.shape[0])
                    except Exception:
                        pass
                    return vec
            except Exception:
                # Fall back to local embedding on provider failure
                pass

        # Simple bag-of-words embedding (placeholder)
        words = text.lower().split()
        embedding = np.zeros(self.dimension)  # Standard embedding size

        for i, word in enumerate(words[: self.dimension]):
            embedding[i % self.dimension] += hash(word) % 100 / 100.0

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _index_note_in_memory(self, note: Note):
        text = f"{note.title}\n\n{note.content}\n\n{' '.join(note.tags)}"
        embedding = self.compute_embedding(text)
        self.embeddings_cache[note.id] = embedding
        if note.id not in self.note_ids:
            self.note_ids.append(note.id)

    def index_note(self, note: Note):
        """Index a note for semantic search."""
        self.meta = self._expected_meta()
        self._index_note_in_memory(note)
        self._save_index()

    def search(
        self, query: str, limit: int = 10, min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for notes by semantic similarity.
        Returns list of (note_id, similarity_score) tuples.
        """
        if not self.embeddings_cache:
            return []

        query_embedding = self.compute_embedding(query)

        # Compute cosine similarities
        similarities = []
        for note_id in self.note_ids:
            if note_id in self.embeddings_cache:
                embedding = self.embeddings_cache[note_id]
                if (
                    embedding is None
                    or not getattr(embedding, "shape", None)
                    or embedding.shape != query_embedding.shape
                ):
                    continue
                similarity = np.dot(query_embedding, embedding)
                if similarity >= min_score:
                    similarities.append((note_id, float(similarity)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:limit]

    def delete_note(self, note_id: str):
        """Remove a note from the semantic index."""
        if note_id in self.embeddings_cache:
            del self.embeddings_cache[note_id]
        if note_id in self.note_ids:
            self.note_ids.remove(note_id)
        self._save_index()

    def get_embedding(self, note_id: str) -> Optional[np.ndarray]:
        """Get the embedding for a note."""
        return self.embeddings_cache.get(note_id)


class NotesSearch:
    """Combined search interface."""

    def __init__(self, storage: NotesStorage, embedding_provider: EmbeddingProvider | None = None):
        self.storage = storage
        self.keyword_search = KeywordSearch(storage.db_dir / "search.db")
        self.semantic_search = SemanticSearch(
            storage.db_dir / "embeddings.npy", storage.index_dir, provider=embedding_provider
        )
        self.semantic_search.ensure_compatible(storage)

    def index_note(self, note: Note):
        """Index a note for both keyword and semantic search."""
        self.keyword_search.index_note(note)
        self.semantic_search.index_note(note)

    def keyword_search_notes(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search notes by keyword."""
        return self.keyword_search.search(query, limit)

    def semantic_search_notes(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search notes by semantic similarity."""
        results = self.semantic_search.search(query, limit)

        search_results = []
        for note_id, score in results:
            note = self.storage.load_note(note_id)
            if note:
                snippet = (
                    note.content[:200] + "..."
                    if len(note.content) > 200
                    else note.content
                )
                search_results.append(
                    SearchResult(
                        note_id=note_id, score=score, title=note.title, snippet=snippet
                    )
                )

        return search_results

    def hybrid_search(
        self, query: str, limit: int = 10, keyword_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        Hybrid search combining keyword and semantic search.
        keyword_weight: 0-1, how much to weight keyword vs semantic (1-keyword_weight)
        """
        keyword_results = self.keyword_search_notes(query, limit * 2)
        semantic_results = self.semantic_search_notes(query, limit * 2)

        # Combine scores
        combined = {}

        # Normalize keyword scores
        if keyword_results:
            max_keyword = max(r.score for r in keyword_results)
            for result in keyword_results:
                norm_score = result.score / max_keyword if max_keyword > 0 else 0
                combined[result.note_id] = {
                    "keyword": norm_score * keyword_weight,
                    "semantic": 0,
                    "result": result,
                }

        # Add semantic scores
        for result in semantic_results:
            if result.note_id in combined:
                combined[result.note_id]["semantic"] = result.score * (
                    1 - keyword_weight
                )
            else:
                combined[result.note_id] = {
                    "keyword": 0,
                    "semantic": result.score * (1 - keyword_weight),
                    "result": result,
                }

        # Calculate final scores
        final_results = []
        for note_id, scores in combined.items():
            total_score = scores["keyword"] + scores["semantic"]
            result = scores["result"]
            result.score = total_score
            final_results.append(result)

        # Sort and limit
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:limit]

    def delete_note(self, note_id: str):
        """Remove a note from all search indices."""
        self.keyword_search.delete_note(note_id)
        self.semantic_search.delete_note(note_id)

    def close(self):
        """Close all search resources."""
        self.keyword_search.close()
