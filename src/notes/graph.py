"""
Graph management for automatic note linking.
Computes edge weights based on semantic similarity and keyword co-occurrence.
"""

from typing import Dict, List, Tuple, Set
import json
import numpy as np
from collections import defaultdict

from .datamodel import Note
from .storage import NotesStorage
from .search import SemanticSearch


class NotesGraph:
    """Manages the graph structure of linked notes."""

    def __init__(self, storage: NotesStorage, semantic_search: SemanticSearch):
        self.storage = storage
        self.semantic_search = semantic_search
        self.graph_path = storage.db_dir / "graph.json"

        # adjacency list: note_id -> [(target_id, weight), ...]
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        self._load_graph()

    def _load_graph(self):
        """Load graph from disk."""
        if self.graph_path.exists():
            with open(self.graph_path, "r") as f:
                data = json.load(f)
                self.graph = defaultdict(
                    list,
                    {
                        k: [(item["target"], item["weight"]) for item in v]
                        for k, v in data.items()
                    },
                )

    def _save_graph(self):
        """Save graph to disk."""
        data = {
            note_id: [{"target": target, "weight": weight} for target, weight in edges]
            for note_id, edges in self.graph.items()
        }

        with open(self.graph_path, "w") as f:
            json.dump(data, f, indent=2)

    def compute_similarity(self, note1: Note, note2: Note) -> float:
        """
        Compute similarity between two notes.
        Combines semantic similarity with keyword overlap.
        """
        # Semantic similarity
        emb1 = self.semantic_search.get_embedding(note1.id)
        emb2 = self.semantic_search.get_embedding(note2.id)

        semantic_sim = 0.0
        if emb1 is not None and emb2 is not None:
            semantic_sim = float(np.dot(emb1, emb2))

        # Keyword overlap (Jaccard similarity)
        words1 = set(note1.content.lower().split()) | set(note1.title.lower().split())
        words2 = set(note2.content.lower().split()) | set(note2.title.lower().split())

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "be",
            "been",
            "it",
            "this",
            "that",
        }
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if words1 and words2:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            keyword_sim = intersection / union if union > 0 else 0.0
        else:
            keyword_sim = 0.0

        # Tag overlap
        tags1 = set(note1.tags)
        tags2 = set(note2.tags)
        tag_sim = 0.0

        if tags1 and tags2:
            tag_intersection = len(tags1 & tags2)
            tag_union = len(tags1 | tags2)
            tag_sim = tag_intersection / tag_union if tag_union > 0 else 0.0

        # Weighted combination
        similarity = 0.5 * semantic_sim + 0.3 * keyword_sim + 0.2 * tag_sim

        return similarity

    def update_note_links(
        self, note_id: str, threshold: float = 0.3, max_links: int = 10
    ):
        """
        Update links for a specific note.
        Links to notes with similarity above threshold.
        """
        note = self.storage.load_note(note_id)
        if not note:
            return

        all_note_ids = self.storage.list_notes()
        similarities = []

        for other_id in all_note_ids:
            if other_id == note_id:
                continue

            other_note = self.storage.load_note(other_id)
            if not other_note:
                continue

            similarity = self.compute_similarity(note, other_note)

            if similarity >= threshold:
                similarities.append((other_id, similarity))

        # Sort by similarity and keep top max_links
        similarities.sort(key=lambda x: x[1], reverse=True)
        self.graph[note_id] = similarities[:max_links]

        self._save_graph()

    def rebuild_graph(self, threshold: float = 0.3, max_links: int = 10):
        """
        Rebuild the entire graph from scratch.
        Call this after adding/updating multiple notes.
        """
        all_note_ids = self.storage.list_notes()

        for note_id in all_note_ids:
            self.update_note_links(note_id, threshold, max_links)

    def get_linked_notes(
        self, note_id: str, min_weight: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Get all notes linked to the given note.
        Returns list of (note_id, weight) tuples.
        """
        links = self.graph.get(note_id, [])
        return [(nid, weight) for nid, weight in links if weight >= min_weight]

    def get_backlinks(
        self, note_id: str, min_weight: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Get all notes that link to the given note (backlinks).
        """
        backlinks = []

        for source_id, edges in self.graph.items():
            for target_id, weight in edges:
                if target_id == note_id and weight >= min_weight:
                    backlinks.append((source_id, weight))

        return backlinks

    def find_related_notes(self, note_id: str, depth: int = 2) -> Set[str]:
        """
        Find notes related to the given note up to depth hops away.
        """
        visited = set()
        current_level = {note_id}

        for _ in range(depth):
            next_level = set()

            for nid in current_level:
                if nid in visited:
                    continue

                visited.add(nid)

                # Add linked notes
                for linked_id, _ in self.get_linked_notes(nid):
                    if linked_id not in visited:
                        next_level.add(linked_id)

            current_level = next_level

            if not current_level:
                break

        visited.discard(note_id)  # Remove the original note
        return visited

    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph."""
        total_notes = len(self.graph)
        total_edges = sum(len(edges) for edges in self.graph.values())

        if total_notes > 0:
            avg_degree = total_edges / total_notes
        else:
            avg_degree = 0

        return {
            "total_notes": total_notes,
            "total_edges": total_edges,
            "avg_degree": avg_degree,
        }

    def delete_note(self, note_id: str):
        """Remove a note from the graph."""
        # Remove the node
        if note_id in self.graph:
            del self.graph[note_id]

        # Remove edges pointing to this node
        for edges in self.graph.values():
            edges[:] = [
                (target, weight) for target, weight in edges if target != note_id
            ]
            edges[:] = [(target, weight) for target, weight in edges if target != note_id]

        self._save_graph()
