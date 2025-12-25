#!/usr/bin/env python3
"""
CLI interface for the notes system.
"""

import argparse
from pathlib import Path
from typing import Optional
import os
import subprocess

from .datamodel import Note
from .storage import NotesStorage
from .search import NotesSearch
from .graph import NotesGraph
from .config import load_config
from .providers_openai import OpenAIEmbeddingProvider, OpenAITagger
from .migrations.reintegrate import reintegrate


class NotesCLI:
    """Command-line interface for notes management."""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path.home() / ".notes"

        self.config = load_config(base_path)
        self.embedding_provider = self._build_embedding_provider()
        self.tagger = self._build_tagger()

        self.storage = NotesStorage(base_path)
        self.search = NotesSearch(self.storage, embedding_provider=self.embedding_provider)
        self.graph = NotesGraph(self.storage, self.search.semantic_search)

    def _build_embedding_provider(self):
        provider_cfg = self.config.get("embedding_provider", {}) or {}
        provider_type = provider_cfg.get("type")
        if provider_type == "openai":
            return OpenAIEmbeddingProvider(model=provider_cfg.get("model", "text-embedding-3-small"))
        return None

    def _build_tagger(self):
        tagger_cfg = self.config.get("tagger", {}) or {}
        tagger_type = tagger_cfg.get("type")
        if tagger_type == "openai":
            return OpenAITagger(model=tagger_cfg.get("model", "gpt-4o-mini"))
        return None

    def create_note(
        self,
        title: str,
        content: str = "",
        tags: list = None,
        attachments: list = None,
        auto_tags: bool = False,
    ):
        """Create a new note."""
        # Merge user tags with auto-tag suggestions if enabled
        final_tags = tags or []
        if auto_tags and self.tagger:
            try:
                suggested = self.tagger.suggest_tags(title, content)
                final_tags = self._merge_tags(final_tags, suggested)
            except Exception as exc:
                print(f"Warning: auto-tagging failed ({exc}); proceeding without suggestions.")
        elif auto_tags and not self.tagger:
            print("Warning: auto-tagging requested but no tagger is configured.")

        note = Note(
            title=title,
            content=content,
            tags=final_tags,
        )

        # Save note
        note_path = self.storage.save_note(note)

        # Add attachments if provided
        if attachments:
            for attachment_path in attachments:
                file_path = Path(attachment_path)
                if file_path.exists():
                    self.storage.add_attachment(note.id, file_path)
                else:
                    print(f"Warning: Attachment not found: {attachment_path}")

        # Index for search
        self.search.index_note(note)

        # Update graph
        self.graph.update_note_links(note.id)

        print(f"Created note: {note.id}")
        print(f"Title: {note.title}")
        print(f"Path: {note_path}")

        return note.id

    @staticmethod
    def _merge_tags(existing, suggested):
        """Merge and deduplicate while preserving order."""
        seen = set()
        merged = []
        for tag in (existing or []) + (suggested or []):
            if tag and tag not in seen:
                merged.append(tag)
                seen.add(tag)
        return merged

    def get_note(self, note_id: str):
        """Display a note."""
        note = self.storage.load_note(note_id)

        if not note:
            print(f"Note not found: {note_id}")
            return

        print(f"\nID: {note.id}")
        print(f"Title: {note.title}")
        print(f"Created: {note.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modified: {note.modified_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if note.tags:
            print(f"Tags: {', '.join(note.tags)}")

        print(f"\nContent:\n{'-' * 80}")
        print(note.content)
        print("-" * 80)

        if note.attachments:
            print("\nAttachments:")
            for att in note.attachments:
                print(
                    f"  - {att.filename} ({att.media_type.value}, {att.size_bytes} bytes)"
                )

        # Show linked notes
        links = self.graph.get_linked_notes(note.id, min_weight=0.3)
        if links:
            print("\nLinked Notes:")
            for linked_id, weight in links[:5]:
                linked_note = self.storage.load_note(linked_id)
                if linked_note:
                    print(f"  - {linked_note.title} (similarity: {weight:.2f})")

    def edit_note(self, note_id: str):
        """Open note in editor."""
        note = self.storage.load_note(note_id)

        if not note:
            print(f"Note not found: {note_id}")
            return

        content_path = self.storage._content_path(note_id)

        # Open in editor
        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, str(content_path)])

        # Reload content
        with open(content_path, "r") as f:
            lines = f.readlines()

            # Extract title from first line if it's a header
            new_title = note.title
            new_content = "".join(lines)

            if lines and lines[0].startswith("# "):
                new_title = lines[0][2:].strip()
                new_content = "".join(lines[2:]) if len(lines) > 2 else ""

        # Update note
        self.storage.update_note_content(note_id, new_content, new_title)

        # Reindex
        note = self.storage.load_note(note_id)
        self.search.index_note(note)
        self.graph.update_note_links(note_id)

        print(f"Updated note: {note_id}")

    def search_notes(self, query: str, mode: str = "hybrid", limit: int = 10):
        """Search for notes."""
        if mode == "keyword":
            results = self.search.keyword_search_notes(query, limit)
        elif mode == "semantic":
            results = self.search.semantic_search_notes(query, limit)
        else:  # hybrid
            results = self.search.hybrid_search(query, limit)

        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            note = self.storage.load_note(result.note_id)
            if note:
                print(f"{i}. {note.title}")
                print(f"   ID: {result.note_id}")
                print(f"   Score: {result.score:.3f}")
                print(f"   {result.snippet}")
                print()

    def list_notes(self, sort_by: str = "modified"):
        """List all notes."""
        note_ids = self.storage.list_notes()

        if not note_ids:
            print("No notes found.")
            return

        notes = []
        for note_id in note_ids:
            note = self.storage.load_note(note_id)
            if note:
                notes.append(note)

        # Sort
        if sort_by == "created":
            notes.sort(key=lambda n: n.created_at, reverse=True)
        elif sort_by == "title":
            notes.sort(key=lambda n: n.title.lower())
        else:  # modified
            notes.sort(key=lambda n: n.modified_at, reverse=True)

        print(f"\nTotal notes: {len(notes)}\n")

        for note in notes:
            print(f"{note.title}")
            print(f"  ID: {note.id}")
            print(f"  Modified: {note.modified_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if note.tags:
                print(f"  Tags: {', '.join(note.tags)}")
            print()

    def delete_note(self, note_id: str):
        """Delete a note."""
        note = self.storage.load_note(note_id)

        if not note:
            print(f"Note not found: {note_id}")
            return

        print(f"Delete note: {note.title} ({note_id})?")
        confirm = input("Type 'yes' to confirm: ")

        if confirm.lower() != "yes":
            print("Cancelled.")
            return

        self.storage.delete_note(note_id)
        self.search.delete_note(note_id)
        self.graph.delete_note(note_id)

        print(f"Deleted note: {note_id}")

    def add_attachment(self, note_id: str, file_path: str):
        """Add an attachment to a note."""
        attachment = self.storage.add_attachment(note_id, Path(file_path))

        if attachment:
            print(f"Added attachment: {attachment.filename}")

            # Reindex note
            note = self.storage.load_note(note_id)
            self.search.index_note(note)
        else:
            print("Failed to add attachment.")

    def show_links(self, note_id: str):
        """Show linked notes for a note."""
        note = self.storage.load_note(note_id)

        if not note:
            print(f"Note not found: {note_id}")
            return

        print(f"\nLinks for: {note.title}\n")

        # Forward links
        links = self.graph.get_linked_notes(note_id)
        if links:
            print("Linked to:")
            for linked_id, weight in links:
                linked_note = self.storage.load_note(linked_id)
                if linked_note:
                    print(f"  - {linked_note.title} (weight: {weight:.3f})")

        # Backlinks
        backlinks = self.graph.get_backlinks(note_id)
        if backlinks:
            print("\nLinked from:")
            for source_id, weight in backlinks:
                source_note = self.storage.load_note(source_id)
                if source_note:
                    print(f"  - {source_note.title} (weight: {weight:.3f})")

    def rebuild_graph(self):
        """Rebuild the entire graph."""
        print("Rebuilding graph...")
        self.graph.rebuild_graph()
        stats = self.graph.get_graph_stats()
        print("Graph rebuilt:")
        print(f"  Notes: {stats['total_notes']}")
        print(f"  Edges: {stats['total_edges']}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Notes management CLI")
    parser.add_argument("--base-path", type=Path, help="Base path for notes storage")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create
    create_parser = subparsers.add_parser(
        "create", aliases=["new"], help="Create a new note"
    )
    create_parser.add_argument("title", help="Note title")
    create_parser.add_argument("-c", "--content", default="", help="Note content")
    create_parser.add_argument("-t", "--tags", nargs="+", help="Tags")
    create_parser.add_argument(
        "-a", "--attachments", nargs="+", help="Attachment file paths"
    )
    create_parser.add_argument(
        "--auto-tags",
        action="store_true",
        help="Automatically suggest tags using configured tagger",
    )

    # Get
    get_parser = subparsers.add_parser("get", aliases=["show"], help="Display a note")
    get_parser.add_argument("note_id", help="Note ID")

    # Edit
    edit_parser = subparsers.add_parser("edit", help="Edit a note")
    edit_parser.add_argument("note_id", help="Note ID")

    # Search
    search_parser = subparsers.add_parser(
        "search", aliases=["find"], help="Search notes"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-m",
        "--mode",
        choices=["keyword", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode",
    )
    search_parser.add_argument(
        "-l", "--limit", type=int, default=10, help="Max results"
    )

    # List
    list_parser = subparsers.add_parser("list", aliases=["ls"], help="List all notes")
    list_parser.add_argument(
        "-s",
        "--sort",
        choices=["modified", "created", "title"],
        default="modified",
        help="Sort by",
    )

    # Delete
    delete_parser = subparsers.add_parser(
        "delete", aliases=["rm"], help="Delete a note"
    )
    delete_parser.add_argument("note_id", help="Note ID")

    # Attach
    attach_parser = subparsers.add_parser("attach", help="Add attachment to note")
    attach_parser.add_argument("note_id", help="Note ID")
    attach_parser.add_argument("file_path", help="File path")

    # Links
    links_parser = subparsers.add_parser("links", help="Show linked notes")
    links_parser.add_argument("note_id", help="Note ID")

    # Rebuild
    rebuild_parser = subparsers.add_parser(
        "rebuild-graph", help="Rebuild the notes graph"
    )

    reintegrate_parser = subparsers.add_parser(
        "reintegrate", help="Reindex notes and optionally rebuild graph"
    )
    reintegrate_parser.add_argument(
        "--after",
        help="Only process notes modified after this ISO datetime (e.g., 2024-01-01T00:00:00)",
    )
    reintegrate_parser.add_argument(
        "--before",
        help="Only process notes modified before this ISO datetime",
    )
    reintegrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )
    reintegrate_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip graph rebuild after reindexing",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = NotesCLI(args.base_path)

    try:
        if args.command in ["create", "new"]:
            cli.create_note(
                args.title,
                args.content,
                args.tags,
                args.attachments,
                auto_tags=args.auto_tags,
            )
        elif args.command in ["get", "show"]:
            cli.get_note(args.note_id)
        elif args.command == "edit":
            cli.edit_note(args.note_id)
        elif args.command in ["search", "find"]:
            cli.search_notes(args.query, args.mode, args.limit)
        elif args.command in ["list", "ls"]:
            cli.list_notes(args.sort)
        elif args.command in ["delete", "rm"]:
            cli.delete_note(args.note_id)
        elif args.command == "attach":
            cli.add_attachment(args.note_id, args.file_path)
        elif args.command == "links":
            cli.show_links(args.note_id)
        elif args.command == "rebuild-graph":
            cli.rebuild_graph()
        elif args.command == "reintegrate":
            try:
                summary = reintegrate(
                    cli.storage,
                    cli.search,
                    cli.graph,
                    after=args.after,
                    before=args.before,
                    dry_run=args.dry_run,
                    rebuild_graph=not args.no_graph,
                )
                print(
                    f"Reintegrate complete. Processed {summary['processed']}/{summary['total']} "
                    f"(skipped {summary['skipped']}). "
                    f"Graph rebuilt: {summary['rebuild_graph']}"
                )
            except ValueError as exc:
                print(f"Error: {exc}")
    finally:
        cli.search.close()


if __name__ == "__main__":
    main()
