#!/usr/bin/env python3
"""
Black Hole CLI wrapper.

Keeps argparse/printing here while core BH behaviors live in `notes.bh.app`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .bh.app import BlackHoleApp


# Back-compat for tests/imports
BlackHoleCLI = BlackHoleApp


def main():
    parser = argparse.ArgumentParser(description="Black Hole CLI")
    parser.add_argument("--base-path", type=Path, help="Base path for storage (defaults to ~/.notes)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add raw content or ingest a file")
    add_parser.add_argument("content", help="Content text or path to a file")
    add_parser.add_argument("-t", "--title", help="Optional title override")
    add_parser.add_argument("--tags", nargs="+", help="Optional tags")
    add_parser.add_argument("--auto-tags", action="store_true", help="Use configured tagger to suggest tags")
    add_parser.add_argument("--no-transcribe", action="store_true", help="Skip audio transcription")

    search_parser = subparsers.add_parser("search", help="Semantic/hybrid search")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("-l", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument(
        "-m",
        "--mode",
        choices=["semantic", "keyword", "hybrid"],
        default="semantic",
        help="Search mode",
    )

    proactive_parser = subparsers.add_parser("proactive", help="Proactive routines (todo, brief)")
    proactive_sub = proactive_parser.add_subparsers(dest="mode", required=True)

    todo_parser = proactive_sub.add_parser("todo", help="Aggregate TODOs into a superstructure note")
    todo_parser.add_argument("--scope", choices=["new", "all"], default="new")

    brief_parser = proactive_sub.add_parser("brief", help="Generate context briefings grouped by category/tag")
    brief_parser.add_argument("--scope", choices=["new", "all"], default="new")

    digest_parser = subparsers.add_parser("digest", help="Generate a digest of notes")
    digest_parser.add_argument("--scope", choices=["new", "all"], default="new")

    browse_parser = subparsers.add_parser("browse", help="Traverse notes in a compact view")
    browse_parser.add_argument("-l", "--limit", type=int, default=15, help="Number of notes to show")
    browse_parser.add_argument("-s", "--sort", choices=["modified", "created", "title"], default="modified")

    show_parser = subparsers.add_parser("show", help="Render a specific note")
    show_parser.add_argument("note_id", help="Note ID to render")

    open_parser = subparsers.add_parser("open", help="Open a note in $EDITOR")
    open_parser.add_argument("note_id", help="Note ID to open")

    subparsers.add_parser("status", help="Show Black Hole status")

    watch_parser = subparsers.add_parser("create-bh", help="Watch a directory and ingest new files")
    watch_parser.add_argument("path", type=Path, help="Directory to watch")
    watch_parser.add_argument("--interval", type=int, default=15, help="Polling interval seconds when following")
    watch_parser.add_argument("--follow", action="store_true", help="Continue watching instead of a single sync pass")

    retag_parser = subparsers.add_parser("retag", help="LLM-retag notes")
    retag_parser.add_argument("--scope", choices=["new", "all"], default="new")

    subparsers.add_parser("lint", help="Check storage/index health")

    link_parser = subparsers.add_parser("link", help="Refresh and show links for a note")
    link_parser.add_argument("note_id", help="Note ID to link")
    link_parser.add_argument("--min-weight", type=float, default=0.0, help="Minimum weight to show")

    subparsers.add_parser("yank", help="Create a note from clipboard text")

    args = parser.parse_args()

    app = BlackHoleApp(args.base_path)
    try:
        if args.command == "add":
            note_id = app.add(
                args.content,
                title=args.title,
                tags=args.tags,
                auto_tags=args.auto_tags,
                transcribe_audio=not args.no_transcribe,
            )
            print(f"Created note: {note_id}")
        elif args.command == "search":
            results = app.search_bh(args.query, limit=args.limit, mode=args.mode)
            if not results:
                print("No results found.")
            else:
                print(f"\nFound {len(results)} results:\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r.title}")
                    print(f"   ID: {r.note_id}")
                    print(f"   Score: {r.score:.3f}")
                    print(f"   {r.snippet}")
                    print()
        elif args.command == "proactive":
            if args.mode == "todo":
                note_id = app.proactive_todo(scope=args.scope)
                print(f"Proactive TODO complete -> note {note_id}")
            elif args.mode == "brief":
                note_id = app.proactive_brief(scope=args.scope)
                print(f"Proactive brief complete -> note {note_id}")
        elif args.command == "digest":
            note_id = app.digest(scope=args.scope)
            print(f"Digest complete -> note {note_id}")
        elif args.command == "browse":
            notes = app.browse(limit=args.limit, sort_by=args.sort)
            if not notes:
                print("No notes available.")
            else:
                print(f"Browsing {len(notes)} notes (sorted by {args.sort}):\n")
                for note in notes:
                    preview = (note.content.strip().splitlines() or [""])[0]
                    if len(preview) > 140:
                        preview = preview[:137] + "..."
                    tags = f" [{', '.join(note.tags)}]" if note.tags else ""
                    print(f"- {note.title}{tags}")
                    print(f"  ID: {note.id}")
                    print(f"  Modified: {note.modified_at.isoformat()}")
                    if preview:
                        print(f"  {preview}")
                    print()
        elif args.command == "show":
            note = app.get_note(args.note_id)
            if not note:
                print(f"Note not found: {args.note_id}")
            else:
                print(f"# {note.title} ({note.id})")
                if note.tags:
                    print(f"Tags: {', '.join(note.tags)}")
                print(f"Created: {note.created_at.isoformat()}")
                print(f"Modified: {note.modified_at.isoformat()}")
                print("")
                print(note.content.strip())
        elif args.command == "open":
            if not app.open_note(args.note_id):
                print(f"Note not found: {args.note_id}")
        elif args.command == "status":
            st = app.status()
            print("Black Hole status")
            print(f"- Notes path: {st['notes_path']}")
            print(f"- Total notes: {st['total_notes']}")
            last = st.get("last_processed") or {}
            print(f"- Last TODO run: {last.get('todo', 'never')}")
            print(f"- Last brief run: {last.get('brief', 'never')}")
            print(f"- Last digest run: {last.get('digest', 'never')}")
            watches = st.get("watches") or {}
            if watches:
                print("- Watched paths:")
                for path, ts in watches.items():
                    print(f"  - {path} (last seen: {ts})")
            else:
                print("- Watched paths: none")
        elif args.command == "create-bh":
            if not args.follow:
                note_ids = app.ingest_directory_once(args.path)
                print(f"Ingested {len(note_ids)} new files from {args.path}")
            else:
                print(f"Ingesting from {args.path} (follow=True, interval={args.interval}s)")
                try:
                    while True:
                        note_ids = app.ingest_directory_once(args.path)
                        if note_ids:
                            print(f"Ingested {len(note_ids)} new files from {args.path}")
                        time.sleep(args.interval)
                except KeyboardInterrupt:
                    print("Stopped watching.")
        elif args.command == "retag":
            updated = app.retag(scope=args.scope)
            print(f"Retagged {updated} notes.")
        elif args.command == "lint":
            issues = app.lint()
            if not issues:
                print("Lint OK: no issues found.")
            else:
                print("Lint issues:")
                for issue in issues:
                    print(f"- {issue}")
        elif args.command == "link":
            links = app.link(args.note_id, min_weight=args.min_weight)
            if not links:
                print("No links found.")
            else:
                print(f"Links for {args.note_id}:")
                for l in links:
                    print(f"- {l.title} ({l.note_id}): {l.weight:.3f}")
        elif args.command == "yank":
            text = app.yank()
            if not text:
                print("Clipboard is empty or unavailable.")
            else:
                print("Captured clipboard into a new note.")
    finally:
        app.close()


if __name__ == "__main__":
    main()

