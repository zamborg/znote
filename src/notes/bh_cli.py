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
    add_parser.add_argument("--stream", help="Stream (e.g. inbox, work/bh)")
    add_parser.add_argument("--auto-tags", action="store_true", help="Use configured tagger to suggest tags")
    add_parser.add_argument("--no-transcribe", action="store_true", help="Skip audio transcription")

    new_parser = subparsers.add_parser("new", help="Create a markdown note in $EDITOR")
    new_parser.add_argument("-t", "--title", help="Optional title prefill")
    new_parser.add_argument("--tags", nargs="+", help="Optional tags")
    new_parser.add_argument("--stream", help="Stream (e.g. inbox, work/bh)")
    new_parser.add_argument("--auto-tags", action="store_true", help="Use configured tagger to suggest tags")

    todo_parser = subparsers.add_parser("todo", help="Todo items")
    todo_sub = todo_parser.add_subparsers(dest="todo_command", required=True)

    todo_add = todo_sub.add_parser("add", help="Create a todo from text")
    todo_add.add_argument("text", help="Todo text (markdown allowed)")
    todo_add.add_argument("-t", "--title", help="Optional title override")
    todo_add.add_argument("--due", help="Optional due datetime (ISO format)")
    todo_add.add_argument("--tags", nargs="+", help="Optional tags")
    todo_add.add_argument("--stream", help="Stream (e.g. inbox, work/bh)")
    todo_add.add_argument("--auto-tags", action="store_true", help="Use configured tagger to suggest tags")

    todo_new = todo_sub.add_parser("new", help="Create a todo in $EDITOR")
    todo_new.add_argument("-t", "--title", help="Optional title prefill")
    todo_new.add_argument("--due", help="Optional due datetime (ISO format)")
    todo_new.add_argument("--tags", nargs="+", help="Optional tags")
    todo_new.add_argument("--stream", help="Stream (e.g. inbox, work/bh)")
    todo_new.add_argument("--auto-tags", action="store_true", help="Use configured tagger to suggest tags")

    todo_list = todo_sub.add_parser("list", help="List todos")
    todo_list.add_argument("--status", choices=["open", "done", "archived", "all"], default="open")
    todo_list.add_argument("--sort", choices=["due", "modified", "created", "title"], default="due")
    todo_list.add_argument("-l", "--limit", type=int, default=25, help="Max results")
    todo_list.add_argument("--stream", help="Filter to a stream subtree (e.g. work)")

    todo_done = todo_sub.add_parser("done", help="Mark a todo done")
    todo_done.add_argument("todo_id", help="Todo note id")

    todo_archive = todo_sub.add_parser("archive", help="Archive a todo (or any item)")
    todo_archive.add_argument("item_id", help="Item id to archive")

    search_parser = subparsers.add_parser("search", help="Semantic/hybrid search")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("-l", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument("--stream", help="Filter results to a stream subtree (e.g. work)")
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
    todo_parser.add_argument("--stream", help="Filter to a stream subtree (e.g. work)")

    brief_parser = proactive_sub.add_parser("brief", help="Generate context briefings grouped by category/tag")
    brief_parser.add_argument("--scope", choices=["new", "all"], default="new")
    brief_parser.add_argument("--stream", help="Filter to a stream subtree (e.g. work)")

    digest_parser = subparsers.add_parser("digest", help="Generate a digest of notes")
    digest_parser.add_argument("--scope", choices=["new", "all"], default="new")
    digest_parser.add_argument("--stream", help="Filter to a stream subtree (e.g. work)")

    browse_parser = subparsers.add_parser("browse", help="Traverse notes in a compact view")
    browse_parser.add_argument("-l", "--limit", type=int, default=15, help="Number of notes to show")
    browse_parser.add_argument("-s", "--sort", choices=["modified", "created", "title"], default="modified")
    browse_parser.add_argument("--stream", help="Filter to a stream subtree (e.g. work)")

    show_parser = subparsers.add_parser("show", help="Render a specific note")
    show_parser.add_argument("note_id", help="Note ID to render")

    open_parser = subparsers.add_parser("open", help="Open a note in $EDITOR")
    open_parser.add_argument("note_id", help="Note ID to open")

    subparsers.add_parser("status", help="Show Black Hole status")

    watch_parser = subparsers.add_parser("create-bh", help="Watch a directory and ingest new files")
    watch_parser.add_argument("path", type=Path, help="Directory to watch")
    watch_parser.add_argument("--interval", type=int, default=15, help="Polling interval seconds when following")
    watch_parser.add_argument("--follow", action="store_true", help="Continue watching instead of a single sync pass")
    watch_parser.add_argument("--stream", help="Stream to assign ingested notes")

    retag_parser = subparsers.add_parser("retag", help="LLM-retag notes")
    retag_parser.add_argument("--scope", choices=["new", "all"], default="new")
    retag_parser.add_argument("--stream", help="Filter to a stream subtree (e.g. work)")

    cost_parser = subparsers.add_parser("cost", help="Cost tracking / ledger")
    cost_sub = cost_parser.add_subparsers(dest="cost_command", required=True)
    cost_daily = cost_sub.add_parser("daily", help="Daily totals")
    cost_daily.add_argument("--days", type=int, default=14, help="Number of days to show")
    cost_events = cost_sub.add_parser("events", help="List recent cost events")
    cost_events.add_argument("--day", help="Filter by day (YYYY-MM-DD)")
    cost_events.add_argument("-l", "--limit", type=int, default=50, help="Max events")

    completion_parser = subparsers.add_parser("completion", help="Completionist documents")
    completion_sub = completion_parser.add_subparsers(dest="completion_mode", required=True)
    completion_stream = completion_sub.add_parser("stream", help="Build a stream index note")
    completion_stream.add_argument("stream", help="Stream to index (e.g. work/bh)")
    completion_stream.add_argument("-l", "--limit", type=int, default=500, help="Max items")

    subparsers.add_parser("lint", help="Check storage/index health")

    link_parser = subparsers.add_parser("link", help="Refresh and show links for a note")
    link_parser.add_argument("note_id", help="Note ID to link")
    link_parser.add_argument("--min-weight", type=float, default=0.0, help="Minimum weight to show")

    yank_parser = subparsers.add_parser("yank", help="Create a note from clipboard text")
    yank_parser.add_argument("--stream", help="Stream (e.g. inbox, work/bh)")

    args = parser.parse_args()

    app = BlackHoleApp(args.base_path)
    try:
        if args.command == "add":
            created = app.add(
                args.content,
                title=args.title,
                tags=args.tags,
                auto_tags=args.auto_tags,
                transcribe_audio=not args.no_transcribe,
                stream=args.stream,
            )
            if isinstance(created, list):
                print(f"Ingested {len(created)} items.")
                for note_id in created:
                    print(f"- {note_id}")
            else:
                print(f"Created note: {created}")
        elif args.command == "new":
            note_id = app.new_note(
                title=args.title,
                tags=args.tags,
                auto_tags=args.auto_tags,
                stream=args.stream,
            )
            if not note_id:
                print("Cancelled (empty note).")
            else:
                print(f"Created note: {note_id}")
        elif args.command == "todo":
            if args.todo_command == "add":
                todo_id = app.create_todo(
                    args.text,
                    title=args.title,
                    tags=args.tags,
                    auto_tags=args.auto_tags,
                    due=args.due,
                    stream=args.stream,
                )
                print(f"Created todo: {todo_id}")
            elif args.todo_command == "new":
                todo_id = app.new_todo(
                    title=args.title,
                    tags=args.tags,
                    auto_tags=args.auto_tags,
                    stream=args.stream,
                    due=args.due,
                )
                if not todo_id:
                    print("Cancelled (empty todo).")
                else:
                    print(f"Created todo: {todo_id}")
            elif args.todo_command == "list":
                todos = app.list_todos(
                    status=args.status,
                    stream=args.stream,
                    limit=args.limit,
                    sort_by=args.sort,
                )
                if not todos:
                    print("No todos.")
                else:
                    print(f"Todos ({len(todos)}) [{args.status}]:\n")
                    for t in todos:
                        due = f" due {t.due_at.date().isoformat()}" if t.due_at else ""
                        state = "archived" if t.archived_at else ("done" if t.completed_at else "open")
                        tags = f" [{', '.join(t.tags)}]" if t.tags else ""
                        print(f"- {t.title}{tags}{due}")
                        print(f"  ID: {t.id} ({state})")
                        print(f"  Stream: {t.stream}")
            elif args.todo_command == "done":
                if app.mark_todo_done(args.todo_id):
                    print(f"Marked done: {args.todo_id}")
                else:
                    print(f"Todo not found: {args.todo_id}")
            elif args.todo_command == "archive":
                if app.archive_item(args.item_id):
                    print(f"Archived: {args.item_id}")
                else:
                    print(f"Item not found: {args.item_id}")
        elif args.command == "search":
            results = app.search_bh(args.query, limit=args.limit, mode=args.mode, stream=args.stream)
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
                note_id = app.proactive_todo(scope=args.scope, stream=args.stream)
                print(f"Proactive TODO complete -> note {note_id}")
            elif args.mode == "brief":
                note_id = app.proactive_brief(scope=args.scope, stream=args.stream)
                print(f"Proactive brief complete -> note {note_id}")
        elif args.command == "digest":
            note_id = app.digest(scope=args.scope, stream=args.stream)
            print(f"Digest complete -> note {note_id}")
        elif args.command == "browse":
            notes = app.browse(limit=args.limit, sort_by=args.sort, stream=args.stream)
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
                note_ids = app.ingest_directory_once(args.path, stream=args.stream)
                print(f"Ingested {len(note_ids)} new files from {args.path}")
            else:
                print(f"Ingesting from {args.path} (follow=True, interval={args.interval}s)")
                try:
                    while True:
                        note_ids = app.ingest_directory_once(args.path, stream=args.stream)
                        if note_ids:
                            print(f"Ingested {len(note_ids)} new files from {args.path}")
                        time.sleep(args.interval)
                except KeyboardInterrupt:
                    print("Stopped watching.")
        elif args.command == "retag":
            updated = app.retag(scope=args.scope, stream=args.stream)
            print(f"Retagged {updated} notes.")
        elif args.command == "cost":
            if args.cost_command == "daily":
                rows = app.cost_daily(days=args.days)
                if not rows:
                    print("No cost events recorded.")
                else:
                    print("Daily costs:")
                    for row in rows:
                        print(
                            f"- {row['day']}: ${row['cost_usd']:.6f} "
                            f"({row['events']} events, {row['unknown_cost_events']} unknown)"
                        )
            elif args.cost_command == "events":
                rows = app.cost_events(day=args.day, limit=args.limit)
                if not rows:
                    print("No cost events.")
                else:
                    for row in rows:
                        cost = row["cost_usd"]
                        cost_str = f"${float(cost):.6f}" if cost is not None else "unknown"
                        model = row["model"] or "unknown-model"
                        print(f"- {row['created_at']} {row['provider']} {row['operation']} {model} -> {cost_str}")
        elif args.command == "completion":
            if args.completion_mode == "stream":
                note_id = app.completion_stream(args.stream, limit=args.limit)
                print(f"Completion stream index -> note {note_id}")
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
            text = app.yank(stream=getattr(args, "stream", None))
            if not text:
                print("Clipboard is empty or unavailable.")
            else:
                print("Captured clipboard into a new note.")
    finally:
        app.close()


if __name__ == "__main__":
    main()
