"""`geyi library` command group."""

from __future__ import annotations

import json
from pathlib import Path

from geyi.analysis import analyze
from geyi.library.index import (
    DEFAULT_INDEX_PATH,
    DEFAULT_SESSION_ROOT,
    build_library_index,
    load_or_build_library_index,
    search_library_index,
)
from geyi.library.retrieval import recall_exact_signature
from geyi.session import SessionStore


def add_arguments(subparsers) -> None:
    parser = subparsers.add_parser("library", help="index and search CANN library metadata")
    library_subparsers = parser.add_subparsers(dest="library_command", required=True)

    index = library_subparsers.add_parser("index", help="validate lockfile and rebuild CANN metadata index")
    index.add_argument("--lock", default="geyi-library.lock", help="path to geyi-library.lock")
    index.add_argument("--out", default=DEFAULT_INDEX_PATH, help="path to write metadata index JSON")
    index.add_argument("--session-root", default=DEFAULT_SESSION_ROOT, help="library session artifact root")
    index.add_argument("--json", action="store_true", help="print index JSON")
    index.set_defaults(func=run_index)

    search = library_subparsers.add_parser("search", help="search hotset metadata by exact op/alias")
    search.add_argument("--op", required=True, help="operator name or alias, e.g. rms_norm")
    search.add_argument("--lock", default="geyi-library.lock", help="path to geyi-library.lock")
    search.add_argument("--index", default=DEFAULT_INDEX_PATH, help="metadata index JSON")
    search.add_argument("--session-root", default=DEFAULT_SESSION_ROOT, help="library session artifact root")
    search.add_argument("--json", action="store_true", help="print result JSON")
    search.set_defaults(func=run_search)

    recall = library_subparsers.add_parser("recall", help="run Strategy 0 exact signature recall for a source/spec")
    recall.add_argument("source", help="CUDA source file")
    recall.add_argument("--spec", required=True, help="path to geyi.yaml")
    recall.add_argument("--lock", default="geyi-library.lock", help="path to geyi-library.lock")
    recall.add_argument("--index", default=DEFAULT_INDEX_PATH, help="metadata index JSON")
    recall.add_argument("--session-root", default=DEFAULT_SESSION_ROOT, help="library session artifact root")
    recall.add_argument("--json", action="store_true", help="print result JSON")
    recall.set_defaults(func=run_recall)


def run_index(args) -> int:
    session = SessionStore.create(args.session_root)
    index = build_library_index(lock_path=args.lock, out_path=args.out)
    session.write_json("library_index.json", index)
    session.write_log("library.log", "Phase 3a library index rebuilt: %s" % args.out)
    payload = {"index": index, "session": str(session.path)}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Indexed %d CANN hotset entries from %d sources" % (len(index["hotset"]), len(index["sources"])))
        print("Index: %s" % Path(args.out))
        print("Session: %s" % session.path)
    return 0


def run_search(args) -> int:
    session = SessionStore.create(args.session_root)
    index = load_or_build_library_index(lock_path=args.lock, index_path=args.index)
    results = search_library_index(index, args.op)
    payload = {
        "op": args.op,
        "match_policy": "exact_op_or_alias",
        "results": results,
        "session": str(session.path),
    }
    session.write_json("query.json", {"op": args.op, "index": args.index, "lock": args.lock})
    session.write_json("results.json", payload)
    session.write_log("library.log", "Phase 3a retrieval query completed for %s" % args.op)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Matches: %d" % len(results))
        for item in results:
            print("- %s source=%s paths=%s" % (item["op"], item["source"], ", ".join(item["source_paths"])))
        print("Session: %s" % session.path)
    return 0 if results else 1


def run_recall(args) -> int:
    session = SessionStore.create(args.session_root)
    analysis = analyze(args.source, spec=args.spec, session_root=str(session.path / "analysis"), write_session=True)
    index = load_or_build_library_index(lock_path=args.lock, index_path=args.index)
    results = recall_exact_signature(analysis.contract, index)
    payload = {
        "entry": analysis.contract.entry,
        "contract_hash": analysis.contract.contract_hash,
        "match_policy": "exact_signature",
        "results": results,
        "session": str(session.path),
    }
    session.write_json("recall.json", payload)
    session.write_log("library.log", "Phase 3a exact signature recall completed for %s" % analysis.contract.entry)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Exact signature matches: %d" % len(results))
        for item in results:
            print("- %s source=%s" % (item["op"], item["source"]))
        print("Session: %s" % session.path)
    return 0 if results else 1
