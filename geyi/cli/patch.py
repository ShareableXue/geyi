"""`geyi patch` command group."""

from __future__ import annotations

import contextlib
import io
import json

from geyi.phase4 import DEFAULT_MODEL_HARNESS_ROOT, run_patched_python


def add_arguments(subparsers) -> None:
    parser = subparsers.add_parser("patch", help="run model-level patch harnesses")
    patch_subparsers = parser.add_subparsers(dest="patch_command", required=True)

    python = patch_subparsers.add_parser("python", help="run a Python script with Phase 4 PyTorch extension patching")
    python.add_argument("script", help="Python entrypoint to run")
    python.add_argument("script_args", nargs="*", help="arguments passed to the Python script")
    python.add_argument("--cache-root", default=DEFAULT_MODEL_HARNESS_ROOT, help="op-level model harness cache root")
    python.add_argument("--session-root", default=".geyi/sessions", help="session artifact root")
    python.add_argument(
        "--execute-original",
        action="store_true",
        help="call the original PyTorch loader after capture; default is capture-only offline mode",
    )
    python.add_argument("--json", action="store_true", help="print model harness report JSON")
    python.set_defaults(func=run_python)


def run_python(args) -> int:
    if args.json:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = run_patched_python(
                args.script,
                script_args=list(args.script_args),
                cache_root=args.cache_root,
                session_root=args.session_root,
                execute_original=args.execute_original,
            )
    else:
        result = run_patched_python(
            args.script,
            script_args=list(args.script_args),
            cache_root=args.cache_root,
            session_root=args.session_root,
            execute_original=args.execute_original,
        )
    payload = result.report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Phase 4 patch: %s exit_code=%d" % (result.report.status, result.report.exit_code))
        print("Captured source-available ops: %d" % len(result.report.captured_ops))
        for op in result.report.captured_ops:
            print("- %s entry=%s cache_hit=%s spec=%s" % (op.name, op.entry, op.cache_hit, op.spec_path))
        print("Black-box extensions: %d" % len(result.report.black_box_extensions))
        for boundary in result.report.black_box_extensions:
            print("- %s %s" % (boundary.api, boundary.path))
        print("Session: %s" % result.session.path)
    return result.report.exit_code
