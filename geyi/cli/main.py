"""Geyi command line entry point."""

from __future__ import annotations

import argparse

from geyi.config import DEFAULT_SESSION_ROOT
from geyi.cli import info as info_command
from geyi.cli import run as run_command
from geyi.cli import setup as setup_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geyi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="analyze a CUDA source and emit a Semantic Contract")
    info.add_argument("source", help="CUDA source file")
    info.add_argument("--spec", required=True, help="path to geyi.yaml")
    info.add_argument("--json", action="store_true", help="print contract JSON")
    info.add_argument("--session-root", default=DEFAULT_SESSION_ROOT, help="session artifact root")
    info.add_argument("--no-session", action="store_true", help="do not write session artifacts")
    info.set_defaults(func=info_command.run)

    run = subparsers.add_parser("run", help="run the Phase 0 end-to-end MVP")
    run_command.add_arguments(run)
    run.set_defaults(func=run_command.run)

    setup = subparsers.add_parser("setup", help="write a minimal Phase -1 local config")
    setup_command.add_arguments(setup)
    setup.set_defaults(func=setup_command.run)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
