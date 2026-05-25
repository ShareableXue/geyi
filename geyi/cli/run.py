"""`geyi run` command."""

from __future__ import annotations

import json
import shlex

from geyi.phase0 import run_phase0


def add_arguments(parser) -> None:
    parser.add_argument("source", help="CUDA source file")
    parser.add_argument("--spec", required=True, help="path to geyi.yaml")
    parser.add_argument("--out", default=None, help="generated project/cache output directory")
    parser.add_argument("--json", action="store_true", help="print verification report JSON")
    parser.add_argument("--session-root", default=".geyi/sessions", help="session artifact root")


def run(args) -> int:
    result = run_phase0(
        args.source,
        spec=args.spec,
        out=args.out,
        session_root=args.session_root,
        reproducible_command=reproducible_command(args),
    )
    if args.json:
        print(json.dumps(result.verification_report.to_dict(), indent=2, sort_keys=True))
    else:
        print_text_report(result)
    return 0 if result.verification_report.passed else 1


def print_text_report(result) -> None:
    report = result.verification_report
    print("Kernel: %s" % result.analysis.contract.entry)
    print("Contract: %s" % report.contract_hash)
    print("Plan: %s backend=%s template=%s" % (result.plan.strategy, result.plan.backend, result.plan.template))
    print("Generated: %s" % result.project.root)
    print("Artifact: %s hash=%s reused=%s" % (result.artifact.path, report.artifact_hash, result.artifact.reused))
    print("Verify: %s passed=%s max_abs_diff=%s" % (report.level.value, report.passed, report.max_abs_diff))
    print("Coverage: shapes=%s dtypes=%s hardware=%s" % (report.coverage.shapes, report.coverage.dtypes, report.coverage.hardware))
    print("Out: %s cache_hit=%s" % (result.out_path, result.cache_hit))
    print("Session: %s" % result.analysis.session.path)


def reproducible_command(args) -> str:
    pieces = ["geyi", "run", args.source, "--spec", args.spec]
    if args.out:
        pieces.extend(["--out", args.out])
    return " ".join(shlex.quote(str(item)) for item in pieces)

