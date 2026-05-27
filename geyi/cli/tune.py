"""`geyi tune` command."""

from __future__ import annotations

import json
import shlex

from geyi.tuning import run_tune


def add_arguments(parser) -> None:
    parser.add_argument("source", help="CUDA source file")
    parser.add_argument("--spec", required=True, help="path to geyi.yaml")
    parser.add_argument("--out", default=None, help="tuning report output directory")
    parser.add_argument("--backend", default="tilelang", choices=["auto", "tilelang", "ascendc"], help="deterministic backend")
    parser.add_argument("--target", default="auto", choices=["auto", "local_cpu", "scaffold", "cann"], help="execution target")
    parser.add_argument("--npu-arch", default="dav-2201", help="AscendC NPU architecture")
    parser.add_argument("--search-space", default="small", choices=["small"], help="autotune search-space size")
    parser.add_argument("--kernel-name", default=None, help="kernel name passed to msprof op")
    parser.add_argument("--profile-command", default=None, help="operator execution command wrapped by msprof op")
    parser.add_argument("--profile-generated", action="store_true", help="profile the generated AscendC executable after --target cann verification")
    parser.add_argument("--msprof-bin", default="msprof", help="msprof executable path")
    parser.add_argument("--profile-timeout", default=300, type=int, help="msprof timeout in seconds")
    parser.add_argument("--profile-warm-up", default=10, type=int, help="msprof warm-up count")
    parser.add_argument("--profile-launch-count", default=5, type=int, help="msprof launch count")
    parser.add_argument("--profile-output", default=None, help="msprof raw output directory")
    parser.add_argument("--session-root", default=".geyi/sessions", help="session artifact root")
    parser.add_argument("--json", action="store_true", help="print tuning report JSON")


def run(args) -> int:
    result = run_tune(
        args.source,
        spec=args.spec,
        out=args.out,
        session_root=args.session_root,
        reproducible_command=reproducible_command(args),
        backend=args.backend,
        target=args.target,
        npu_arch=args.npu_arch,
        search_space=args.search_space,
        kernel_name=args.kernel_name,
        profile_command=args.profile_command,
        profile_generated=args.profile_generated,
        msprof_bin=args.msprof_bin,
        profile_timeout=args.profile_timeout,
        profile_warm_up=args.profile_warm_up,
        profile_launch_count=args.profile_launch_count,
        profile_output=args.profile_output,
    )
    if args.json:
        print(json.dumps(result.tuning_report, indent=2, sort_keys=True))
    else:
        print("Tune: search_space=%s candidates=%d" % (args.search_space, result.tuning_report["search_space"]["candidate_count"]))
        print("Baseline verify: %s passed=%s" % (result.baseline.verification_report.level.value, result.baseline.verification_report.passed))
        print("Performance: %s metrics=%d" % (
            result.tuning_report["performance_report"]["status"],
            len(result.tuning_report["performance_report"].get("metrics", [])),
        ))
        print("Selected: %s" % result.tuning_report["selected_candidate"])
        print("Out: %s" % result.out_path)
        print("Session: %s" % result.baseline.analysis.session.path)
    return 0 if result.baseline.verification_report.passed else 1


def reproducible_command(args) -> str:
    pieces = ["geyi", "tune", args.source, "--spec", args.spec]
    if args.backend != "tilelang":
        pieces.extend(["--backend", args.backend])
    if args.target != "auto":
        pieces.extend(["--target", args.target])
    if args.npu_arch != "dav-2201":
        pieces.extend(["--npu-arch", args.npu_arch])
    if args.search_space != "small":
        pieces.extend(["--search-space", args.search_space])
    if args.out:
        pieces.extend(["--out", args.out])
    if args.kernel_name:
        pieces.extend(["--kernel-name", args.kernel_name])
    if args.profile_command:
        pieces.extend(["--profile-command", args.profile_command])
    if args.profile_generated:
        pieces.append("--profile-generated")
    if args.msprof_bin != "msprof":
        pieces.extend(["--msprof-bin", args.msprof_bin])
    if args.profile_timeout != 300:
        pieces.extend(["--profile-timeout", args.profile_timeout])
    if args.profile_warm_up != 10:
        pieces.extend(["--profile-warm-up", args.profile_warm_up])
    if args.profile_launch_count != 5:
        pieces.extend(["--profile-launch-count", args.profile_launch_count])
    if args.profile_output:
        pieces.extend(["--profile-output", args.profile_output])
    return " ".join(shlex.quote(str(item)) for item in pieces)
