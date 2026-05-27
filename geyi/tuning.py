"""Phase 3b autotune harness."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from geyi.config import DEFAULT_SESSION_ROOT
from geyi.phase1 import Phase1RunResult, run_phase1
from geyi.phase3 import build_tuning_report, write_tuning_report
from geyi.profiler import no_profile_report, run_msprof_profile


@dataclass
class TuneResult:
    baseline: Phase1RunResult
    tuning_report: Dict
    out_path: Path


@dataclass
class GeneratedProfileCommand:
    kernel_name: str
    command: str
    case_index: int
    case_name: str
    output_path: Path


def run_tune(
    source: str,
    spec: str,
    out: Optional[str] = None,
    session_root: str = DEFAULT_SESSION_ROOT,
    reproducible_command: Optional[str] = None,
    backend: str = "tilelang",
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
    search_space: str = "small",
    kernel_name: Optional[str] = None,
    profile_command: Optional[str] = None,
    profile_generated: bool = False,
    msprof_bin: str = "msprof",
    profile_timeout: int = 300,
    profile_warm_up: int = 10,
    profile_launch_count: int = 5,
    profile_output: Optional[str] = None,
) -> TuneResult:
    out_path = Path(out) if out else Path(".geyi") / "tune" / Path(source).stem
    baseline = run_phase1(
        source,
        spec=spec,
        out=str(out_path / "baseline"),
        session_root=session_root,
        reproducible_command=reproducible_command,
        backend=backend,
        target=target,
        npu_arch=npu_arch,
    )
    performance_dir = baseline.analysis.session.path / "performance"
    profiler_output = Path(profile_output) if profile_output else performance_dir / "msprof_raw"
    if profile_command:
        performance = run_msprof_profile(
            kernel_name=kernel_name or baseline.analysis.contract.entry,
            profile_command=profile_command,
            output_dir=performance_dir,
            msprof_bin=msprof_bin,
            timeout_seconds=profile_timeout,
            warm_up=profile_warm_up,
            launch_count=profile_launch_count,
            profiler_output=profiler_output,
        )
    elif profile_generated:
        generated_profile = build_generated_profile_command(baseline, performance_dir)
        performance = run_msprof_profile(
            kernel_name=kernel_name or generated_profile.kernel_name,
            profile_command=generated_profile.command,
            output_dir=performance_dir,
            msprof_bin=msprof_bin,
            timeout_seconds=profile_timeout,
            warm_up=profile_warm_up,
            launch_count=profile_launch_count,
            profiler_output=profiler_output,
        )
        performance.notes.append(
            "profiled generated operator case_%02d_%s" % (generated_profile.case_index, generated_profile.case_name)
        )
    else:
        performance = no_profile_report()

    report = build_tuning_report(
        baseline.analysis.contract,
        baseline.plan,
        baseline.verification_report,
        search_space=search_space,
        performance_report=performance,
    )
    write_tuning_report(out_path, baseline.analysis.session, report)
    return TuneResult(baseline=baseline, tuning_report=report, out_path=out_path)


def build_generated_profile_command(baseline: Phase1RunResult, output_dir: Path) -> GeneratedProfileCommand:
    plan = baseline.plan
    if plan.backend != "ascendc" or str(plan.parameters.get("target")) != "cann":
        raise RuntimeError("--profile-generated requires --backend ascendc --target cann")
    if not baseline.verification_report.passed:
        raise RuntimeError("--profile-generated requires a passing generated-operator verification run")

    executable = Path(baseline.artifact.path)
    if not executable.exists():
        raise RuntimeError("generated executable is missing: %s" % executable)

    case_index, case = first_profile_case(baseline.verification_report.case_results)
    case_name = str(case.get("case") or case_index)
    case_dir = Path(baseline.project.root) / "build" / "output" / ("case_%02d_%s" % (case_index, case_name))
    input_paths = [case_dir / "input" / ("input%d.bin" % index) for index in range(len(plan.parameters.get("inputs") or []))]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise RuntimeError("generated profile inputs are missing: %s" % ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("generated_case_%02d_%s_output.bin" % (case_index, case_name))
    dims = profile_dims(plan, case)
    command = [str(executable)] + [str(path) for path in input_paths] + [str(output_path)] + dims
    entry = str(baseline.project.metadata.get("operator_entry") or baseline.analysis.contract.entry)
    return GeneratedProfileCommand(
        kernel_name="%s_kernel" % entry,
        command=" ".join(shlex.quote(part) for part in command),
        case_index=case_index,
        case_name=case_name,
        output_path=output_path,
    )


def first_profile_case(case_results) -> tuple[int, Dict]:
    for index, case in enumerate(case_results or []):
        if case.get("passed") and not case.get("skipped_device"):
            return index, dict(case)
    raise RuntimeError("no non-empty passing generated-operator case is available for profiling")


def profile_dims(plan, case: Dict) -> list[str]:
    operation = str(plan.parameters.get("operation") or "")
    if operation in {"transpose2d", "row_sum"}:
        return [str(case["rows"]), str(case["cols"])]
    return [str(case["n"])]
