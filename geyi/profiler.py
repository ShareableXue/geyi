"""Optional msprof performance profiling integration."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from geyi.contract.model import to_jsonable


@dataclass
class PerformanceMetric:
    name: str
    value: float
    unit: str
    source: str


@dataclass
class PerformanceReport:
    status: str
    profiler: str
    kernel_name: Optional[str]
    command: List[str]
    exit_code: Optional[int]
    elapsed_seconds: Optional[float]
    metrics: List[PerformanceMetric] = field(default_factory=list)
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return to_jsonable(self)


def no_profile_report(reason: str = "no profile command supplied") -> PerformanceReport:
    return PerformanceReport(
        status="not_requested",
        profiler="msprof",
        kernel_name=None,
        command=[],
        exit_code=None,
        elapsed_seconds=None,
        notes=[reason],
    )


def run_msprof_profile(
    kernel_name: str,
    profile_command: str,
    output_dir: Path,
    msprof_bin: str = "msprof",
    timeout_seconds: int = 300,
    warm_up: Optional[int] = None,
    launch_count: Optional[int] = None,
    profiler_output: Optional[Path] = None,
) -> PerformanceReport:
    if not kernel_name:
        raise ValueError("kernel_name is required when profile_command is supplied")
    command = [msprof_bin, "op", "--kernel-name=%s" % kernel_name]
    if warm_up is not None:
        command.append("--warm-up=%d" % int(warm_up))
    if launch_count is not None:
        command.append("--launch-count=%d" % int(launch_count))
    if profiler_output is not None:
        command.append("--output=%s" % profiler_output)
    command.extend(shlex.split(profile_command))
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "msprof_stdout.log"
    stderr_log = output_dir / "msprof_stderr.log"

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.monotonic() - started
    except FileNotFoundError as exc:
        report = PerformanceReport(
            status="profiler_missing",
            profiler=msprof_bin,
            kernel_name=kernel_name,
            command=command,
            exit_code=None,
            elapsed_seconds=None,
            notes=["msprof executable was not found: %s" % exc],
        )
        write_report(output_dir / "performance_report.json", report)
        return report
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        report = PerformanceReport(
            status="timeout",
            profiler=msprof_bin,
            kernel_name=kernel_name,
            command=command,
            exit_code=None,
            elapsed_seconds=elapsed,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            stdout_tail=tail_text(stdout),
            stderr_tail=tail_text(stderr),
            notes=["msprof command timed out after %d seconds" % timeout_seconds],
        )
        write_report(output_dir / "performance_report.json", report)
        return report

    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    metrics = parse_msprof_metrics(completed.stdout + "\n" + completed.stderr)
    report = PerformanceReport(
        status="measured" if completed.returncode == 0 else "failed",
        profiler=msprof_bin,
        kernel_name=kernel_name,
        command=command,
        exit_code=completed.returncode,
        elapsed_seconds=elapsed,
        metrics=metrics,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        stdout_tail=tail_text(completed.stdout),
        stderr_tail=tail_text(completed.stderr),
        notes=[] if metrics else ["msprof completed but no known latency metric pattern was parsed"],
    )
    write_report(output_dir / "performance_report.json", report)
    return report


def parse_msprof_metrics(text: str) -> List[PerformanceMetric]:
    metrics: List[PerformanceMetric] = []
    metrics.extend(parse_json_metrics(text))
    patterns = [
        r"(?P<name>task[_\s-]*duration|kernel[_\s-]*time|op[_\s-]*time|duration|elapsed|aicore[_\s-]*time)\s*(?:\([^)]*\))?\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>us|µs|ms|s)?",
        r"(?P<name>kernel[_\s-]*time|op[_\s-]*time|duration|elapsed|aicore[_\s-]*time)_(?P<unit>us|ms|s)\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_name = normalize_metric_name(match.group("name"))
            unit = normalize_unit(match.groupdict().get("unit") or "us")
            metrics.append(
                PerformanceMetric(
                    name=raw_name,
                    value=normalize_value(float(match.group("value")), unit),
                    unit="us",
                    source="msprof_text",
                )
            )
    for match in re.finditer(r"(?P<name>block[_\s-]*dim|current[_\s-]*freq|rated[_\s-]*freq)\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)", text, flags=re.IGNORECASE):
        metrics.append(
            PerformanceMetric(
                name=normalize_metric_name(match.group("name")),
                value=float(match.group("value")),
                unit="count",
                source="msprof_text",
            )
        )
    return dedupe_metrics(metrics)


def parse_json_metrics(text: str) -> List[PerformanceMetric]:
    metrics = []
    for match in re.finditer(r"\{[^{}]*(?:kernel_time|op_time|duration|elapsed)[^{}]*\}", text, flags=re.IGNORECASE):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for key, value in payload.items():
            key_lower = str(key).lower()
            if not any(token in key_lower for token in ["kernel_time", "op_time", "duration", "elapsed"]):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            unit = "us"
            if key_lower.endswith("_ms"):
                unit = "ms"
            elif key_lower.endswith("_s"):
                unit = "s"
            metrics.append(
                PerformanceMetric(
                    name=normalize_metric_name(str(key)),
                    value=normalize_value(numeric, unit),
                    unit="us",
                    source="msprof_json",
                )
            )
    return metrics


def normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def normalize_unit(unit: str) -> str:
    unit = unit.lower().replace("µ", "u")
    return unit or "us"


def normalize_value(value: float, unit: str) -> float:
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1000000.0
    return value


def dedupe_metrics(metrics: List[PerformanceMetric]) -> List[PerformanceMetric]:
    seen = set()
    deduped = []
    for metric in metrics:
        key = (metric.name, metric.value, metric.unit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(metric)
    return deduped


def tail_text(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def write_report(path: Path, report: PerformanceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
