from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from geyi.phase1 import run_phase1
from geyi.tuning import build_generated_profile_command, run_tune


ROOT = Path(__file__).resolve().parents[1]


def test_conservative_opt_level_preserves_golden_verification(tmp_path):
    result = run_phase1(
        "examples/vector_add/vector_add.cu",
        spec="examples/vector_add/geyi.yaml",
        out=str(tmp_path / "out"),
        session_root=str(tmp_path / "sessions"),
        opt_level="conservative",
    )

    assert result.verification_report.passed
    assert result.plan.optimization_hints["opt_level"] == "conservative"
    assert result.plan.optimization_hints["status"] == "eligible_after_verification"
    first_hint = result.plan.optimization_hints["hints"][0]
    assert first_hint["confidence"] > 0.80
    assert first_hint["source_digest"]
    assert first_hint["knowledge_claims"]
    assert first_hint["evidence"][0]["kind"] == "cannbot_tiling_knowledge"
    assert (result.analysis.session.path / "optimization_hints.json").exists()
    assert (tmp_path / "out" / "optimization_hints.json").exists()


def test_tune_small_search_space_includes_verification_report(tmp_path):
    result = run_tune(
        "examples/vector_add/vector_add.cu",
        spec="examples/vector_add/geyi.yaml",
        out=str(tmp_path / "tune"),
        session_root=str(tmp_path / "sessions"),
        search_space="small",
    )

    assert result.baseline.verification_report.passed
    assert result.tuning_report["verification_required_before_selection"] is True
    assert result.tuning_report["search_space"]["candidate_count"] > 0
    assert result.tuning_report["search_space"]["measurement"]["recommended_profiler"].startswith("msprof op")
    assert result.tuning_report["search_space"]["candidates"][0]["estimated_ub_bytes"] is not None
    assert result.tuning_report["performance_report"]["status"] == "not_requested"
    assert result.tuning_report["candidate_reports"][0]["verification"]["passed"] is True
    assert (tmp_path / "tune" / "tuning_report.json").exists()


def test_tune_can_capture_msprof_baseline_report(tmp_path):
    fake_msprof = tmp_path / "msprof"
    fake_msprof.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"fake msprof args: $@\"\n"
        "echo \"kernel_time_us: 123.5\"\n",
        encoding="utf-8",
    )
    fake_msprof.chmod(0o755)

    result = run_tune(
        "examples/vector_add/vector_add.cu",
        spec="examples/vector_add/geyi.yaml",
        out=str(tmp_path / "tune"),
        session_root=str(tmp_path / "sessions"),
        search_space="small",
        kernel_name="vector_add_kernel",
        profile_command=f"{sys.executable} -c \"print('operator run')\"",
        msprof_bin=str(fake_msprof),
    )

    perf = result.tuning_report["performance_report"]
    assert perf["status"] == "measured"
    assert perf["kernel_name"] == "vector_add_kernel"
    assert perf["metrics"][0]["name"] == "kernel_time"
    assert perf["metrics"][0]["value"] == 123.5
    assert (result.baseline.analysis.session.path / "performance_report.json").exists()


def test_generated_profile_command_uses_ascendc_output_case(tmp_path):
    project_root = tmp_path / "generated"
    case_dir = project_root / "build" / "output" / "case_00_basic"
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True)
    executable = project_root / "build" / "vector_add"
    executable.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    executable.chmod(0o755)
    for index in range(2):
        (input_dir / ("input%d.bin" % index)).write_bytes(b"data")

    baseline = SimpleNamespace(
        plan=SimpleNamespace(backend="ascendc", parameters={"target": "cann", "operation": "add", "inputs": ["a", "b"]}),
        artifact=SimpleNamespace(path=str(executable)),
        project=SimpleNamespace(root=str(project_root), metadata={"operator_entry": "vector_add"}),
        verification_report=SimpleNamespace(passed=True, case_results=[{"case": "basic", "n": 1024, "passed": True}]),
        analysis=SimpleNamespace(contract=SimpleNamespace(entry="vector_add")),
    )

    generated = build_generated_profile_command(baseline, tmp_path / "perf")

    assert generated.kernel_name == "vector_add_kernel"
    assert str(executable) in generated.command
    assert "input0.bin" in generated.command
    assert generated.command.endswith("1024")
    assert generated.output_path.parent == tmp_path / "perf"


def test_phase3_acceptance_cli_commands(tmp_path):
    run_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "run",
            "examples/vector_add/vector_add.cu",
            "--spec",
            "examples/vector_add/geyi.yaml",
            "--opt-level",
            "conservative",
            "--out",
            str(tmp_path / "run_out"),
            "--session-root",
            str(tmp_path / "sessions"),
            "--json",
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    run_payload = json.loads(run_completed.stdout)
    assert run_payload["passed"] is True
    assert (tmp_path / "run_out" / "optimization_hints.json").exists()

    tune_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "tune",
            "examples/vector_add/vector_add.cu",
            "--spec",
            "examples/vector_add/geyi.yaml",
            "--search-space",
            "small",
            "--out",
            str(tmp_path / "tune_out"),
            "--session-root",
            str(tmp_path / "sessions"),
            "--json",
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    tune_payload = json.loads(tune_completed.stdout)
    assert tune_payload["baseline_verification"]["passed"] is True
    assert tune_payload["search_space"]["candidate_count"] > 0
