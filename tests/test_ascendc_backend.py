from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from geyi.phase1 import run_phase1


ROOT = Path(__file__).resolve().parents[1]


def test_ascendc_scaffold_generates_direct_invoke_project(tmp_path):
    result = run_phase1(
        "examples/elementwise_mul/mul.cu",
        spec="examples/elementwise_mul/geyi.yaml",
        backend="ascendc",
        target="scaffold",
        out=str(tmp_path / "out" / "mul"),
        session_root=str(tmp_path / "sessions"),
    )

    generated = result.analysis.session.path / "generated"
    assert result.plan.backend == "ascendc"
    assert result.plan.template == "ascendc.elementwise_binary_1d"
    assert result.plan.parameters["target"] == "scaffold"
    assert result.verification_report.level.value == "compiles_only"
    assert result.verification_report.passed
    assert result.verification_report.coverage.hardware == ["not_run"]
    assert (generated / "op_kernel" / "elementwise_mul_kernel.asc").exists()
    assert (generated / "op_host" / "elementwise_mul.asc").exists()
    assert (generated / "CMakeLists.txt").exists()
    assert (generated / "run.sh").exists()

    kernel = (generated / "op_kernel" / "elementwise_mul_kernel.asc").read_text(encoding="utf-8")
    assert "extern \"C\" __global__ __vector__ void elementwise_mul_kernel" in kernel
    assert "__global__ __aicore__" not in kernel
    assert "elementwise_mul_kernel" in kernel
    assert "AscendC::DataCopyPad(input0Local, input0Gm[progress]" in kernel
    assert "AscendC::Mul(outputLocal, input0Local, input1Local, count)" in kernel
    assert "AscendC::DataCopyPad(outputGm[progress], outputLocal" in kernel

    host = (generated / "op_host" / "elementwise_mul.asc").read_text(encoding="utf-8")
    assert "std::vector<uint8_t*> inputDevice" in host
    assert "uint8_t* outputDevice" in host
    assert "elementwise_mul_kernel<<<tiling.blockNum, nullptr, stream>>>(inputDevice[0], inputDevice[1], outputDevice, tilingDevice);" in host
    assert "std::vector<void*> inputDevice" not in host

    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["generated_project"]["backend"] == "ascendc"
    assert metadata["plan"]["backend"] == "ascendc"
    assert metadata["server_command"] == "bash run.sh"


def test_ascendc_cli_scaffold_json(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "run",
            "examples/elementwise_mul/mul.cu",
            "--spec",
            "examples/elementwise_mul/geyi.yaml",
            "--backend",
            "ascendc",
            "--target",
            "scaffold",
            "--out",
            str(tmp_path / "out"),
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

    payload = json.loads(completed.stdout)
    assert payload["backend"] == "ascendc"
    assert payload["level"] == "compiles_only"
    assert payload["passed"] is True
    assert payload["coverage"]["hardware"] == ["not_run"]


def test_ascendc_scaffold_supports_transpose_project(tmp_path):
    result = run_phase1(
        "examples/transpose2d/transpose.cu",
        spec="examples/transpose2d/geyi.yaml",
        backend="ascendc",
        target="scaffold",
        out=str(tmp_path / "out" / "transpose"),
        session_root=str(tmp_path / "sessions"),
    )

    assert result.plan.template == "ascendc.transpose2d"
    assert {"rows": 31, "cols": 65} in result.verification_report.coverage.shapes
    assert result.verification_report.coverage.strides == [[0, 1]]
    kernel = (result.analysis.session.path / "generated" / "op_kernel" / "transpose2d_kernel.asc").read_text(encoding="utf-8")
    assert "outIdx = col * rows + row" in kernel


def test_ascendc_copy_uses_vector_noop_instead_of_local_datacopy(tmp_path):
    result = run_phase1(
        "examples/copy1d/copy.cu",
        spec="examples/copy1d/geyi.yaml",
        backend="ascendc",
        target="scaffold",
        out=str(tmp_path / "out" / "copy"),
        session_root=str(tmp_path / "sessions"),
    )

    assert result.plan.template == "ascendc.copy_cast_1d"
    kernel = (result.analysis.session.path / "generated" / "op_kernel" / "copy1d_kernel.asc").read_text(encoding="utf-8")
    assert "AscendC::Adds(outputLocal, input0Local, static_cast<float>(0), count)" in kernel
    assert "AscendC::DataCopy(outputLocal, input0Local, count)" not in kernel
