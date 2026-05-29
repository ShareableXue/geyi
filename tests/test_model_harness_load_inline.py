from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import yaml

from geyi.phase4 import Phase4PatchContext, run_patched_python


ROOT = Path(__file__).resolve().parents[1]


def write_inline_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "from torch.utils.cpp_extension import load_inline",
                "ext = load_inline(",
                "    name='inline_add',",
                "    cpp_sources='torch::Tensor inline_add(torch::Tensor a, torch::Tensor b);',",
                "    cuda_sources='__global__ void inline_add_kernel(float* out) {}',",
                "    functions=['inline_add'],",
                "    extra_cuda_cflags=['-O2'],",
                ")",
                "assert ext.inline_add(1, 2) == 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_inline_capture_generates_source_available_harness(tmp_path):
    script = tmp_path / "run_inline.py"
    write_inline_script(script)

    result = run_patched_python(
        str(script),
        cache_root=str(tmp_path / "cache"),
        session_root=str(tmp_path / "sessions"),
    )

    report = result.report
    assert report.status == "completed"
    assert report.exit_code == 0
    assert len(report.captured_ops) == 1
    op = report.captured_ops[0]
    assert op.name == "inline_add"
    assert op.entry == "inline_add"
    assert op.cache_hit is False
    assert Path(op.spec_path).exists()
    assert Path(op.manifest_path).exists()
    assert sorted(Path(op.root, item).suffix for item in op.source_files) == [".cpp", ".cu"]

    spec = yaml.safe_load(Path(op.spec_path).read_text(encoding="utf-8"))
    assert spec["source"]["entry"] == "inline_add"
    assert spec["source"]["inline_extension"]["source_available"] is True
    assert spec["verification"]["reference"] == "python_harness"
    assert (result.session.path / "model_harness_report.json").exists()


def test_load_inline_op_level_cache_reuses_existing_harness(tmp_path):
    script = tmp_path / "run_inline.py"
    write_inline_script(script)
    cache_root = tmp_path / "cache"

    first = run_patched_python(str(script), cache_root=str(cache_root), session_root=str(tmp_path / "sessions"))
    second = run_patched_python(str(script), cache_root=str(cache_root), session_root=str(tmp_path / "sessions"))

    assert first.report.captured_ops[0].cache_key == second.report.captured_ops[0].cache_key
    assert first.report.captured_ops[0].cache_hit is False
    assert second.report.captured_ops[0].cache_hit is True
    assert first.report.captured_ops[0].root == second.report.captured_ops[0].root


def test_black_box_so_load_is_recorded_as_boundary(tmp_path):
    script = tmp_path / "run_black_box.py"
    script.write_text(
        "import torch\n"
        "torch.ops.load_library('custom_extension.so')\n"
        "torch.classes.load_library('custom_classes.so')\n",
        encoding="utf-8",
    )

    result = run_patched_python(
        str(script),
        cache_root=str(tmp_path / "cache"),
        session_root=str(tmp_path / "sessions"),
    )

    assert result.report.status == "completed"
    assert len(result.report.captured_ops) == 0
    assert [item.status for item in result.report.black_box_extensions] == [
        "black_box_unsupported",
        "black_box_unsupported",
    ]
    assert ".so" in result.report.black_box_extensions[0].path
    assert "will not decompile" in result.report.black_box_extensions[0].reason


def test_patch_context_restores_existing_torch_loader(monkeypatch, tmp_path):
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    ops = types.SimpleNamespace()
    classes = types.SimpleNamespace()

    def original_load_inline(*args, **kwargs):
        return "original"

    def original_load_library(path):
        return "loaded:%s" % path

    cpp_extension.load_inline = original_load_inline
    ops.load_library = original_load_library
    classes.load_library = original_load_library
    utils.cpp_extension = cpp_extension
    torch.utils = utils
    torch.ops = ops
    torch.classes = classes

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)

    with Phase4PatchContext(cache_root=str(tmp_path / "cache")):
        assert cpp_extension.load_inline is not original_load_inline
        assert ops.load_library is not original_load_library

    assert cpp_extension.load_inline is original_load_inline
    assert ops.load_library is original_load_library
    assert classes.load_library is original_load_library


def test_phase4_acceptance_cli_command(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "patch",
            "python",
            "examples/pytorch_load_inline/run.py",
            "--cache-root",
            str(tmp_path / "cache"),
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
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert len(payload["captured_ops"]) == 1
    assert payload["captured_ops"][0]["entry"] == "vector_add"
    assert len(payload["black_box_extensions"]) == 1
    assert Path(payload["captured_ops"][0]["spec_path"]).exists()

