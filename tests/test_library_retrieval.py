from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from geyi.analysis import analyze
from geyi.library.index import build_library_index, search_library_index
from geyi.library.retrieval import recall_exact_signature


ROOT = Path(__file__).resolve().parents[1]


def test_library_search_returns_source_paths_and_evidence(tmp_path):
    index = build_library_index(str(ROOT / "geyi-library.lock"), out_path=str(tmp_path / "index.json"))
    results = search_library_index(index, "rms_norm")

    assert results
    first = results[0]
    assert first["op"] == "rms_norm"
    assert first["match_type"] == "exact_op_or_alias"
    assert first["source_paths"]
    assert Path(first["source_paths"][0]).exists()
    assert first["evidence"][0]["license"] == "CANN-Open-Software-License-2.0"


def test_strategy0_exact_signature_recalls_vector_add():
    index = build_library_index(str(ROOT / "geyi-library.lock"), out_path=None)
    analysis = analyze(
        "examples/vector_add/vector_add.cu",
        spec="examples/vector_add/geyi.yaml",
        write_session=False,
    )

    results = recall_exact_signature(analysis.contract, index)

    assert results
    assert results[0]["op"] == "add"
    assert results[0]["match_type"] == "exact_signature"
    assert results[0]["strategy"] == "library"


def test_library_acceptance_cli_commands(tmp_path):
    index_path = tmp_path / "index.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "library",
            "index",
            "--lock",
            "geyi-library.lock",
            "--out",
            str(index_path),
            "--session-root",
            str(tmp_path / "library_sessions"),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geyi.cli.main",
            "library",
            "search",
            "--op",
            "rms_norm",
            "--index",
            str(index_path),
            "--json",
            "--session-root",
            str(tmp_path / "library_sessions"),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["match_policy"] == "exact_op_or_alias"
    assert payload["results"][0]["op"] == "rms_norm"
    assert Path(payload["session"]).exists()
