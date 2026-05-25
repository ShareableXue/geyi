from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from geyi.analysis import analyze
from geyi.phase0 import run_phase0
from geyi.planner.plan import PlanError, create_phase0_plan


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "contracts"


class PhaseZeroEndToEndTests(unittest.TestCase):
    def test_vector_add_run_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run_phase0(
                "examples/vector_add/vector_add.cu",
                spec="examples/vector_add/geyi.yaml",
                out=str(tmp_path / "out" / "vector_add"),
                session_root=str(tmp_path / "sessions"),
                reproducible_command="geyi run examples/vector_add/vector_add.cu --spec examples/vector_add/geyi.yaml",
            )

            session_path = result.analysis.session.path
            self.assertTrue((session_path / "contract.json").exists())
            self.assertTrue((session_path / "plan.json").exists())
            self.assertTrue((session_path / "generated" / "tilelang_vector_add.py").exists())
            self.assertTrue((session_path / "generated" / "metadata.json").exists())
            self.assertTrue((session_path / "build" / "tilelang_vector_add.pyc").exists())
            self.assertTrue((session_path / "build" / "artifact_metadata.json").exists())
            self.assertTrue((session_path / "verification_report.json").exists())

            report = result.verification_report
            self.assertEqual(report.level.value, "golden")
            self.assertTrue(report.passed)
            self.assertEqual(report.contract_hash, result.analysis.contract.contract_hash)
            self.assertTrue(report.artifact_hash)
            self.assertIn({"n": 1025}, report.coverage.shapes)
            self.assertEqual(report.coverage.dtypes, ["float32"])
            self.assertEqual(report.max_abs_diff, 0.0)

            out_path = tmp_path / "out" / "vector_add"
            self.assertTrue((out_path / "cache_manifest.json").exists())
            self.assertTrue((out_path / "generated" / "metadata.json").exists())
            self.assertTrue((out_path / "build" / "tilelang_vector_add.pyc").exists())

    def test_vector_add_run_reuses_out_artifact_on_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / "out" / "vector_add"
            first = run_phase0(
                "examples/vector_add/vector_add.cu",
                spec="examples/vector_add/geyi.yaml",
                out=str(out),
                session_root=str(tmp_path / "sessions"),
            )
            second = run_phase0(
                "examples/vector_add/vector_add.cu",
                spec="examples/vector_add/geyi.yaml",
                out=str(out),
                session_root=str(tmp_path / "sessions"),
            )

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertTrue(second.artifact.reused)
            self.assertEqual(first.artifact.artifact_hash, second.artifact.artifact_hash)
            self.assertEqual(second.verification_report.cache["hit"], True)

    def test_vector_add_run_rebuilds_incompatible_bytecode_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / "out" / "vector_add"
            first = run_phase0(
                "examples/vector_add/vector_add.cu",
                spec="examples/vector_add/geyi.yaml",
                out=str(out),
                session_root=str(tmp_path / "sessions"),
            )

            manifest_path = out / "cache_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["python_cache_tag"] = "cpython-stale"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            second = run_phase0(
                "examples/vector_add/vector_add.cu",
                spec="examples/vector_add/geyi.yaml",
                out=str(out),
                session_root=str(tmp_path / "sessions"),
            )

            self.assertFalse(second.cache_hit)
            self.assertFalse(second.artifact.reused)
            self.assertEqual(first.artifact.artifact_hash, second.artifact.artifact_hash)
            self.assertTrue(second.verification_report.passed)

    def test_cli_run_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "geyi.cli.main",
                    "run",
                    "examples/vector_add/vector_add.cu",
                    "--spec",
                    "examples/vector_add/geyi.yaml",
                    "--out",
                    str(tmp_path / "out" / "vector_add"),
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
        self.assertEqual(payload["level"], "golden")
        self.assertEqual(payload["strategy"], "rule")
        self.assertEqual(payload["backend"], "tilelang")
        self.assertIn({"n": 1025}, payload["coverage"]["shapes"])
        self.assertEqual(payload["max_abs_diff"], 0.0)
        self.assertFalse(payload["llm_used"])

    def test_phase0_rejects_phase_minus_one_mul_rule(self):
        result = analyze(
            str(FIXTURES / "vector_mul_1d" / "vector_mul.cu"),
            spec=str(FIXTURES / "vector_mul_1d" / "geyi.yaml"),
            write_session=False,
        )
        with self.assertRaises(PlanError):
            create_phase0_plan(result.contract)


if __name__ == "__main__":
    unittest.main()
