from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from geyi.analysis import analyze


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "contracts"


class PhaseMinusOneContractTests(unittest.TestCase):
    def analyze_fixture(self, name: str, source_name: str):
        fixture = FIXTURES / name
        return analyze(
            str(fixture / source_name),
            spec=str(fixture / "geyi.yaml"),
            write_session=False,
        )

    def test_vector_add_1d_contract_and_artifacts(self):
        fixture = FIXTURES / "vector_add_1d"
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze(
                str(fixture / "vector_add.cu"),
                spec=str(fixture / "geyi.yaml"),
                session_root=str(Path(tmp) / "sessions"),
                write_session=True,
            )
            session_path = result.session.path
            self.assertTrue((session_path / "input" / "source_snapshot" / "vector_add.cu").exists())
            self.assertTrue((session_path / "input" / "geyi.yaml").exists())
            self.assertTrue((session_path / "evidence" / "scanner.json").exists())
            self.assertTrue((session_path / "contract.json").exists())
            self.assertTrue((session_path / "confidence_report.json").exists())
            self.assertTrue((session_path / "logs").exists())

        contract = result.contract
        self.assertEqual(contract.entry, "vector_add")
        self.assertGreaterEqual(contract.confidence, 0.95)
        self.assertEqual(contract.recommended_path, "rule")
        self.assertEqual(contract.unknowns, [])
        self.assertEqual(contract.rejections, [])
        self.assertEqual(set(contract.tensors.keys()), {"a", "b", "out"})
        self.assertEqual(contract.intents[0].kind, "elementwise")
        self.assertEqual(contract.intents[0].subkind, "add")
        self.assertEqual(contract.effects[0].kind, "pure_store")
        self.assertEqual(contract.control_flow[0].kind, "guarded_store")
        spaces = {space.name: space.space for space in contract.memory_spaces}
        self.assertEqual(spaces, {"a": "global", "b": "global", "out": "global"})

    def test_vector_mul_1d_routes_to_rule(self):
        result = self.analyze_fixture("vector_mul_1d", "vector_mul.cu")
        self.assertGreaterEqual(result.contract.confidence, 0.95)
        self.assertEqual(result.contract.recommended_path, "rule")
        self.assertEqual(result.contract.intents[0].subkind, "mul")

    def test_missing_shape_caps_confidence(self):
        result = self.analyze_fixture("missing_shape", "vector_add.cu")
        self.assertLessEqual(result.contract.confidence, 0.55)
        self.assertTrue(any(item.id.startswith("missing_shape") for item in result.contract.unknowns))

    def test_missing_launch_caps_confidence(self):
        result = self.analyze_fixture("missing_launch", "vector_add.cu")
        self.assertLessEqual(result.contract.confidence, 0.70)
        self.assertTrue(any(item.id == "missing_launch" for item in result.contract.unknowns))

    def test_inline_ptx_is_hard_reject(self):
        result = self.analyze_fixture("inline_ptx", "inline_ptx.cu")
        self.assertTrue(any(item.feature == "inline_ptx" and item.hard for item in result.contract.rejections))
        self.assertLessEqual(result.contract.confidence, 0.10)
        self.assertEqual(result.contract.recommended_path, "human")

    def test_black_box_only_caps_confidence(self):
        result = self.analyze_fixture("black_box_only", "black_box.txt")
        self.assertLessEqual(result.contract.confidence, 0.40)
        self.assertTrue(any(item.id == "black_box_only" for item in result.contract.unknowns))

    def test_cli_info_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "geyi.cli.main",
                    "info",
                    "examples/vector_add/vector_add.cu",
                    "--spec",
                    "examples/vector_add/geyi.yaml",
                    "--json",
                    "--session-root",
                    str(Path(tmp) / "sessions"),
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["entry"], "vector_add")
        self.assertEqual(payload["recommended_path"], "rule")
        self.assertGreaterEqual(payload["confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()
