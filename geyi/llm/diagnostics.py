"""Phase 2 precision diagnostics."""

from __future__ import annotations

from typing import Any

from geyi.verifier.report import VerificationReport


def diagnose_precision_mismatch(report: VerificationReport) -> dict[str, Any]:
    failed_cases = [case for case in report.case_results if not case.get("passed", False)]
    if report.passed or not failed_cases:
        return {
            "kind": "precision",
            "status": "no_mismatch",
            "max_abs_diff": report.max_abs_diff,
            "max_rel_diff": report.max_rel_diff,
        }
    return {
        "kind": "precision",
        "status": "mismatch",
        "max_abs_diff": report.max_abs_diff,
        "max_rel_diff": report.max_rel_diff,
        "tolerance": dict(report.tolerance),
        "failed_cases": failed_cases,
        "suggested_resolution": "inspect expression mapping, dtype casts, and tolerance source; rerun repair only after preserving contract assumptions",
    }
