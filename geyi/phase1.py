"""Phase 1 deterministic end-to-end translation pipeline."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from geyi.analysis import AnalysisResult, analyze
from geyi.backend.model import CompiledArtifact, GeneratedProject
from geyi.backend.ascendc import AscendCBackend
from geyi.backend.tilelang import TileLangBackend
from geyi.config import DEFAULT_SESSION_ROOT
from geyi.planner.plan import TranslationPlan, create_deterministic_plan
from geyi.session import SessionStore
from geyi.verifier.ascendc import verify_ascendc
from geyi.verifier.golden import verify_with_golden
from geyi.verifier.report import VerificationReport


@dataclass
class Phase1RunResult:
    analysis: AnalysisResult
    plan: TranslationPlan
    project: GeneratedProject
    artifact: CompiledArtifact
    verification_report: VerificationReport
    out_path: Path
    cache_hit: bool


def run_phase1(
    source: str,
    spec: str,
    out: Optional[str] = None,
    session_root: str = DEFAULT_SESSION_ROOT,
    reproducible_command: Optional[str] = None,
    backend: str = "tilelang",
    target: str = "local_cpu",
    npu_arch: str = "dav-2201",
) -> Phase1RunResult:
    analysis = analyze(source, spec=spec, session_root=session_root, write_session=True)
    if analysis.session is None:
        raise RuntimeError("Phase 1 requires session artifacts")

    session = analysis.session
    contract = analysis.contract
    plan = create_deterministic_plan(contract, backend=backend, target=target, npu_arch=npu_arch)
    session.write_json("plan.json", plan.to_dict())

    out_path = Path(out) if out else Path(".geyi") / "out" / contract.entry
    backend_impl = create_backend(plan.backend)
    cache_hit = plan.backend == "tilelang" and can_reuse_out(out_path, contract.contract_hash, plan.backend)

    if cache_hit:
        copy_tree(out_path / "generated", session.path / "generated")
        copy_tree(out_path / "build", session.path / "build")
        project = backend_impl.load_project(session.path / "generated")
        artifact = backend_impl.load_artifact(session.path / "build", reused=True)
    else:
        project = backend_impl.generate(contract, plan, session.path / "generated")
        if plan.backend == "ascendc":
            artifact = backend_impl.compile(project, session.path / "build", target=str(plan.parameters["target"]))
        else:
            artifact = backend_impl.compile(project, session.path / "build")

    if plan.backend == "ascendc":
        report = verify_ascendc(
            contract,
            plan,
            project,
            artifact,
            reproducible_command=reproducible_command,
            cache_hit=cache_hit,
        )
    else:
        report = verify_with_golden(
            contract,
            plan,
            project,
            artifact,
            reproducible_command=reproducible_command,
            cache_hit=cache_hit,
        )
    session.write_json("verification_report.json", report.to_dict())
    session.write_log("run.log", render_run_log(contract.entry, out_path, cache_hit, report, plan))
    mirror_to_out(session, out_path, contract.contract_hash, artifact.artifact_hash, cache_hit, plan)

    return Phase1RunResult(
        analysis=analysis,
        plan=plan,
        project=project,
        artifact=artifact,
        verification_report=report,
        out_path=out_path,
        cache_hit=cache_hit,
    )


def create_backend(name: str):
    if name == "ascendc":
        return AscendCBackend()
    if name == "tilelang":
        return TileLangBackend()
    raise RuntimeError("unsupported backend: %s" % name)


def can_reuse_out(out_path: Path, contract_hash: str, backend: str) -> bool:
    manifest = out_path / "cache_manifest.json"
    generated = out_path / "generated" / "metadata.json"
    artifact = out_path / "build" / "artifact_metadata.json"
    if not manifest.exists() or not generated.exists() or not artifact.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("contract_hash") != contract_hash:
        return False
    if payload.get("backend") != backend:
        return False
    if payload.get("python_cache_tag") != sys.implementation.cache_tag:
        return False
    if payload.get("python_magic_number") != importlib.util.MAGIC_NUMBER.hex():
        return False
    try:
        artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    metadata = artifact_payload.get("metadata") or {}
    return (
        artifact_payload.get("compiler") == "py_compile"
        and metadata.get("python_cache_tag") == sys.implementation.cache_tag
        and metadata.get("python_magic_number") == importlib.util.MAGIC_NUMBER.hex()
    )


def mirror_to_out(
    session: SessionStore,
    out_path: Path,
    contract_hash: str,
    artifact_hash: str,
    cache_hit: bool,
    plan: TranslationPlan,
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    for name in ["contract.json", "plan.json", "verification_report.json"]:
        shutil.copy2(str(session.path / name), str(out_path / name))
    copy_tree(session.path / "generated", out_path / "generated")
    copy_tree(session.path / "build", out_path / "build")
    write_json(
        out_path / "cache_manifest.json",
        {
            "phase": "phase1",
            "contract_hash": contract_hash,
            "artifact_hash": artifact_hash,
            "backend": plan.backend,
            "target": plan.parameters.get("target"),
            "cache_hit_for_this_run": cache_hit,
            "python_cache_tag": sys.implementation.cache_tag,
            "python_magic_number": importlib.util.MAGIC_NUMBER.hex(),
            "session_id": session.session_id,
        },
    )


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_run_log(entry: str, out_path: Path, cache_hit: bool, report: VerificationReport, plan: TranslationPlan) -> str:
    return "\n".join(
        [
            "Phase 1 deterministic run completed for %s" % entry,
            "backend=%s" % plan.backend,
            "target=%s" % plan.parameters.get("target"),
            "out=%s" % out_path,
            "cache_hit=%s" % cache_hit,
            "level=%s" % report.level.value,
            "passed=%s" % report.passed,
            "artifact_hash=%s" % report.artifact_hash,
        ]
    )
