"""CANN Knowledge Lake lockfile validation and metadata indexing."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_INDEX_PATH = ".geyi/library/index.json"
DEFAULT_SESSION_ROOT = ".geyi/library/sessions"


class LibraryError(ValueError):
    """Raised when a CANN library lock/index is unsafe or malformed."""


def build_library_index(lock_path: str = "geyi-library.lock", out_path: Optional[str] = DEFAULT_INDEX_PATH) -> Dict[str, Any]:
    lock_file = Path(lock_path)
    lock = load_lock(lock_file)
    allowed_licenses = set(str(item) for item in lock.get("allowed_licenses", []))
    if not allowed_licenses:
        raise LibraryError("lockfile must declare allowed_licenses")

    source_validations = []
    sources_by_id: Dict[str, Dict[str, Any]] = {}
    for source in require_list(lock, "sources"):
        validation = validate_source(lock_file, source, allowed_licenses)
        source_validations.append(validation)
        sources_by_id[validation["id"]] = validation

    entries = []
    for order, raw_entry in enumerate(require_list(lock, "hotset"), start=1):
        entries.append(normalize_hotset_entry(lock_file, raw_entry, sources_by_id, order))

    if len(entries) < 20:
        raise LibraryError("Phase 3 hotset requires at least 20 annotated entries")

    index = {
        "version": int(lock.get("version", 1)),
        "metadata_kind": "cann_library_index",
        "lockfile": str(lock_file),
        "sources": source_validations,
        "hotset": entries,
        "safety": {
            "exact_signature_only": True,
            "fuzzy_matches_are_not_semantic_equivalence": True,
            "external_scripts_executed": False,
        },
    }

    if out_path:
        write_json(Path(out_path), index)
    return index


def load_library_index(path: str = DEFAULT_INDEX_PATH) -> Dict[str, Any]:
    index_path = Path(path)
    if not index_path.exists():
        raise LibraryError("library index does not exist: %s" % index_path)
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LibraryError("library index is not valid JSON: %s" % index_path) from exc


def load_or_build_library_index(
    lock_path: str = "geyi-library.lock",
    index_path: str = DEFAULT_INDEX_PATH,
) -> Dict[str, Any]:
    path = Path(index_path)
    if path.exists():
        return load_library_index(index_path)
    return build_library_index(lock_path=lock_path, out_path=index_path)


def search_library_index(index: Dict[str, Any], op: str) -> List[Dict[str, Any]]:
    query = normalize_key(op)
    if not query:
        raise LibraryError("library search requires a non-empty op")

    results = []
    for entry in index.get("hotset", []):
        keys = [entry.get("op", "")]
        keys.extend(entry.get("aliases", []))
        normalized = [normalize_key(item) for item in keys]
        if query not in normalized:
            continue
        results.append(
            {
                "op": entry["op"],
                "aliases": list(entry.get("aliases", [])),
                "match_type": "exact_op_or_alias",
                "source": entry["source"],
                "source_paths": list(entry["source_paths"]),
                "contract_signature": entry["contract_signature"],
                "evidence": list(entry["evidence"]),
                "rank": entry["rank"],
                "notes": list(entry.get("notes", [])),
            }
        )
    return sorted(results, key=lambda item: (item["rank"], item["op"]))


def load_lock(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise LibraryError("lockfile does not exist: %s" % path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise LibraryError("lockfile must be a YAML mapping: %s" % path)
    if int(payload.get("version", 0)) != 1:
        raise LibraryError("unsupported geyi-library.lock version: %s" % payload.get("version"))
    return payload


def validate_source(lock_file: Path, source: Dict[str, Any], allowed_licenses: set[str]) -> Dict[str, Any]:
    for key in ["id", "path", "revision", "checksum", "license", "license_file"]:
        if key not in source or source[key] in (None, ""):
            raise LibraryError("library source is missing required key: %s" % key)

    license_id = str(source["license"])
    if license_id not in allowed_licenses:
        raise LibraryError("license is not allowed for source %s: %s" % (source["id"], license_id))

    root = resolve_path(lock_file, str(source["path"]))
    if not root.exists() or not root.is_dir():
        raise LibraryError("library source path does not exist: %s" % root)

    license_file = root / str(source["license_file"])
    if not license_file.exists():
        raise LibraryError("license file does not exist for source %s: %s" % (source["id"], license_file))

    checksum = source["checksum"]
    if not isinstance(checksum, dict):
        raise LibraryError("source %s checksum must be a mapping" % source["id"])
    checksum_file = root / str(checksum.get("file") or source["license_file"])
    expected_checksum = normalize_checksum(str(checksum.get("value", "")))
    actual_checksum = sha256_file(checksum_file)
    if actual_checksum != expected_checksum:
        raise LibraryError(
            "checksum mismatch for source %s file %s: expected %s actual %s"
            % (source["id"], checksum_file, expected_checksum, actual_checksum)
        )

    actual_revision = read_git_revision(root)
    locked_revision = str(source["revision"])
    if actual_revision and actual_revision != locked_revision:
        raise LibraryError(
            "revision mismatch for source %s: expected %s actual %s"
            % (source["id"], locked_revision, actual_revision)
        )

    return {
        "id": str(source["id"]),
        "kind": str(source.get("kind") or "local_git"),
        "path": str(root),
        "revision": locked_revision,
        "actual_revision": actual_revision or locked_revision,
        "checksum": {
            "algorithm": "sha256",
            "file": str(checksum_file),
            "value": actual_checksum,
        },
        "license": license_id,
        "license_file": str(license_file),
        "status": "validated",
    }


def normalize_hotset_entry(
    lock_file: Path,
    raw_entry: Dict[str, Any],
    sources_by_id: Dict[str, Dict[str, Any]],
    order: int,
) -> Dict[str, Any]:
    for key in ["op", "source", "paths", "contract_signature"]:
        if key not in raw_entry or raw_entry[key] in (None, ""):
            raise LibraryError("hotset entry is missing required key: %s" % key)

    source_id = str(raw_entry["source"])
    source = sources_by_id.get(source_id)
    if source is None:
        raise LibraryError("hotset entry %s references unknown source: %s" % (raw_entry["op"], source_id))

    source_root = Path(source["path"])
    source_paths = []
    evidence = []
    for relpath in require_list(raw_entry, "paths"):
        path = source_root / str(relpath)
        if not path.exists():
            raise LibraryError("hotset entry %s source path does not exist: %s" % (raw_entry["op"], path))
        source_paths.append(str(path))
        evidence.append(
            {
                "kind": "library",
                "source_id": source_id,
                "source_revision": source["revision"],
                "path": str(path),
                "license": source["license"],
                "checksum": source["checksum"]["value"],
                "claim": "CANN hotset metadata entry for %s" % raw_entry["op"],
            }
        )

    signature = raw_entry["contract_signature"]
    if not isinstance(signature, dict):
        raise LibraryError("hotset entry %s contract_signature must be a mapping" % raw_entry["op"])
    if "intent" not in signature:
        raise LibraryError("hotset entry %s contract_signature requires intent" % raw_entry["op"])

    return {
        "op": str(raw_entry["op"]),
        "aliases": [str(item) for item in raw_entry.get("aliases", [])],
        "family": str(raw_entry.get("family") or "unknown"),
        "source": source_id,
        "source_paths": source_paths,
        "contract_signature": signature,
        "evidence": evidence,
        "rank": int(raw_entry.get("rank") or order),
        "notes": [str(item) for item in raw_entry.get("notes", [])],
    }


def require_list(payload: Dict[str, Any], key: str) -> List[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise LibraryError("lockfile key %s must be a non-empty list" % key)
    return value


def resolve_path(lock_file: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = lock_file.parent / path
    return path.resolve()


def read_git_revision(root: Path) -> Optional[str]:
    if not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LibraryError("could not read git revision for source %s: %s" % (root, exc)) from exc
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise LibraryError("checksum file does not exist: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_checksum(value: str) -> str:
    value = value.strip()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise LibraryError("checksum value must be a sha256 hex digest")
    return value.lower()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    values = []
    for item in items:
        if item not in seen:
            values.append(item)
            seen.add(item)
    return values
