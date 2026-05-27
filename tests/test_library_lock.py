from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from geyi.library.index import LibraryError, build_library_index


ROOT = Path(__file__).resolve().parents[1]


def test_default_library_lock_builds_valid_hotset_index(tmp_path):
    index = build_library_index(str(ROOT / "geyi-library.lock"), out_path=str(tmp_path / "index.json"))

    assert len(index["sources"]) == 4
    assert len(index["hotset"]) >= 20
    assert index["safety"]["exact_signature_only"] is True
    assert index["safety"]["external_scripts_executed"] is False

    for source in index["sources"]:
        assert source["revision"]
        assert source["checksum"]["value"]
        assert source["license"] == "CANN-Open-Software-License-2.0"
        assert source["status"] == "validated"

    for entry in index["hotset"]:
        assert entry["contract_signature"]["intent"]
        assert entry["source_paths"]
        assert entry["evidence"][0]["source_revision"]


def test_checksum_mismatch_fails(tmp_path):
    lock = write_fake_lock(tmp_path, checksum="0" * 64)

    with pytest.raises(LibraryError, match="checksum mismatch"):
        build_library_index(str(lock), out_path=None)


def test_disallowed_license_fails(tmp_path):
    lock = write_fake_lock(tmp_path, license_id="Unknown-License", allowed=["CANN-Open-Software-License-2.0"])

    with pytest.raises(LibraryError, match="license is not allowed"):
        build_library_index(str(lock), out_path=None)


def write_fake_lock(
    tmp_path: Path,
    checksum: str | None = None,
    license_id: str = "CANN-Open-Software-License-2.0",
    allowed: list[str] | None = None,
) -> Path:
    source = tmp_path / "source"
    op_dir = source / "ops" / "rms_norm"
    op_dir.mkdir(parents=True)
    license_text = "CANN Open Software License Agreement Version 2.0\n"
    (source / "LICENSE").write_text(license_text, encoding="utf-8")
    actual = hashlib.sha256(license_text.encode("utf-8")).hexdigest()
    payload = {
        "version": 1,
        "allowed_licenses": allowed or [license_id],
        "sources": [
            {
                "id": "fake-source",
                "kind": "local_directory",
                "path": str(source),
                "revision": "local-test-revision",
                "license": license_id,
                "license_file": "LICENSE",
                "checksum": {"algorithm": "sha256", "file": "LICENSE", "value": checksum or actual},
            }
        ],
        "hotset": [
            {
                "op": "rms_norm",
                "source": "fake-source",
                "paths": ["ops/rms_norm"],
                "contract_signature": {
                    "intent": {"kind": "reduce", "subkind": "rms_norm"},
                    "inputs": [{"name": "x", "rank": 2, "dtypes": ["float32"], "access": "read"}],
                    "outputs": [{"name": "y", "rank": 2, "dtypes": ["float32"], "access": "write"}],
                },
            }
            for _ in range(20)
        ],
    }
    lock = tmp_path / "geyi-library.lock"
    lock.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return lock
