"""Phase 4 model-level harness for source-available PyTorch extensions."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import runpy
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from geyi.config import DEFAULT_SESSION_ROOT
from geyi.contract.model import to_jsonable
from geyi.session import SessionStore


DEFAULT_MODEL_HARNESS_ROOT = ".geyi/model_harness"


@dataclass
class InlineHarnessArtifact:
    name: str
    entry: str
    functions: List[str]
    cache_key: str
    cache_hit: bool
    root: str
    spec_path: str
    manifest_path: str
    source_files: List[str]
    status: str = "source_available"
    fallback: str = "stub_module"
    message: str = "captured torch.utils.cpp_extension.load_inline sources"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class BlackBoxBoundary:
    path: str
    api: str
    status: str = "black_box_unsupported"
    reason: str = "compiled extension has no source available; Geyi will not decompile .so files"
    suggestion: str = "provide load_inline sources, source files in geyi.yaml, or use library recall"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class Phase4Report:
    script: str
    cache_root: str
    session: str
    status: str
    captured_ops: List[InlineHarnessArtifact] = field(default_factory=list)
    black_box_extensions: List[BlackBoxBoundary] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    exit_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class PatchedPythonResult:
    report: Phase4Report
    session: SessionStore


class Phase4PatchContext:
    """Patch only the current Python process and restore everything on exit."""

    def __init__(
        self,
        cache_root: str = DEFAULT_MODEL_HARNESS_ROOT,
        execute_original: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.execute_original = execute_original
        self.captured_ops: List[InlineHarnessArtifact] = []
        self.black_box_extensions: List[BlackBoxBoundary] = []
        self.failures: List[Dict[str, Any]] = []
        self._torch_module = None
        self._created_fake_torch = False
        self._originals: List[tuple[Any, str, Any, bool]] = []

    def __enter__(self) -> "Phase4PatchContext":
        torch = self._load_or_create_torch()
        self._torch_module = torch
        cpp_extension = resolve_cpp_extension_module(torch)
        self._patch_attr(cpp_extension, "load_inline", self._load_inline)
        ops = ensure_attr_path(torch, ["ops"])
        ops_original = getattr(ops, "load_library", None)
        self._patch_attr(ops, "load_library", self._load_black_box_library("torch.ops.load_library", ops_original))
        classes = ensure_attr_path(torch, ["classes"])
        classes_original = getattr(classes, "load_library", None)
        self._patch_attr(
            classes,
            "load_library",
            self._load_black_box_library("torch.classes.load_library", classes_original),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for target, attr, original, existed in reversed(self._originals):
            if existed:
                setattr(target, attr, original)
            elif hasattr(target, attr):
                delattr(target, attr)
        if self._created_fake_torch:
            for name in [
                "torch.utils.cpp_extension",
                "torch.utils",
                "torch.ops",
                "torch.classes",
                "torch",
            ]:
                module = sys.modules.get(name)
                if module is not None and getattr(module, "__geyi_fake__", False):
                    sys.modules.pop(name, None)

    def _load_or_create_torch(self):
        if "torch" in sys.modules:
            return sys.modules["torch"]
        if not self.execute_original and os.environ.get("GEYI_PHASE4_USE_REAL_TORCH") != "1":
            self._created_fake_torch = True
            return create_fake_torch()
        try:
            import torch  # type: ignore

            return torch
        except Exception:
            self._created_fake_torch = True
            return create_fake_torch()

    def _patch_attr(self, target: Any, attr: str, replacement: Any) -> None:
        existed = hasattr(target, attr)
        original = getattr(target, attr, None)
        self._originals.append((target, attr, original, existed))
        setattr(target, attr, replacement)

    def _load_inline(self, *args, **kwargs):
        try:
            artifact = self.capture_load_inline(*args, **kwargs)
        except Exception as exc:  # keep user script in control after recording failure
            self.failures.append(
                {
                    "api": "torch.utils.cpp_extension.load_inline",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )
            if self.execute_original:
                original = self._original_for("load_inline")
                if original is not None:
                    return original(*args, **kwargs)
            name = str(kwargs.get("name") or (args[0] if args else "geyi_inline_extension"))
            return InlineStubModule(name=name, functions=[])

        if self.execute_original:
            original = self._original_for("load_inline")
            if original is not None:
                try:
                    return original(*args, **kwargs)
                except Exception as exc:
                    self.failures.append(
                        {
                            "api": "torch.utils.cpp_extension.load_inline.original",
                            "name": artifact.name,
                            "error": "%s: %s" % (type(exc).__name__, exc),
                            "fallback": "stub_module",
                        }
                    )
        return InlineStubModule(name=artifact.name, functions=artifact.functions)

    def _original_for(self, attr: str):
        for _target, original_attr, original, existed in reversed(self._originals):
            if original_attr == attr and existed:
                return original
        return None

    def _load_black_box_library(self, api: str, original=None):
        def wrapper(path: str, *args, **kwargs):
            boundary = BlackBoxBoundary(path=str(path), api=api)
            self.black_box_extensions.append(boundary)
            if self.execute_original and original is not None:
                return original(path, *args, **kwargs)
            return None

        return wrapper

    def capture_load_inline(self, *args, **kwargs) -> InlineHarnessArtifact:
        request = normalize_load_inline_request(args, kwargs)
        cache_key = stable_hash(request)
        safe_name = safe_identifier(str(request["name"]))
        op_root = self.cache_root / ("%s-%s" % (safe_name, cache_key[:12]))
        cache_manifest = op_root / "cache_manifest.json"
        cache_hit = manifest_matches(cache_manifest, cache_key)

        if not cache_hit:
            write_inline_harness(op_root, request, cache_key)

        spec_path = op_root / "geyi.yaml"
        harness_path = op_root / "harness.json"
        source_files = list((json_load(harness_path).get("source_files") if harness_path.exists() else []) or [])
        functions = list(request["functions"])
        entry = functions[0] if functions else safe_name
        artifact = InlineHarnessArtifact(
            name=str(request["name"]),
            entry=entry,
            functions=functions,
            cache_key=cache_key,
            cache_hit=cache_hit,
            root=str(op_root),
            spec_path=str(spec_path),
            manifest_path=str(cache_manifest),
            source_files=source_files,
        )
        self.captured_ops.append(artifact)
        return artifact


class InlineStubModule:
    """Small callable module substitute for capture-only patch mode."""

    def __init__(self, name: str, functions: Iterable[str]) -> None:
        self.__name__ = name
        self.__geyi_harness__ = True
        for function in functions:
            setattr(self, str(function), self._make_stub(str(function)))

    def _make_stub(self, function: str):
        def stub(*args, **kwargs):
            if len(args) >= 2:
                try:
                    return args[0] + args[1]
                except Exception:
                    return args[0]
            if args:
                return args[0]
            return {"geyi_stub": function, "kwargs": kwargs}

        stub.__name__ = function
        return stub


def run_patched_python(
    script: str,
    script_args: Optional[List[str]] = None,
    cache_root: str = DEFAULT_MODEL_HARNESS_ROOT,
    session_root: str = DEFAULT_SESSION_ROOT,
    execute_original: bool = False,
) -> PatchedPythonResult:
    session = SessionStore.create(session_root)
    script_path = Path(script)
    status = "completed"
    exit_code = 0
    failures: List[Dict[str, Any]] = []

    original_argv = list(sys.argv)
    original_path = list(sys.path)
    sys.argv = [str(script_path)] + list(script_args or [])
    sys.path.insert(0, str(script_path.resolve().parent))

    with Phase4PatchContext(cache_root=cache_root, execute_original=execute_original) as patch:
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            exit_code = int(code)
            if exit_code != 0:
                status = "script_failed"
                failures.append({"api": "python", "error": "SystemExit(%s)" % exc.code})
        except Exception as exc:
            status = "script_failed"
            exit_code = 1
            failures.append({"api": "python", "error": "%s: %s" % (type(exc).__name__, exc)})
        finally:
            sys.argv = original_argv
            sys.path[:] = original_path

        all_failures = failures + patch.failures
        if all_failures and status == "completed":
            status = "completed_with_downgrade"
        report = Phase4Report(
            script=str(script_path),
            cache_root=str(cache_root),
            session=str(session.path),
            status=status,
            captured_ops=list(patch.captured_ops),
            black_box_extensions=list(patch.black_box_extensions),
            failures=all_failures,
            exit_code=exit_code,
        )

    session.write_json("model_harness_report.json", report.to_dict())
    session.write_log("patch.log", render_patch_log(report))
    return PatchedPythonResult(report=report, session=session)


def normalize_load_inline_request(args, kwargs) -> Dict[str, Any]:
    names = [
        "name",
        "cpp_sources",
        "cuda_sources",
        "functions",
        "extra_cflags",
        "extra_cuda_cflags",
        "extra_ldflags",
        "extra_include_paths",
        "with_cuda",
        "is_python_module",
        "with_pytorch_error_handling",
        "keep_intermediates",
        "build_directory",
    ]
    request = {key: kwargs.get(key) for key in names if key in kwargs}
    for index, value in enumerate(args[:4]):
        request.setdefault(names[index], value)
    request.setdefault("name", "geyi_inline_extension")
    request["cpp_sources"] = normalize_sources(request.get("cpp_sources"))
    request["cuda_sources"] = normalize_sources(request.get("cuda_sources"))
    request["functions"] = normalize_functions(request.get("functions"))
    request["build_options"] = {
        key: safe_jsonable(request.pop(key))
        for key in [
            "extra_cflags",
            "extra_cuda_cflags",
            "extra_ldflags",
            "extra_include_paths",
            "with_cuda",
            "is_python_module",
            "with_pytorch_error_handling",
            "keep_intermediates",
            "build_directory",
        ]
        if key in request
    }
    return safe_jsonable(request)


def normalize_sources(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def normalize_functions(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(key) for key in value.keys()]
    return [str(item) for item in value]


def write_inline_harness(root: Path, request: Dict[str, Any], cache_key: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sources_dir = root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    source_files = []
    safe_name = safe_identifier(str(request["name"]))
    for index, source in enumerate(request["cpp_sources"]):
        relative = Path("sources") / ("%s_cpp_%d.cpp" % (safe_name, index))
        (root / relative).write_text(source, encoding="utf-8")
        source_files.append(str(relative))
    for index, source in enumerate(request["cuda_sources"]):
        relative = Path("sources") / ("%s_cuda_%d.cu" % (safe_name, index))
        (root / relative).write_text(source, encoding="utf-8")
        source_files.append(str(relative))

    functions = list(request["functions"])
    entry = functions[0] if functions else safe_name
    spec = {
        "version": 1,
        "source": {
            "entry": entry,
            "files": source_files,
            "inline_extension": {
                "api": "torch.utils.cpp_extension.load_inline",
                "name": request["name"],
                "functions": functions,
                "cache_key": cache_key,
                "source_available": True,
                "build_options": request.get("build_options", {}),
            },
        },
        "assumptions": [
            {
                "id": "source_available_load_inline",
                "text": "PyTorch load_inline supplied source text; no compiled .so reverse engineering is used.",
                "required_for": ["correctness"],
                "source": "phase4_patch",
            }
        ],
        "verification": {
            "reference": "python_harness",
            "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        },
    }
    write_yaml(root / "geyi.yaml", spec)
    write_json(
        root / "harness.json",
        {
            "phase": "phase4",
            "kind": "source_available_load_inline",
            "api": "torch.utils.cpp_extension.load_inline",
            "name": request["name"],
            "entry": entry,
            "functions": functions,
            "cache_key": cache_key,
            "source_files": source_files,
            "build_options": request.get("build_options", {}),
            "boundary": "source capture only; translation remains op-level",
        },
    )
    write_json(
        root / "cache_manifest.json",
        {
            "phase": "phase4",
            "cache_key": cache_key,
            "name": request["name"],
            "functions": functions,
            "source_files": source_files,
            "reusable": True,
        },
    )


def create_fake_torch():
    torch = types.ModuleType("torch")
    torch.__geyi_fake__ = True
    utils = types.ModuleType("torch.utils")
    utils.__geyi_fake__ = True
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.__geyi_fake__ = True
    ops = types.ModuleType("torch.ops")
    ops.__geyi_fake__ = True
    classes = types.ModuleType("torch.classes")
    classes.__geyi_fake__ = True

    utils.cpp_extension = cpp_extension
    torch.utils = utils
    torch.ops = ops
    torch.classes = classes

    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.cpp_extension"] = cpp_extension
    sys.modules["torch.ops"] = ops
    sys.modules["torch.classes"] = classes
    return torch


def resolve_cpp_extension_module(torch) -> Any:
    try:
        return importlib.import_module("torch.utils.cpp_extension")
    except Exception:
        module = ensure_attr_path(torch, ["utils", "cpp_extension"])
        if isinstance(module, types.ModuleType):
            sys.modules.setdefault("torch.utils.cpp_extension", module)
        return module


def ensure_attr_path(root: Any, path: List[str]) -> Any:
    target = root
    module_name = getattr(root, "__name__", "torch")
    for item in path:
        if not hasattr(target, item):
            namespace = types.SimpleNamespace()
            setattr(target, item, namespace)
        target = getattr(target, item)
        module_name = "%s.%s" % (module_name, item)
    return target


def manifest_matches(path: Path, cache_key: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json_load(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("cache_key") == cache_key


def stable_hash(payload: Dict[str, Any]) -> str:
    data = json.dumps(safe_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def safe_jsonable(value):
    if isinstance(value, dict):
        return {str(key): safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def safe_identifier(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char == "_":
            cleaned.append(char)
        else:
            cleaned.append("_")
    candidate = "".join(cleaned).strip("_") or "geyi_inline_extension"
    if candidate[0].isdigit():
        candidate = "op_%s" % candidate
    return candidate


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def render_patch_log(report: Phase4Report) -> str:
    lines = [
        "Phase 4 model harness patch completed",
        "script=%s" % report.script,
        "status=%s" % report.status,
        "exit_code=%s" % report.exit_code,
        "captured_ops=%d" % len(report.captured_ops),
        "black_box_extensions=%d" % len(report.black_box_extensions),
    ]
    for op in report.captured_ops:
        lines.append(
            "- load_inline name=%s entry=%s cache_hit=%s spec=%s"
            % (op.name, op.entry, op.cache_hit, op.spec_path)
        )
    for boundary in report.black_box_extensions:
        lines.append("- black_box api=%s path=%s status=%s" % (boundary.api, boundary.path, boundary.status))
    for failure in report.failures:
        lines.append("- failure api=%s error=%s" % (failure.get("api"), failure.get("error")))
    return "\n".join(lines)
