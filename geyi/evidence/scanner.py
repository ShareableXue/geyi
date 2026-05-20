"""Small CUDA source scanner used by the Phase -1 contract prototype."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ScannerRejection:
    feature: str
    reason: str
    hard: bool
    suggestion: str


@dataclass
class KernelScan:
    name: str
    params: List[str]
    body: str
    start_line: int


@dataclass
class ScannerResult:
    source_file: Optional[str]
    entry: str
    source_available: bool
    kernels: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    cuda_features: List[str] = field(default_factory=list)
    rejections: List[ScannerRejection] = field(default_factory=list)
    idx_var: Optional[str] = None
    idx_expression: Optional[str] = None
    guarded: bool = False
    guard_condition: Optional[str] = None
    write_tensor: Optional[str] = None
    write_index: Optional[str] = None
    expression: Optional[str] = None
    read_tensors: List[str] = field(default_factory=list)
    operation: Optional[str] = None
    evidence: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_file": self.source_file,
            "entry": self.entry,
            "source_available": self.source_available,
            "kernels": self.kernels,
            "params": self.params,
            "cuda_features": self.cuda_features,
            "rejections": [rejection.__dict__ for rejection in self.rejections],
            "idx_var": self.idx_var,
            "idx_expression": self.idx_expression,
            "guarded": self.guarded,
            "guard_condition": self.guard_condition,
            "write_tensor": self.write_tensor,
            "write_index": self.write_index,
            "expression": self.expression,
            "read_tensors": self.read_tensors,
            "operation": self.operation,
            "evidence": self.evidence,
        }


def scan_cuda_source(source_path: Optional[str], entry: str = "", black_box: bool = False) -> ScannerResult:
    if black_box or not source_path:
        return ScannerResult(
            source_file=source_path,
            entry=entry,
            source_available=False,
            cuda_features=["black_box"],
            evidence=[{"id": "scan.black_box", "claim": "source is black-box only", "confidence": 1.0}],
        )

    path = Path(source_path)
    if not path.exists():
        return ScannerResult(
            source_file=str(path),
            entry=entry,
            source_available=False,
            evidence=[{"id": "scan.missing_source", "claim": "source file is not available", "confidence": 1.0}],
        )

    text = path.read_text(encoding="utf-8")
    result = ScannerResult(source_file=str(path), entry=entry, source_available=True)
    result.cuda_features = detect_cuda_features(text)
    result.rejections = detect_rejections(text)

    kernels = find_kernels(text)
    result.kernels = [kernel.name for kernel in kernels]
    selected = select_kernel(kernels, entry)
    if selected is None:
        result.evidence.append({"id": "scan.no_kernel", "claim": "no matching __global__ kernel found", "confidence": 1.0})
        return result

    result.entry = selected.name
    result.params = selected.params
    result.evidence.append(
        {
            "id": "scan.kernel",
            "claim": "found __global__ kernel %s" % selected.name,
            "confidence": 0.98,
            "line": selected.start_line,
        }
    )

    extract_index(selected, result)
    extract_store(selected, result)
    classify_operation(result)
    return result


def detect_cuda_features(text: str) -> List[str]:
    features = []
    checks = {
        "inline_ptx": r"(__asm__|\basm\s*(?:volatile)?\s*\()",
        "atomic": r"\batomic[A-Za-z_]*\s*\(",
        "syncthreads": r"__syncthreads\s*\(",
        "syncwarp": r"__syncwarp\s*\(",
        "warp_shuffle": r"__shfl_[A-Za-z_]+\s*\(",
        "device_malloc": r"\b(malloc|free)\s*\(",
        "cooperative_groups": r"cooperative_groups::|namespace\s+cg\s*=\s*cooperative_groups",
    }
    for feature, pattern in checks.items():
        if re.search(pattern, text):
            features.append(feature)
    return features


def detect_rejections(text: str) -> List[ScannerRejection]:
    rejections = []
    if re.search(r"(__asm__|\basm\s*(?:volatile)?\s*\()", text):
        rejections.append(
            ScannerRejection(
                feature="inline_ptx",
                reason="inline PTX is outside Phase -1 rule coverage",
                hard=True,
                suggestion="replace inline PTX with analyzable CUDA or add a later-phase custom annotation",
            )
        )
    if re.search(r"\b(malloc|free)\s*\(", text):
        rejections.append(
            ScannerRejection(
                feature="device_malloc",
                reason="device malloc/free is outside Phase -1 rule coverage",
                hard=True,
                suggestion="provide a simpler source-available fixture without dynamic device allocation",
            )
        )
    return rejections


def find_kernels(text: str) -> List[KernelScan]:
    pattern = re.compile(r"__global__\s+(?:[\w:<>]+\s+)*void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)
    kernels = []
    for match in pattern.finditer(text):
        body_start = match.end() - 1
        body_end = find_matching_brace(text, body_start)
        if body_end is None:
            continue
        body = text[body_start + 1 : body_end]
        params = parse_param_names(match.group(2))
        start_line = text.count("\n", 0, match.start()) + 1
        kernels.append(KernelScan(match.group(1), params, body, start_line))
    return kernels


def find_matching_brace(text: str, start: int) -> Optional[int]:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def parse_param_names(params: str) -> List[str]:
    names = []
    for raw in params.split(","):
        item = raw.strip()
        if not item:
            continue
        item = item.split("=")[0].strip()
        item = re.sub(r"\b(const|volatile|__restrict__|restrict)\b", " ", item)
        item = item.replace("*", " ").replace("&", " ")
        pieces = [piece for piece in item.split() if piece]
        if pieces:
            names.append(pieces[-1])
    return names


def select_kernel(kernels: List[KernelScan], entry: str) -> Optional[KernelScan]:
    if entry:
        for kernel in kernels:
            if kernel.name == entry:
                return kernel
        return None
    if len(kernels) == 1:
        return kernels[0]
    return None


def extract_index(kernel: KernelScan, result: ScannerResult) -> None:
    idx_pattern = re.compile(
        r"(?:int|unsigned\s+int|long|size_t)?\s*(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;"
    )
    match = idx_pattern.search(kernel.body)
    if match:
        result.idx_var = match.group(1)
        result.idx_expression = "blockIdx.x * blockDim.x + threadIdx.x"
        result.evidence.append(
            {
                "id": "scan.idx",
                "claim": "%s is blockIdx.x * blockDim.x + threadIdx.x" % result.idx_var,
                "confidence": 0.99,
            }
        )


def extract_store(kernel: KernelScan, result: ScannerResult) -> None:
    body = kernel.body
    guarded = re.search(r"if\s*\((?P<cond>[^)]*)\)\s*\{?(?P<stmt>[^{};]*\[[^\]]+\]\s*=\s*[^;]+;)", body, re.S)
    statement = None
    if guarded:
        result.guarded = True
        result.guard_condition = compact(guarded.group("cond"))
        statement = guarded.group("stmt")
    else:
        store = re.search(r"(?P<stmt>\b\w+\s*\[[^\]]+\]\s*=\s*[^;]+;)", body, re.S)
        if store:
            statement = store.group("stmt")

    if not statement:
        return

    store_match = re.search(r"\b(?P<out>\w+)\s*\[\s*(?P<idx>[^\]]+)\s*\]\s*=\s*(?P<expr>[^;]+);", statement, re.S)
    if not store_match:
        return

    result.write_tensor = store_match.group("out")
    result.write_index = compact(store_match.group("idx"))
    result.expression = compact(store_match.group("expr"))
    result.evidence.append(
        {
            "id": "scan.store",
            "claim": "single %sstore to %s[%s]" % ("guarded " if result.guarded else "", result.write_tensor, result.write_index),
            "confidence": 0.98,
        }
    )


def classify_operation(result: ScannerResult) -> None:
    if not result.expression or not result.idx_var:
        return
    indexed_reads = re.findall(r"\b(\w+)\s*\[\s*%s\s*\]" % re.escape(result.idx_var), result.expression)
    reads = []
    for name in indexed_reads:
        if name != result.write_tensor and name not in reads:
            reads.append(name)
    result.read_tensors = reads

    escaped_reads = [re.escape(name) + r"\s*\[\s*" + re.escape(result.idx_var) + r"\s*\]" for name in reads]
    if len(escaped_reads) >= 2:
        add_pattern = escaped_reads[0] + r"\s*\+\s*" + escaped_reads[1]
        mul_pattern = escaped_reads[0] + r"\s*\*\s*" + escaped_reads[1]
        if re.search(add_pattern, result.expression):
            result.operation = "add"
        elif re.search(mul_pattern, result.expression):
            result.operation = "mul"
        elif "+" in result.expression:
            result.operation = "add"
        elif "*" in result.expression:
            result.operation = "mul"

    if result.operation:
        result.evidence.append(
            {
                "id": "scan.intent",
                "claim": "recognized 1D elementwise %s" % result.operation,
                "confidence": 0.98,
            }
        )


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

