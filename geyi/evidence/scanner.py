"""Small CUDA source scanner used by deterministic Geyi contracts."""

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
    index_vars: Dict[str, str] = field(default_factory=dict)
    rank: int = 1
    dimensions: List[str] = field(default_factory=list)
    guarded: bool = False
    guard_condition: Optional[str] = None
    write_tensor: Optional[str] = None
    write_index: Optional[str] = None
    expression: Optional[str] = None
    read_tensors: List[str] = field(default_factory=list)
    read_indices: Dict[str, List[str]] = field(default_factory=dict)
    operation: Optional[str] = None
    reduction_axis: Optional[str] = None
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
            "index_vars": self.index_vars,
            "rank": self.rank,
            "dimensions": self.dimensions,
            "guarded": self.guarded,
            "guard_condition": self.guard_condition,
            "write_tensor": self.write_tensor,
            "write_index": self.write_index,
            "expression": self.expression,
            "read_tensors": self.read_tensors,
            "read_indices": self.read_indices,
            "operation": self.operation,
            "reduction_axis": self.reduction_axis,
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

    extract_grid_indices(selected, result)
    if not extract_row_reduce(selected, result):
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
                reason="inline PTX is outside deterministic rule coverage",
                hard=True,
                suggestion="replace inline PTX with analyzable CUDA or add a later-phase custom annotation",
            )
        )
    if re.search(r"\b(malloc|free)\s*\(", text):
        rejections.append(
            ScannerRejection(
                feature="device_malloc",
                reason="device malloc/free is outside deterministic rule coverage",
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


def extract_grid_indices(kernel: KernelScan, result: ScannerResult) -> None:
    index_pattern = re.compile(
        r"(?:int|unsigned\s+int|long|size_t)?\s*(\w+)\s*=\s*"
        r"blockIdx\.(x|y)\s*\*\s*blockDim\.\2\s*\+\s*threadIdx\.\2\s*;"
    )
    for match in index_pattern.finditer(kernel.body):
        name = match.group(1)
        axis = match.group(2)
        result.index_vars[axis] = name
        result.evidence.append(
            {
                "id": "scan.idx.%s" % axis,
                "claim": "%s is blockIdx.%s * blockDim.%s + threadIdx.%s" % (name, axis, axis, axis),
                "confidence": 0.99,
            }
        )

    if "y" in result.index_vars and "x" in result.index_vars:
        result.rank = 2
        result.index_vars["row"] = result.index_vars["y"]
        result.index_vars["col"] = result.index_vars["x"]
    elif "x" in result.index_vars:
        result.idx_var = result.index_vars["x"]
        result.idx_expression = "blockIdx.x * blockDim.x + threadIdx.x"
        result.index_vars["linear"] = result.idx_var


def extract_row_reduce(kernel: KernelScan, result: ScannerResult) -> bool:
    reduce_pattern = re.compile(
        r"(?:float|double|int|half)\s+(?P<acc>\w+)\s*=\s*0(?:\.0)?f?\s*;\s*"
        r"for\s*\(\s*(?:int|unsigned\s+int|long|size_t)?\s*(?P<axis>\w+)\s*=\s*0\s*;\s*"
        r"(?P=axis)\s*<\s*(?P<extent>\w+)\s*;\s*(?:\+\+(?P=axis)|(?P=axis)\+\+|(?P=axis)\s*\+=\s*1)\s*\)\s*\{\s*"
        r"(?P=acc)\s*\+=\s*(?P<input>\w+)\s*\[\s*(?P<input_idx>[^\]]+)\s*\]\s*;\s*"
        r"\}\s*(?P<out>\w+)\s*\[\s*(?P<out_idx>[^\]]+)\s*\]\s*=\s*(?P=acc)\s*;",
        re.S,
    )
    match = reduce_pattern.search(kernel.body)
    if not match:
        return False

    result.rank = 2
    result.operation = "row_sum"
    result.reduction_axis = match.group("axis")
    result.write_tensor = match.group("out")
    result.write_index = compact(match.group("out_idx"))
    result.expression = "%s[%s] = sum(%s[%s])" % (
        result.write_tensor,
        result.write_index,
        match.group("input"),
        compact(match.group("input_idx")),
    )
    result.read_tensors = [match.group("input")]
    result.read_indices = {match.group("input"): [compact(match.group("input_idx"))], result.write_tensor: [result.write_index]}
    result.dimensions = infer_reduce_dimensions(result, match.group("extent"))

    guard = re.search(r"if\s*\((?P<cond>[^)]*)\)", kernel.body, re.S)
    if guard:
        result.guarded = True
        result.guard_condition = compact(guard.group("cond"))

    result.evidence.append(
        {
            "id": "scan.store",
            "claim": "single guarded row reduce store to %s[%s]" % (result.write_tensor, result.write_index),
            "confidence": 0.96,
        }
    )
    result.evidence.append(
        {
            "id": "scan.intent",
            "claim": "recognized row-wise contiguous reduce sum",
            "confidence": 0.96,
        }
    )
    return True


def infer_reduce_dimensions(result: ScannerResult, reduction_extent: str) -> List[str]:
    row_var = result.index_vars.get("linear") or result.index_vars.get("x") or result.write_index or "row"
    if result.guard_condition:
        bound = bound_for_var(row_var, result.guard_condition)
        if bound:
            return [bound, reduction_extent]
    return ["rows", reduction_extent]


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
    if result.guard_condition:
        result.dimensions = infer_dimensions_from_guard(result)
    result.evidence.append(
        {
            "id": "scan.store",
            "claim": "single %sstore to %s[%s]" % ("guarded " if result.guarded else "", result.write_tensor, result.write_index),
            "confidence": 0.98,
        }
    )


def classify_operation(result: ScannerResult) -> None:
    if result.operation or not result.expression:
        return

    reads = []
    read_indices: Dict[str, List[str]] = {}
    for name, index in re.findall(r"\b(\w+)\s*\[\s*([^\]]+)\s*\]", result.expression):
        if name == result.write_tensor:
            continue
        if name not in reads:
            reads.append(name)
        read_indices.setdefault(name, []).append(compact(index))
    result.read_tensors = reads
    result.read_indices = read_indices

    if result.rank == 2 and classify_transpose(result):
        return

    idx = result.idx_var
    if not idx:
        return

    expr = compact(result.expression)
    direct_reads = [name for name in reads if all(normalize_index(index) == normalize_index(idx) for index in read_indices.get(name, []))]
    if len(direct_reads) == 1:
        read = direct_reads[0]
        ref = indexed_ref(read, idx)
        if is_cast_expression(expr, read, idx):
            set_operation(result, "cast", "recognized 1D contiguous cast")
            return
        if normalize_expr(expr) == normalize_expr(ref):
            set_operation(result, "copy", "recognized 1D contiguous copy")
            return
        if re.fullmatch(r"-\s*%s" % re.escape(ref), expr):
            set_operation(result, "neg", "recognized 1D elementwise neg")
            return
        if is_relu_expression(expr, read, idx):
            set_operation(result, "relu", "recognized 1D elementwise relu")
            return
        if is_exp_expression(expr, read, idx):
            set_operation(result, "exp", "recognized 1D elementwise exp")
            return

    escaped_reads = [re.escape(indexed_ref(name, idx)) for name in direct_reads]
    if len(escaped_reads) >= 2:
        if re.search(escaped_reads[0] + r"\s*\+\s*" + escaped_reads[1], expr):
            set_operation(result, "add", "recognized 1D elementwise add")
        elif re.search(escaped_reads[0] + r"\s*\*\s*" + escaped_reads[1], expr):
            set_operation(result, "mul", "recognized 1D elementwise mul")
        elif "+" in expr:
            set_operation(result, "add", "recognized 1D elementwise add")
        elif "*" in expr:
            set_operation(result, "mul", "recognized 1D elementwise mul")


def classify_transpose(result: ScannerResult) -> bool:
    if len(result.read_tensors) != 1 or not result.write_index:
        return False
    row = result.index_vars.get("row")
    col = result.index_vars.get("col")
    if not row or not col:
        return False
    input_name = result.read_tensors[0]
    read_indexes = result.read_indices.get(input_name) or []
    if len(read_indexes) != 1:
        return False
    read_index = normalize_index(read_indexes[0])
    write_index = normalize_index(result.write_index)
    read_is_row_major = starts_with_var(read_index, row) and read_index.endswith("+%s" % col)
    write_is_transposed = starts_with_var(write_index, col) and write_index.endswith("+%s" % row)
    if read_is_row_major and write_is_transposed:
        set_operation(result, "transpose2d", "recognized 2D contiguous row-major transpose")
        return True
    return False


def set_operation(result: ScannerResult, operation: str, claim: str) -> None:
    result.operation = operation
    result.evidence.append({"id": "scan.intent", "claim": claim, "confidence": 0.98})


def is_cast_expression(expr: str, read: str, idx: str) -> bool:
    ref = re.escape(indexed_ref(read, idx))
    return bool(
        re.fullmatch(r"\([A-Za-z_][\w:<>]*\)\s*%s" % ref, expr)
        or re.fullmatch(r"static_cast\s*<\s*[^>]+\s*>\s*\(\s*%s\s*\)" % ref, expr)
    )


def is_relu_expression(expr: str, read: str, idx: str) -> bool:
    ref = re.escape(indexed_ref(read, idx))
    ternary = r"%s\s*>\s*0(?:\.0)?f?\s*\?\s*%s\s*:\s*0(?:\.0)?f?" % (ref, ref)
    max_call = r"(?:fmaxf|max)\s*\(\s*%s\s*,\s*0(?:\.0)?f?\s*\)" % ref
    return bool(re.fullmatch(ternary, expr) or re.fullmatch(max_call, expr))


def is_exp_expression(expr: str, read: str, idx: str) -> bool:
    ref = re.escape(indexed_ref(read, idx))
    return bool(re.fullmatch(r"(?:expf|exp)\s*\(\s*%s\s*\)" % ref, expr))


def indexed_ref(name: str, index: str) -> str:
    return "%s[%s]" % (name, index)


def infer_dimensions_from_guard(result: ScannerResult) -> List[str]:
    if not result.guard_condition:
        return []
    if result.rank == 2:
        row_bound = bound_for_var(result.index_vars.get("row", ""), result.guard_condition)
        col_bound = bound_for_var(result.index_vars.get("col", ""), result.guard_condition)
        return [value for value in [row_bound, col_bound] if value]
    if result.idx_var:
        bound = bound_for_var(result.idx_var, result.guard_condition)
        return [bound] if bound else []
    return []


def bound_for_var(var: str, condition: str) -> Optional[str]:
    if not var:
        return None
    match = re.search(r"\b%s\s*<\s*(\w+)" % re.escape(var), condition)
    if match:
        return match.group(1)
    return None


def normalize_expr(text: str) -> str:
    return re.sub(r"\s+", "", text)


def normalize_index(text: str) -> str:
    return normalize_expr(text).replace("(", "").replace(")", "")


def starts_with_var(index: str, var: str) -> bool:
    return index == var or index.startswith("%s*" % var) or index.startswith("%s+" % var)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
