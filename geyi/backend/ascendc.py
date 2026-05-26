"""AscendC backend scaffold for Phase 1 hardware validation."""

from __future__ import annotations

import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from geyi.backend.model import CompiledArtifact, GeneratedProject, sha256_file
from geyi.contract.model import SemanticContract
from geyi.planner.plan import TranslationPlan


SUPPORTED_TEMPLATES = {
    "ascendc.elementwise_binary_1d",
    "ascendc.elementwise_unary_1d",
    "ascendc.copy_cast_1d",
    "ascendc.transpose2d",
    "ascendc.row_reduce_sum",
}


class AscendCBackendError(RuntimeError):
    pass


class AscendCBackend:
    name = "ascendc"

    def can_generate(self, plan: TranslationPlan) -> bool:
        return plan.backend == self.name and plan.template in SUPPORTED_TEMPLATES

    def generate(self, contract: SemanticContract, plan: TranslationPlan, root: Path) -> GeneratedProject:
        if not self.can_generate(plan):
            raise AscendCBackendError("AscendC backend cannot generate this plan")
        if plan.parameters["operation"] == "exp":
            raise AscendCBackendError("AscendC scaffold does not support exp yet")

        root.mkdir(parents=True, exist_ok=True)
        for dirname in ["op_kernel", "op_host", "scripts"]:
            (root / dirname).mkdir(parents=True, exist_ok=True)

        entry = safe_name(plan.operator_entry or contract.entry)
        files = {
            "tiling": root / "op_kernel" / ("%s_tiling.h" % entry),
            "kernel": root / "op_kernel" / ("%s_kernel.asc" % entry),
            "host": root / "op_host" / ("%s.asc" % entry),
            "data_utils": root / "op_host" / "data_utils.h",
            "cmake": root / "CMakeLists.txt",
            "run": root / "run.sh",
            "run_cases": root / "scripts" / "run_cases.py",
            "golden": root / "scripts" / "golden.py",
            "cases": root / "scripts" / "cases.json",
            "readme": root / "README.md",
            "metadata": root / "metadata.json",
        }

        files["tiling"].write_text(render_tiling_header(entry), encoding="utf-8")
        files["kernel"].write_text(render_kernel(plan, entry), encoding="utf-8")
        files["host"].write_text(render_host(plan, entry), encoding="utf-8")
        files["data_utils"].write_text(render_data_utils(), encoding="utf-8")
        files["cmake"].write_text(render_cmake(plan, entry), encoding="utf-8")
        files["run"].write_text(render_run_sh(plan, entry), encoding="utf-8")
        files["run"].chmod(0o755)
        files["run_cases"].write_text(render_run_cases_py(plan, entry), encoding="utf-8")
        files["golden"].write_text(render_golden_py(plan), encoding="utf-8")
        files["cases"].write_text(json_dumps({"coverage_cases": plan.parameters["coverage_cases"]}), encoding="utf-8")
        files["readme"].write_text(render_readme(plan, entry), encoding="utf-8")

        project = GeneratedProject(
            root=str(root),
            backend=self.name,
            kernel_sources=[str(Path("op_kernel") / files["kernel"].name)],
            host_sources=[str(Path("op_host") / files["host"].name)],
            bindings=[],
            build_files=["CMakeLists.txt", "run.sh"],
            tests=[str(Path("scripts") / "run_cases.py"), str(Path("scripts") / "cases.json")],
            metadata={
                "contract_hash": contract.contract_hash,
                "strategy": plan.strategy,
                "template": plan.template,
                "operation": plan.parameters["operation"],
                "operator_entry": entry,
                "execution_mode": "ascendc_direct_invoke",
                "target": plan.parameters.get("target", "scaffold"),
                "npu_arch": plan.parameters.get("npu_arch", "dav-2201"),
                "npu_execution_claimed": plan.parameters.get("target") == "cann",
            },
            assumptions=list(plan.required_assumptions),
        )
        files["metadata"].write_text(
            json_dumps(
                {
                    "generated_project": project.to_dict(),
                    "plan": plan.to_dict(),
                    "server_command": "bash run.sh",
                }
            ),
            encoding="utf-8",
        )
        return project

    def compile(self, project: GeneratedProject, build_root: Path, target: str = "scaffold") -> CompiledArtifact:
        build_root.mkdir(parents=True, exist_ok=True)
        if target == "cann":
            return self._compile_and_run_on_cann(project, build_root)
        return self._validate_scaffold(project, build_root)

    def load_project(self, root: Path) -> GeneratedProject:
        metadata_file = root / "metadata.json"
        if not metadata_file.exists():
            raise AscendCBackendError("generated metadata missing: %s" % metadata_file)
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        return GeneratedProject.from_dict(payload["generated_project"], root=str(root))

    def load_artifact(self, build_root: Path, reused: bool) -> CompiledArtifact:
        metadata_file = build_root / "artifact_metadata.json"
        if not metadata_file.exists():
            raise AscendCBackendError("artifact metadata missing: %s" % metadata_file)
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        artifact_path = build_root / Path(payload["path"]).name
        return CompiledArtifact.from_dict(payload, path=str(artifact_path), reused=reused)

    def _validate_scaffold(self, project: GeneratedProject, build_root: Path) -> CompiledArtifact:
        root = Path(project.root)
        for script in ["scripts/run_cases.py", "scripts/golden.py"]:
            cfile = build_root / (Path(script).stem + ".pyc")
            py_compile.compile(str(root / script), cfile=str(cfile), doraise=True)
        subprocess.run(["bash", "-n", str(root / "run.sh")], check=True)

        manifest = build_root / "ascendc_scaffold_manifest.json"
        source_hashes = {}
        for relpath in project.kernel_sources + project.host_sources + project.build_files + project.tests:
            path = root / relpath
            if path.exists() and path.is_file():
                source_hashes[relpath] = sha256_file(path)
        manifest.write_text(
            json_dumps(
                {
                    "backend": self.name,
                    "compiler": "scaffold_static_check",
                    "source_hashes": source_hashes,
                    "execution_mode": "not_run",
                }
            ),
            encoding="utf-8",
        )
        artifact = CompiledArtifact(
            path=str(manifest),
            artifact_hash=sha256_file(manifest),
            backend=self.name,
            compiler="scaffold_static_check",
            reused=False,
            metadata={"execution_mode": "not_run", "npu_execution_claimed": False},
        )
        (build_root / "artifact_metadata.json").write_text(json_dumps(artifact.to_dict()), encoding="utf-8")
        return artifact

    def _compile_and_run_on_cann(self, project: GeneratedProject, build_root: Path) -> CompiledArtifact:
        root = Path(project.root)
        stdout_log = build_root / "run_stdout.log"
        stderr_log = build_root / "run_stderr.log"
        env = os.environ.copy()
        process = subprocess.run(
            ["bash", "run.sh"],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout_log.write_text(process.stdout, encoding="utf-8")
        stderr_log.write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            failure_report = build_root / "run_failure.json"
            failure_report.write_text(
                json_dumps(
                    {
                        "exit_code": process.returncode,
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                        "stdout_tail": tail_text(process.stdout),
                        "stderr_tail": tail_text(process.stderr),
                    }
                ),
                encoding="utf-8",
            )
            raise AscendCBackendError(
                "AscendC run.sh failed with exit code %d\n"
                "stdout log: %s\n"
                "stderr log: %s\n"
                "failure report: %s\n"
                "--- stdout tail ---\n%s\n"
                "--- stderr tail ---\n%s"
                % (
                    process.returncode,
                    stdout_log,
                    stderr_log,
                    failure_report,
                    tail_text(process.stdout),
                    tail_text(process.stderr),
                )
            )

        entry = safe_name(str(project.metadata.get("operator_entry") or Path(project.root).name))
        artifact_path = root / "build" / entry
        if not artifact_path.exists():
            candidates = [path for path in (root / "build").glob("*") if path.is_file() and os.access(path, os.X_OK)]
            artifact_path = candidates[0] if candidates else root / "build" / "output" / "verification_report.json"
        artifact = CompiledArtifact(
            path=str(artifact_path),
            artifact_hash=sha256_file(artifact_path),
            backend=self.name,
            compiler="cann_cmake",
            reused=False,
            metadata={
                "execution_mode": "cann_runtime",
                "npu_execution_claimed": True,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            },
        )
        (build_root / "artifact_metadata.json").write_text(json_dumps(artifact.to_dict()), encoding="utf-8")
        return artifact


def render_tiling_header(entry: str) -> str:
    guard = "%s_TILING_H" % entry.upper()
    return """#ifndef {guard}
#define {guard}

#include <cstdint>

constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct GeyiTilingData {{
    uint32_t blockNum;
    uint64_t totalLength;
    uint64_t rows;
    uint64_t cols;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
}};

#endif
""".format(guard=guard)


def render_kernel(plan: TranslationPlan, entry: str) -> str:
    operation = str(plan.parameters["operation"])
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    dtypes = dict(plan.parameters.get("dtypes") or {})
    if operation in {"add", "mul", "relu", "neg", "copy", "cast"}:
        return render_vector_1d_kernel(operation, inputs, output, dtypes, entry)
    return render_scalar_kernel(operation, inputs, output, dtypes, entry)


def render_vector_1d_kernel(operation: str, inputs: List[str], output: str, dtypes: Dict[str, Any], entry: str) -> str:
    input0_type = ascend_type(dtypes.get(inputs[0], "float32"))
    input1_type = ascend_type(dtypes.get(inputs[1], "float32")) if len(inputs) > 1 else "float"
    output_type = ascend_type(dtypes.get(output, "float32"))

    input1_arg = ", GM_ADDR input1" if len(inputs) > 1 else ""
    input1_init_arg = ", input1" if len(inputs) > 1 else ""
    input1_member = "    AscendC::GlobalTensor<{type}> input1Gm;\n".format(type=input1_type) if len(inputs) > 1 else ""
    input1_set = (
        "        input1Gm.SetGlobalBuffer((__gm__ {type}*)input1 + gmOffset, total);\n".format(type=input1_type)
        if len(inputs) > 1
        else ""
    )
    input1_queue = "    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue1;\n" if len(inputs) > 1 else ""
    input1_buffer = (
        "        pipe_->InitBuffer(inQueue1, DOUBLE_BUFFER, TILE_LENGTH * sizeof({type}));\n".format(type=input1_type)
        if len(inputs) > 1
        else ""
    )
    input1_copy = (
        """        AscendC::LocalTensor<{type}> input1Local = inQueue1.AllocTensor<{type}>();
        AscendC::DataCopyPad(input1Local, input1Gm[progress],
            {{1, static_cast<uint16_t>(count * sizeof({type})), 0, 0}},
            {{false, 0, 0, 0}});
        inQueue1.EnQue(input1Local);
""".format(type=input1_type)
        if len(inputs) > 1
        else ""
    )
    input1_deque = (
        "        AscendC::LocalTensor<{type}> input1Local = inQueue1.DeQue<{type}>();\n".format(type=input1_type)
        if len(inputs) > 1
        else ""
    )
    input1_free = "        inQueue1.FreeTensor(input1Local);\n" if len(inputs) > 1 else ""
    compute = render_vector_compute(operation, input0_type, input1_type, output_type, len(inputs))

    return """#include "kernel_operator.h"
#include "{entry}_tiling.h"

class KernelGeyiVector1D {{
public:
    __aicore__ inline KernelGeyiVector1D(AscendC::TPipe* pipe)
    {{
        pipe_ = pipe;
    }}

    __aicore__ inline void Init(GM_ADDR input0{input1_arg}, GM_ADDR output, const __gm__ GeyiTilingData* tiling)
    {{
        uint32_t blockIdx = AscendC::GetBlockIdx();
        total = 0;
        tileNum = 0;
        tailTileElementNum = 0;
        if (tiling->totalLength == 0 || blockIdx >= tiling->blockNum) {{
            return;
        }}
        total = (blockIdx < tiling->blockNum - 1) ? tiling->numPerCore : tiling->tailNumLastCore;
        if (total == 0) {{
            return;
        }}
        uint64_t gmOffset = static_cast<uint64_t>(blockIdx) * tiling->numPerCore;
        input0Gm.SetGlobalBuffer((__gm__ {input0_type}*)input0 + gmOffset, total);
{input1_set}        outputGm.SetGlobalBuffer((__gm__ {output_type}*)output + gmOffset, total);

        tileNum = (total + TILE_LENGTH - 1) / TILE_LENGTH;
        tailTileElementNum = total - TILE_LENGTH * (tileNum - 1);

        pipe_->InitBuffer(inQueue0, DOUBLE_BUFFER, TILE_LENGTH * sizeof({input0_type}));
{input1_buffer}        pipe_->InitBuffer(outQueue, DOUBLE_BUFFER, TILE_LENGTH * sizeof({output_type}));
    }}

    __aicore__ inline void Process()
    {{
        for (uint64_t tile = 0; tile < tileNum; ++tile) {{
            uint32_t count = static_cast<uint32_t>((tile == tileNum - 1) ? tailTileElementNum : TILE_LENGTH);
            uint64_t progress = tile * TILE_LENGTH;
            CopyIn(progress, count);
            Compute(count);
            CopyOut(progress, count);
        }}
    }}

private:
    __aicore__ inline void CopyIn(uint64_t progress, uint32_t count)
    {{
        AscendC::LocalTensor<{input0_type}> input0Local = inQueue0.AllocTensor<{input0_type}>();
        AscendC::DataCopyPad(input0Local, input0Gm[progress],
            {{1, static_cast<uint16_t>(count * sizeof({input0_type})), 0, 0}},
            {{false, 0, 0, 0}});
        inQueue0.EnQue(input0Local);
{input1_copy}    }}

    __aicore__ inline void Compute(uint32_t count)
    {{
        AscendC::LocalTensor<{input0_type}> input0Local = inQueue0.DeQue<{input0_type}>();
{input1_deque}        AscendC::LocalTensor<{output_type}> outputLocal = outQueue.AllocTensor<{output_type}>();
{compute}
        outQueue.EnQue<{output_type}>(outputLocal);
        inQueue0.FreeTensor(input0Local);
{input1_free}    }}

    __aicore__ inline void CopyOut(uint64_t progress, uint32_t count)
    {{
        AscendC::LocalTensor<{output_type}> outputLocal = outQueue.DeQue<{output_type}>();
        AscendC::DataCopyPad(outputGm[progress], outputLocal,
            {{1, static_cast<uint16_t>(count * sizeof({output_type})), 0, 0}});
        outQueue.FreeTensor(outputLocal);
    }}

    AscendC::TPipe* pipe_;
    AscendC::GlobalTensor<{input0_type}> input0Gm;
{input1_member}    AscendC::GlobalTensor<{output_type}> outputGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0;
{input1_queue}    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint64_t total;
    uint64_t tileNum;
    uint64_t tailTileElementNum;
}};

extern "C" __global__ __vector__ void {entry}_kernel(GM_ADDR input0{input1_arg}, GM_ADDR output, GM_ADDR tiling)
{{
    AscendC::TPipe pipe;
    KernelGeyiVector1D op(&pipe);
    op.Init(input0{input1_init_arg}, output, reinterpret_cast<const __gm__ GeyiTilingData*>(tiling));
    op.Process();
}}
""".format(
        entry=entry,
        input1_arg=input1_arg,
        input1_init_arg=input1_init_arg,
        input0_type=input0_type,
        input1_set=input1_set,
        input1_buffer=input1_buffer,
        output_type=output_type,
        input1_copy=input1_copy,
        input1_deque=input1_deque,
        compute=compute,
        input1_free=input1_free,
        input1_member=input1_member,
        input1_queue=input1_queue,
    )


def render_vector_compute(operation: str, input0_type: str, input1_type: str, output_type: str, input_count: int) -> str:
    if operation == "add":
        return "        AscendC::Add(outputLocal, input0Local, input1Local, count);\n"
    if operation == "mul":
        return "        AscendC::Mul(outputLocal, input0Local, input1Local, count);\n"
    if operation == "relu":
        return "        AscendC::Relu(outputLocal, input0Local, count);\n"
    if operation == "neg":
        return "        AscendC::Muls(outputLocal, input0Local, static_cast<{type}>(-1.0f), count);\n".format(type=output_type)
    if operation == "cast" or input0_type != output_type:
        return "        AscendC::Cast<{out_type}>(outputLocal, input0Local, AscendC::RoundMode::CAST_NONE, count);\n".format(out_type=output_type)
    if operation == "copy":
        return "        AscendC::Adds(outputLocal, input0Local, static_cast<{type}>(0), count);\n".format(type=output_type)
    raise AscendCBackendError("unsupported vector operation for AscendC kernel: %s" % operation)


def render_scalar_kernel(operation: str, inputs: List[str], output: str, dtypes: Dict[str, Any], entry: str) -> str:
    input0_type = ascend_type(dtypes.get(inputs[0], "float32"))
    input1_type = ascend_type(dtypes.get(inputs[1], "float32")) if len(inputs) > 1 else "float"
    output_type = ascend_type(dtypes.get(output, "float32"))
    kernel_args = "GM_ADDR input0, "
    if len(inputs) > 1:
        kernel_args += "GM_ADDR input1, "
    kernel_args += "GM_ADDR output, GM_ADDR tiling"

    global_tensors = [
        "    AscendC::GlobalTensor<{type}> input0Gm;".format(type=input0_type),
        "    input0Gm.SetGlobalBuffer((__gm__ {type}*)input0, tilingData->totalLength);".format(type=input0_type),
    ]
    if len(inputs) > 1:
        global_tensors.extend(
            [
                "    AscendC::GlobalTensor<{type}> input1Gm;".format(type=input1_type),
                "    input1Gm.SetGlobalBuffer((__gm__ {type}*)input1, tilingData->totalLength);".format(type=input1_type),
            ]
        )
    global_tensors.extend(
        [
            "    AscendC::GlobalTensor<{type}> outputGm;".format(type=output_type),
            "    outputGm.SetGlobalBuffer((__gm__ {type}*)output, outputLength);".format(type=output_type),
        ]
    )

    body = render_kernel_body(operation, len(inputs), output_type)
    return """#include "kernel_operator.h"
#include "{entry}_tiling.h"

extern "C" __global__ __vector__ void {entry}_kernel({kernel_args})
{{
    const __gm__ GeyiTilingData* tilingData = reinterpret_cast<const __gm__ GeyiTilingData*>(tiling);
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = tilingData->blockNum;
    uint32_t rows = tilingData->rows;
    uint32_t cols = tilingData->cols;
    uint32_t total = tilingData->totalLength;
    uint32_t outputLength = {output_length};
{global_tensors}

{body}
}}
""".format(
        entry=entry,
        kernel_args=kernel_args,
        output_length="rows" if operation == "row_sum" else "total",
        global_tensors="\n".join(global_tensors),
        body=body,
    )


def render_kernel_body(operation: str, input_count: int, output_type: str) -> str:
    if operation in {"add", "mul"}:
        op = "+" if operation == "add" else "*"
        return """    for (uint32_t idx = blockIdx; idx < total; idx += blockNum) {
        float lhs = static_cast<float>(input0Gm.GetValue(idx));
        float rhs = static_cast<float>(input1Gm.GetValue(idx));
        outputGm.SetValue(idx, static_cast<%s>(lhs %s rhs));
    }""" % (output_type, op)
    if operation == "relu":
        return """    for (uint32_t idx = blockIdx; idx < total; idx += blockNum) {
        float value = static_cast<float>(input0Gm.GetValue(idx));
        outputGm.SetValue(idx, static_cast<%s>(value > 0.0f ? value : 0.0f));
    }""" % output_type
    if operation == "neg":
        return """    for (uint32_t idx = blockIdx; idx < total; idx += blockNum) {
        float value = static_cast<float>(input0Gm.GetValue(idx));
        outputGm.SetValue(idx, static_cast<%s>(-value));
    }""" % output_type
    if operation in {"copy", "cast"}:
        return """    for (uint32_t idx = blockIdx; idx < total; idx += blockNum) {
        auto value = input0Gm.GetValue(idx);
        outputGm.SetValue(idx, static_cast<%s>(value));
    }""" % output_type
    if operation == "transpose2d":
        return """    for (uint32_t idx = blockIdx; idx < total; idx += blockNum) {
        uint32_t row = idx / cols;
        uint32_t col = idx - row * cols;
        uint32_t outIdx = col * rows + row;
        float value = static_cast<float>(input0Gm.GetValue(idx));
        outputGm.SetValue(outIdx, static_cast<%s>(value));
    }""" % output_type
    if operation == "row_sum":
        return """    for (uint32_t row = blockIdx; row < rows; row += blockNum) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < cols; ++col) {
            acc += static_cast<float>(input0Gm.GetValue(row * cols + col));
        }
        outputGm.SetValue(row, static_cast<%s>(acc));
    }""" % output_type
    raise AscendCBackendError("unsupported operation for AscendC kernel: %s" % operation)


def render_host(plan: TranslationPlan, entry: str) -> str:
    operation = str(plan.parameters["operation"])
    inputs = list(plan.parameters["inputs"])
    output = str(plan.parameters["output"])
    dtypes = dict(plan.parameters.get("dtypes") or {})
    input_sizes = [dtype_size(dtypes.get(name, "float32")) for name in inputs]
    output_size = dtype_size(dtypes.get(output, "float32"))
    input_count = len(inputs)
    parse = render_host_parse(operation, input_count, input_sizes, output_size)
    launch_args = "inputDevice[0], "
    if input_count > 1:
        launch_args += "inputDevice[1], "
    launch_args += "outputDevice, tilingDevice"

    return """#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "data_utils.h"
#include "../op_kernel/{entry}_tiling.h"
#include "../op_kernel/{entry}_kernel.asc"

static void CheckAcl(aclError ret, const char* message)
{{
    if (ret != ACL_SUCCESS) {{
        std::cerr << message << " failed, ret=" << ret << std::endl;
        std::exit(static_cast<int>(ret));
    }}
}}

int32_t main(int32_t argc, char* argv[])
{{
{parse}

    CheckAcl(aclInit(nullptr), "aclInit");
    int32_t deviceId = 0;
    CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice");

    int64_t availableCoreNum = 1;
    aclError coreRet = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (coreRet != ACL_SUCCESS || availableCoreNum <= 0) {{
        availableCoreNum = 1;
    }}

    GeyiTilingData tiling;
    tiling.totalLength = totalLength;
    tiling.rows = rows;
    tiling.cols = cols;
    uint64_t scheduleLength = static_cast<uint64_t>({schedule_length});
    uint64_t totalTiles = (scheduleLength + TILE_LENGTH - 1) / TILE_LENGTH;
    if (totalTiles == 0) {{
        tiling.blockNum = 1;
        tiling.numPerCore = 0;
        tiling.tailNumLastCore = 0;
    }} else {{
        uint64_t activeCoreNum = std::min<uint64_t>(
            static_cast<uint64_t>(std::max<int64_t>(availableCoreNum, 1)),
            totalTiles);
        uint64_t tilesPerCore = (totalTiles + activeCoreNum - 1) / activeCoreNum;
        tiling.blockNum = static_cast<uint32_t>((totalTiles + tilesPerCore - 1) / tilesPerCore);
        tiling.numPerCore = tilesPerCore * TILE_LENGTH;
        tiling.tailNumLastCore = scheduleLength - tiling.numPerCore * (tiling.blockNum - 1);
    }}

    std::vector<uint8_t*> inputHost({input_count}, nullptr);
    std::vector<uint8_t*> inputDevice({input_count}, nullptr);
    for (size_t i = 0; i < {input_count}; ++i) {{
        CheckAcl(aclrtMallocHost(reinterpret_cast<void**>(&inputHost[i]), inputBytes[i]), "aclrtMallocHost(input)");
        CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&inputDevice[i]), inputBytes[i], ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)");
        if (!ReadFile(inputPaths[i], inputBytes[i], inputHost[i], inputBytes[i])) {{
            return 3;
        }}
        CheckAcl(aclrtMemcpy(inputDevice[i], inputBytes[i], inputHost[i], inputBytes[i], ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }}

    uint8_t* outputHost = nullptr;
    uint8_t* outputDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    CheckAcl(aclrtMallocHost(reinterpret_cast<void**>(&outputHost), outputBytes), "aclrtMallocHost(output)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&outputDevice), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(output)");
    CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(GeyiTilingData), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tiling)");
    CheckAcl(aclrtMemcpy(tilingDevice, sizeof(GeyiTilingData), &tiling, sizeof(GeyiTilingData), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy tiling");

    aclrtStream stream = nullptr;
    CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    {entry}_kernel<<<tiling.blockNum, nullptr, stream>>>({launch_args});
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    CheckAcl(aclrtMemcpy(outputHost, outputBytes, outputDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
    if (!WriteFile(outputPath, outputHost, outputBytes)) {{
        return 3;
    }}

    aclrtDestroyStream(stream);
    aclrtFree(tilingDevice);
    aclrtFree(outputDevice);
    aclrtFreeHost(outputHost);
    for (size_t i = 0; i < {input_count}; ++i) {{
        aclrtFree(inputDevice[i]);
        aclrtFreeHost(inputHost[i]);
    }}
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}}
""".format(
        entry=entry,
        parse=parse,
        input_count=input_count,
        schedule_length="rows" if operation == "row_sum" else "totalLength",
        launch_args=launch_args,
    )


def render_host_parse(operation: str, input_count: int, input_sizes: List[int], output_size: int) -> str:
    argc = input_count + 4 if operation in {"transpose2d", "row_sum"} else input_count + 3
    usage_dims = "<rows> <cols>" if operation in {"transpose2d", "row_sum"} else "<n>"
    lines = [
        '    if (argc != %d) {' % argc,
        '        std::cerr << "Usage: " << argv[0] << " %s <output.bin> %s" << std::endl;' % (" ".join(["<input%d.bin>" % i for i in range(input_count)]), usage_dims),
        "        return 2;",
        "    }",
        "    std::vector<std::string> inputPaths;",
    ]
    for index in range(input_count):
        lines.append("    inputPaths.push_back(argv[%d]);" % (index + 1))
    output_arg = input_count + 1
    lines.extend(
        [
            "    std::string outputPath = argv[%d];" % output_arg,
        ]
    )
    if operation in {"transpose2d", "row_sum"}:
        lines.extend(
            [
                "    uint64_t rows = static_cast<uint64_t>(std::stoull(argv[%d]));" % (output_arg + 1),
                "    uint64_t cols = static_cast<uint64_t>(std::stoull(argv[%d]));" % (output_arg + 2),
                "    uint64_t totalLength = rows * cols;",
            ]
        )
    else:
        lines.extend(
            [
                "    uint64_t rows = 1;",
                "    uint64_t cols = static_cast<uint64_t>(std::stoull(argv[%d]));" % (output_arg + 1),
                "    uint64_t totalLength = cols;",
            ]
        )
    lines.append("    std::vector<size_t> inputBytes;")
    for size in input_sizes:
        lines.append("    inputBytes.push_back(static_cast<size_t>(totalLength * %d));" % size)
    output_length = "rows" if operation == "row_sum" else "totalLength"
    lines.append("    size_t outputBytes = static_cast<size_t>(%s * %d);" % (output_length, output_size))
    return "\n".join(lines)


def render_data_utils() -> str:
    return """#pragma once

#include <cstdio>
#include <fstream>
#include <string>

#define GEYI_ERROR_LOG(fmt, args...) fprintf(stderr, "[GEYI][ERROR] " fmt "\\n", ##args)

inline bool ReadFile(const std::string& path, size_t expectedBytes, void* dst, size_t dstBytes)
{
    if (dstBytes < expectedBytes) {
        GEYI_ERROR_LOG("destination buffer is smaller than expected input bytes");
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        GEYI_ERROR_LOG("failed to open input file: %s", path.c_str());
        return false;
    }
    input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(expectedBytes));
    if (static_cast<size_t>(input.gcount()) != expectedBytes) {
        GEYI_ERROR_LOG("input file size mismatch: %s", path.c_str());
        return false;
    }
    return true;
}

inline bool WriteFile(const std::string& path, const void* src, size_t bytes)
{
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        GEYI_ERROR_LOG("failed to open output file: %s", path.c_str());
        return false;
    }
    output.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(output);
}
"""


def render_cmake(plan: TranslationPlan, entry: str) -> str:
    return """cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project({entry} LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(GEYI_NPU_ARCH "{npu_arch}" CACHE STRING "Ascend NPU architecture")
set(ACL_INCLUDE_DIR "$ENV{{ASCEND_HOME_PATH}}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{{ASCEND_HOME_PATH}}/lib64")
set(ACL_AARCH64_LIB_DIR "$ENV{{ASCEND_HOME_PATH}}/aarch64-linux/lib64")

add_executable({entry} op_host/{entry}.asc)

target_include_directories({entry} PRIVATE
    ${{CMAKE_CURRENT_SOURCE_DIR}}/op_kernel
    ${{CMAKE_CURRENT_SOURCE_DIR}}/op_host
    ${{ACL_INCLUDE_DIR}}
)

target_link_directories({entry} PRIVATE
    ${{ACL_LIB_DIR}}
    ${{ACL_AARCH64_LIB_DIR}}
)

target_link_libraries({entry} PRIVATE
    ascendcl
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options({entry} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${{GEYI_NPU_ARCH}}>
)
""".format(entry=entry, npu_arch=plan.parameters.get("npu_arch", "dav-2201"))


def render_run_sh(plan: TranslationPlan, entry: str) -> str:
    text = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="__ENTRY__"
NPU_ARCH="${GEYI_NPU_ARCH:-__NPU_ARCH__}"

if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -n "${ASCEND_HOME:-}" ]; then
  export ASCEND_HOME_PATH="${ASCEND_HOME}"
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  echo "ERROR: ASCEND_HOME_PATH is not set. Source CANN set_env.sh first." >&2
  exit 1
fi

if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
  echo "=== [1/4] Source CANN environment ==="
  source "${ASCEND_HOME_PATH}/set_env.sh"
fi

echo "=== [2/4] Configure CMake ==="
echo "GEYI project: ${SCRIPT_DIR}"
echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
echo "GEYI_NPU_ARCH: ${NPU_ARCH}"
cmake --version
python3 --version
mkdir -p build
cd build
cmake -DGEYI_NPU_ARCH="${NPU_ARCH}" ..
echo "=== [3/4] Build AscendC executable ==="
if [ "${GEYI_VERBOSE_BUILD:-0}" = "1" ]; then
  cmake --build . --verbose -j"${GEYI_BUILD_JOBS:-4}"
else
  cmake --build . -j"${GEYI_BUILD_JOBS:-4}"
fi
echo "=== [4/4] Run coverage cases ==="
python3 ../scripts/run_cases.py "./${OP_NAME}"
"""
    return text.replace("__ENTRY__", entry).replace("__NPU_ARCH__", str(plan.parameters.get("npu_arch", "dav-2201")))


def render_golden_py(plan: TranslationPlan) -> str:
    return """import numpy as np


OPERATION = {operation!r}


def compute_golden(inputs, shape):
    if OPERATION == "add":
        return inputs[0] + inputs[1]
    if OPERATION == "mul":
        return inputs[0] * inputs[1]
    if OPERATION == "relu":
        return np.maximum(inputs[0], 0)
    if OPERATION == "neg":
        return -inputs[0]
    if OPERATION in ("copy", "cast"):
        return inputs[0].astype(shape["output_dtype"])
    if OPERATION == "transpose2d":
        return inputs[0].reshape(shape["rows"], shape["cols"]).T.reshape(-1).astype(shape["output_dtype"])
    if OPERATION == "row_sum":
        return inputs[0].reshape(shape["rows"], shape["cols"]).sum(axis=1).astype(shape["output_dtype"])
    raise ValueError("unsupported operation: %s" % OPERATION)
""".format(operation=plan.parameters["operation"])


def render_run_cases_py(plan: TranslationPlan, entry: str) -> str:
    payload = {
        "operation": plan.parameters["operation"],
        "coverage_cases": plan.parameters["coverage_cases"],
        "inputs": plan.parameters["inputs"],
        "output": plan.parameters["output"],
        "dtypes": plan.parameters["dtypes"],
        "tolerance": plan.parameters["tolerance"],
    }
    text = """from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from golden import compute_golden


CONFIG = __CONFIG__


def main():
    if len(sys.argv) != 2:
        print("Usage: run_cases.py <executable>", file=sys.stderr)
        return 2
    executable = sys.argv[1]
    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)

    case_results = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    passed = True
    for index, case in enumerate(CONFIG["coverage_cases"]):
        result = run_case(executable, output_root, index, case)
        case_results.append(result)
        max_abs_diff = max(max_abs_diff, float(result["max_abs_diff"]))
        max_rel_diff = max(max_rel_diff, float(result["max_rel_diff"]))
        passed = passed and bool(result["passed"])

    report = {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "case_results": case_results,
    }
    (output_root / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


def run_case(executable, output_root, index, case):
    case_dir = output_root / ("case_%02d_%s" % (index, case["case"]))
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "output.bin"
    golden_path = case_dir / "golden.bin"

    shape = shape_for_case(case)
    arrays = make_inputs(shape)
    input_paths = []
    for input_index, array in enumerate(arrays):
        path = input_dir / ("input%d.bin" % input_index)
        array.tofile(path)
        input_paths.append(str(path))

    golden = compute_golden(arrays, shape)
    golden.tofile(golden_path)
    if shape["rows"] * shape["cols"] == 0 or golden.size == 0:
        result = compare(golden, golden)
        return {**case, **result, "skipped_device": True, "skip_reason": "empty input or output shape"}

    dims = [str(shape["rows"]), str(shape["cols"])] if CONFIG["operation"] in ("transpose2d", "row_sum") else [str(shape["n"])]
    command = [executable] + input_paths + [str(out_path)] + dims
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        return {
            **case,
            "passed": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    output = np.fromfile(out_path, dtype=np.dtype(shape["output_dtype"]))
    result = compare(output, golden)
    if not result["passed"]:
        result["output_head"] = output[:8].astype(np.float64).tolist()
        result["golden_head"] = golden[:8].astype(np.float64).tolist()
        result["output_nonzero_count"] = int(np.count_nonzero(output))
    return {**case, **result}


def shape_for_case(case):
    output_dtype = CONFIG["dtypes"][CONFIG["output"]]
    if "n" in case:
        return {"n": int(case["n"]), "rows": 1, "cols": int(case["n"]), "output_dtype": output_dtype}
    return {"rows": int(case["rows"]), "cols": int(case["cols"]), "output_dtype": output_dtype}


def make_inputs(shape):
    total = shape["rows"] * shape["cols"]
    arrays = []
    for index, name in enumerate(CONFIG["inputs"]):
        dtype = np.dtype(CONFIG["dtypes"][name])
        if np.issubdtype(dtype, np.integer):
            data = ((np.arange(total, dtype=np.int64) * 37 + 17 + index * 11) % 251 - 125).astype(dtype)
        else:
            data = (((np.arange(total, dtype=np.float32) * 37 + 17 + index * 11) % 251) - 125) / 17.0
            data = data.astype(dtype)
        arrays.append(data)
    return arrays


def compare(output, golden):
    tolerance = CONFIG["tolerance"]
    if output.shape != golden.shape:
        return {
            "passed": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "reason": "shape mismatch: output %s golden %s" % (output.shape, golden.shape),
        }
    diff = np.abs(output.astype(np.float64) - golden.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(golden.astype(np.float64)), 1e-12)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    threshold = float(tolerance["atol"]) + float(tolerance["rtol"]) * (float(np.abs(golden).max()) if golden.size else 0.0)
    return {
        "passed": bool(max_abs <= threshold),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "tolerance_threshold": threshold,
    }


if __name__ == "__main__":
    raise SystemExit(main())
"""
    return text.replace("__CONFIG__", json.dumps(payload, indent=2, sort_keys=True))


def render_readme(plan: TranslationPlan, entry: str) -> str:
    return """# Geyi AscendC Hardware Scaffold

Generated for `{entry}`.

Run on a CANN machine:

```bash
source $ASCEND_HOME_PATH/set_env.sh
bash run.sh
```

The script builds the direct-invoke AscendC executable, runs all generated
coverage cases, and writes `build/output/verification_report.json`.
""".format(entry=entry)


def ascend_type(dtype: str) -> str:
    return {
        "float32": "float",
        "float16": "half",
        "int32": "int32_t",
        "int64": "int64_t",
    }.get(str(dtype), "float")


def dtype_size(dtype: str) -> int:
    return {
        "float32": 4,
        "float16": 2,
        "int32": 4,
        "int64": 8,
    }.get(str(dtype), 4)


def safe_name(name: str) -> str:
    value = re.sub(r"\W", "_", str(name))
    if not value or value[0].isdigit():
        value = "op_%s" % value
    return value


def json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def tail_text(text: str, max_lines: int = 80) -> str:
    lines = str(text or "").splitlines()
    return "\n".join(lines[-max_lines:])
