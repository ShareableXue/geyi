# Geyi

Phase 4 model-level harness prototype for source-available PyTorch extensions.

## Install

From a fresh clone:

```bash
cd geyi
conda create -n geyi_dev python=3.12
conda activate geyi_dev
python -m pip install -e .
```

After installation, the `geyi` command is available in that conda environment:

```bash
geyi --help
```

For development, keep using editable install (`-e`) so source changes take effect immediately. For a non-development install, use:

```bash
python -m pip install .
```

## First Run: Contract Info

Run the included contract example:

```bash
conda activate geyi_dev
geyi info examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --json
```

This prints the Semantic Contract JSON and writes session artifacts under:

```text
.geyi/sessions/<session_id>/
├── input/
├── evidence/scanner.json
├── contract.json
├── confidence_report.json
└── logs/
```

## Phase 1 Run

Run the first end-to-end MVP:

```bash
conda activate geyi_dev
geyi run examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --out .geyi/out/vector_add
```

This closes the deterministic Phase 1 loop:

```text
CUDA source + geyi.yaml
  -> Semantic Contract
  -> rule TranslationPlan
  -> GeneratedProject
  -> compile artifact
  -> golden VerificationReport
```

The selected Phase 0 backend identity is `tilelang`. In a local environment without an importable TileLang package or Ascend/CANN toolkit, Geyi generates a small TileLang-shaped Python project, compiles it to bytecode, and verifies it against deterministic CPU golden data. The verification report marks this honestly as `golden` on `local_cpu`; it does not claim NPU execution.

Session artifacts include:

```text
.geyi/sessions/<session_id>/
├── contract.json
├── plan.json
├── generated/
├── build/
├── verification_report.json
└── logs/
```

Phase 1 currently supports deterministic 1D elementwise unary/binary, copy/cast,
2D contiguous transpose, and row-wise reduce sum.

## Phase 2 LLM Planner

Run the template-gap example through the constrained planner:

```bash
conda activate geyi_dev
geyi run examples/template_gap/fused_add_relu.cu \
  --spec examples/template_gap/geyi.yaml \
  --allow-llm-plan
```

The default Phase 2 provider is `mock`, so this works offline and still writes
transparent LLM usage metadata. Real OpenAI-compatible calls require an API key:

```bash
geyi run examples/template_gap/fused_add_relu.cu \
  --spec examples/template_gap/geyi.yaml \
  --allow-llm-plan \
  --llm-provider openai-compatible
```

DeepSeek is available as a first-class OpenAI-compatible provider:

```bash
export DEEPSEEK_API_KEY=...
geyi run examples/template_gap/fused_add_relu.cu \
  --spec examples/template_gap/geyi.yaml \
  --allow-llm-plan \
  --llm-provider deepseek
```

You can manually switch provider, model, and endpoint per run:

```bash
geyi run examples/template_gap/fused_add_relu.cu \
  --spec examples/template_gap/geyi.yaml \
  --allow-llm-plan \
  --llm-provider deepseek \
  --llm-model deepseek-v4-pro

geyi run examples/template_gap/fused_add_relu.cu \
  --spec examples/template_gap/geyi.yaml \
  --allow-llm-plan \
  --llm-provider openai-compatible \
  --llm-model vendor-model-name \
  --llm-base-url https://your-gateway.example/v1/chat/completions
```

The Phase 2 path keeps LLM output constrained to planner JSON. Geyi still owns
template code generation, compile, golden verification, repair handoff, and the
final verification report.

## Phase 3 CANN Library Metadata

Build the locked CANN hotset index:

```bash
conda activate geyi_dev
geyi library index --lock geyi-library.lock
```

Search is exact op/alias retrieval. It returns source paths, revision, checksum,
license, and contract-signature evidence:

```bash
geyi library search --op rms_norm --json
```

The default `geyi-library.lock` pins the local CANN open-source operator repos
under `../cann_opensource_ops_repos`. Each source has a locked revision,
license, and checksum. A revision, checksum, or disallowed-license mismatch
fails before an index is emitted. Strategy 0 recall is intentionally exact:
similar names are not treated as semantic equivalence.

## Phase 3 Performance Hints

Conservative hints can be recorded after the existing correctness path passes:

```bash
geyi run examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --opt-level conservative
```

This still runs the normal deterministic compile and golden verification. The
hints are written to `optimization_hints.json`; Phase 3 does not rewrite kernels
with low-confidence performance changes.

Describe a small autotune search space with verification attached:

```bash
geyi tune examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --search-space small
```

The tuning report records candidate parameters and the baseline verification
report. It also records candidate UB estimates, cannbot source digests, and
the profiler policy.

On an NPU environment, run the generated AscendC operator and attach a real
`msprof op` baseline measurement:

```bash
geyi tune examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --backend ascendc \
  --target cann \
  --search-space small \
  --profile-generated
```

This path generates the direct-invoke AscendC executable, runs the generated
coverage cases on NPU, then profiles the first non-empty passing case. For
custom runners, attach an explicit command:

```bash
geyi tune examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --search-space small \
  --kernel-name vector_add_kernel \
  --profile-command "python run_vector_add.py"
```

Geyi wraps this as:

```bash
msprof op --kernel-name=vector_add_kernel --warm-up=10 --launch-count=5 python run_vector_add.py
```

The parsed metrics and raw stdout/stderr logs are written to
`performance_report.json`. Applying candidate-specific code changes and
selecting a measured winner remain later hardware-integration work.

## Phase 4 Model-Level Harness

Run a Python entrypoint with the source-available PyTorch extension patch:

```bash
geyi patch python examples/pytorch_load_inline/run.py
```

This intercepts `torch.utils.cpp_extension.load_inline`, writes inline C++/CUDA
sources plus an auto-generated `geyi.yaml` under `.geyi/model_harness`, and
records an op-level cache key. Compiled-only `.so` loads are reported as
black-box boundaries; Geyi does not decompile them or treat them as source.
The default mode is capture-only and offline-friendly; pass `--execute-original`
when you want to call the real PyTorch loader after capture.

Session artifacts include:

```text
.geyi/sessions/<session_id>/
├── model_harness_report.json
└── logs/patch.log
```

## Workflow

For your own CUDA kernel:

1. Put the CUDA source in your project, for example `kernel.cu`.
2. Add a `geyi.yaml` that names the entry kernel, launch config, tensors, shapes, dtypes, and assumptions.
3. Run:

```bash
geyi info kernel.cu --spec geyi.yaml --json
```

4. Inspect:
   - `confidence`
   - `recommended_path`
   - `intents`
   - `effects`
   - `unknowns`
   - `rejections`
   - `.geyi/sessions/<session_id>/confidence_report.json`

Use `geyi info` first when you want to inspect the contract and confidence report without generating a project. Use `geyi run` for deterministic paths, add `--allow-llm-plan` only when a contract routes to the Phase 2 planner, and use `geyi library` when you want auditable CANN hotset retrieval.

## Verify The Prototype

```bash
conda activate geyi_dev
python -m pytest -q
```
