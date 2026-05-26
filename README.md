# Geyi

Phase 2 constrained LLM planner prototype.

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

Use `geyi info` first when you want to inspect the contract and confidence report without generating a project. Use `geyi run` for deterministic paths, and add `--allow-llm-plan` only when a contract routes to the Phase 2 planner.

## Verify The Prototype

```bash
conda activate geyi_dev
python -m unittest discover -s tests -v
```
