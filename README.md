# Geyi

Phase 0 first end-to-end MVP.

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

## Phase 0 Run

Run the first end-to-end MVP:

```bash
conda activate geyi_dev
geyi run examples/vector_add/vector_add.cu \
  --spec examples/vector_add/geyi.yaml \
  --out .geyi/out/vector_add
```

This closes the Phase 0 loop:

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

Phase 0 currently supports only 1D contiguous `float32` `vector_add`.

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

Use `geyi info` first when you want to inspect the contract and confidence report without generating a Phase 0 project. Use `geyi run` for the supported Phase 0 vector-add end-to-end path.

## Verify The Prototype

```bash
conda activate geyi_dev
python -m unittest discover -s tests -v
```
