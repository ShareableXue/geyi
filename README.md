# Geyi

Phase -1 Contract Prototype.

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

## First Run

Run the included Phase -1 example:

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

Phase -1 only builds a Semantic Contract and confidence report. It does not generate, compile, or run NPU code.

## Verify The Prototype

```bash
conda activate geyi_dev
python -m unittest discover -s tests -v
```
