# DentalHGT

DentalHGT is a research pipeline for multi-label dental pathology detection in
panoramic dental radiographs. It represents each image as a heterogeneous graph
with tooth, quadrant, and arch nodes, then applies a Heterogeneous Graph
Transformer to model anatomical context such as mesial-distal adjacency,
bilateral symmetry, antagonist relationships, and hierarchy edges.

The repository contains the training and evaluation code, publication figures,
paper source, configuration files, tests, and lightweight result tables. The
DENTEX dataset and generated model checkpoints are intentionally excluded from
git because they are large external artifacts.

## Repository Layout

```text
configs/          Default experiment configuration
figures/          Generated paper figures
results/          Lightweight CSV/JSON results; checkpoints are ignored
src/              DentalHGT source code
tests/            Unit and smoke tests
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- torch-geometric
- DENTEX dataset files placed under `dataset/dentex`

Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Optional dependencies:

```bash
python -m pip install -e ".[yolo]"
```

## Data

Download DENTEX from the upstream dataset source and keep it outside git at:

```text
dataset/dentex/
```

The default config expects these paths:

```text
dataset/dentex/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json
dataset/dentex/DENTEX/training_data/quadrant-enumeration-disease/xrays/
dataset/dentex/DENTEX/validation_triple.json
dataset/dentex/DENTEX/validation_data/quadrant_enumeration_disease/xrays/
```

The dataset license and redistribution terms are controlled by the DENTEX
authors. Do not commit the dataset files to this repository.

## Usage

Run the full pipeline:

```bash
python -m src.run_pipeline --config configs/default.json
```

Run a small smoke test on a few images:

```bash
python -m src.run_pipeline --config configs/default.json --dry-run
```

Switch backbone:

```bash
python -m src.run_pipeline --config configs/default.json --backbone dinov2
```

Run edge-type ablations:

```bash
python -m src.edge_ablation --config configs/default.json --backbone resnet50
```

Regenerate figures from saved results:

```bash
python -m src.figures figures
```

## Tests

```bash
python -m pytest -q
```

Current local status before the initial GitHub upload:

```text
19 passed
```

## GitHub Upload Notes

This repository is configured to keep large generated artifacts out of git:

- `dataset/`
- downloaded `.zip` archives
- model checkpoints such as `results/*.pt`
- generated feature tensors such as `results/*.npz`
- Python, LaTeX, and local editor caches

Figures and lightweight CSV/JSON results are trackable so the GitHub repository
remains useful without requiring a full training run. The unpublished manuscript
source and PDF are intentionally kept out of the public repository until release
or publication.
