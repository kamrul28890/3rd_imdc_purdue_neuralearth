# Mosqlimate IMDC 2026 Forecasting Pipeline

This repository contains an IMDC 2026 dengue/chikungunya forecasting pipeline for Mosqlimate work. It includes exploratory notebooks, reusable Python modules, baseline evaluation tests, reports, generated figures, and baseline metrics.

## Contents

- `notebooks/`: data audit and exploratory analyses.
- `src/imdc/`: package code for data loading, aggregation, validation, fold creation, evaluation, metrics, postprocessing, and baseline runs.
- `tests/`: unit tests for folds, leakage checks, metrics, and baseline harness behavior.
- `reports/`: report source and rendered PDF.
- `results/`: generated figures and baseline metric outputs.
- `data/raw/data_imdc_2026/`: raw IMDC 2026 dataset files tracked with Git LFS.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the test suite:

```bash
pytest
```

## Data And Local Artifacts

Most large raw/processed data, model artifacts, cache files, notebook checkpoints, and local environment files are intentionally excluded from Git. The committed reports and results provide the current analysis outputs, and the IMDC 2026 raw dataset is included separately through Git LFS.

The IMDC 2026 raw dataset under `data/raw/data_imdc_2026/` is tracked with Git LFS. After cloning, install Git LFS and fetch the data:

```bash
git lfs install
git lfs pull
```
