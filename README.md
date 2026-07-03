# IMDC 2026 — Dengue Forecasting Pipeline

Forecasting pipeline for the **3rd Infodengue–Mosqlimate Dengue Challenge (IMDC 2026)**:
weekly probabilistic dengue forecasts at the state level for 26 Brazilian states,
with the end goal of a competitive submission and a PNAS-style journal paper.

This README is the entry point for **resuming work on another machine** — it records
what is done, where everything lives, how to set up the environment, and how to
pick up from here.

---

## Current status (as of 2026-07-02)

| Phase | Status |
|------|--------|
| 0. Environment & scaffolding | ✅ Done |
| 1. Data audit & EDA | ✅ Done — `reports/data_findings_report.pdf` |
| 2. Backtesting harness + baselines | ✅ Done |
| 3. Classical ML (LightGBM + CQR) | ✅ Done |
| 4. Deep learning (GRU + NegBinom) | ✅ Done |
| 5. Mechanistic | ✅ Done — local reimplementation (EpiScanner API is down; not needed) |
| 6. Ensemble | ✅ Done |
| 7. Submission formatting + validation | ✅ Done (upload pending model registration) |
| 8. Paper | 🟡 Draft — `paper/imdc_paper.pdf` |
| Optional tracks (chik-state, dengue/chik cities) | ✅ Done — 231 validated submission files |

**62 automated tests pass** (`pytest tests/`). Phases 2–4 + 6 documented in
`reports/modeling_results_report.pdf`; PNAS-style manuscript draft in `paper/`.
Full roadmap in `docs/PLAN.md`.

**To actually submit:** register a model at mosqlimate.org (web UI — the API cannot
create models; you have none yet), then run `python -m imdc.submission.upload <owner/repo>`
(add `--publish` to go live). Your API key works and is in `.env` (gitignored).
The EpiScanner API endpoint is currently down (HTTP 500), which is why Phase 5 is a
local reimplementation.

### Headline result (backtest, 26 states × 4 folds)

Mean Weighted Interval Score (WIS), lower is better:

| Model | Overall WIS | Coverage 50/80/90/95 |
|---|---|---|
| **Ensemble (Vincentization)** | **1288** | 45/75/84/87% |
| Ensemble (inverse-WIS) | 1291 | 47/77/85/89% |
| LightGBM (CQR) | 1309 | 49/80/89/93% |
| Climatological | 1327 | 48/73/82/85% |
| GRU (NegBinom) | 1373 | 32/57/68/75% |
| Seasonal-naive | 1459 | — |
| Naive | 1664 | — |

The **unweighted quantile-median ensemble is the best and most robust model** — no
single model wins every fold (the GRU wins the "normal" seasons but fails
catastrophically on the 2024 outlier; LightGBM is most robust to the outlier but loses
fold 3), so combining them beats all of them. See the modeling report for the full
analysis, including the WIS scale-dependence finding (raw mean vs. relative WIS disagree
on the single-model winner).

---

## Repository layout

```
src/imdc/                     # installable package (pip install -e .)
  config.py                   # paths, state list, target cities, quantile levels
  data/                       # loaders, fold derivation, leakage guards, aggregation
  features/panel.py           # synthetic-origin panel + leakage-safe features (Phase 3/4)
  evaluation/                 # WIS/CRPS metrics, harness, baselines, postprocessing (Phase 2)
  models/                     # ml_boosted (LGBM), dl_sequence (GRU), ensemble + run_*.py scripts
  submission/                 # (empty — Phase 7)
tests/                        # 47 tests: leakage, WIS correctness, every model
notebooks/                    # 00 data audit, 01/02 EDA (executed)
reports/                      # data_findings + modeling_results (LaTeX + compiled PDF)
results/metrics/              # ← ALL scored backtest predictions & leaderboards (the outputs)
results/figures/              # paper-ready figures
data/raw/data_imdc_2026/      # official dataset (Git LFS — see below)
data/processed/               # regenerable feature cache (gitignored; auto-rebuilds)
models/                       # trained weights (gitignored; NOT persisted — see below)
docs/PLAN.md                  # the full phase 2–8 implementation plan
```

## Resuming on another machine

### 1. Clone (with LFS — the raw data is ~800 MB via Git LFS)
```bash
git lfs install
git clone https://github.com/kamrul28890/mosqlimate-imdc-2026.git
cd mosqlimate-imdc-2026
git lfs pull        # fetches data/raw/data_imdc_2026/ (climate.csv.gz is 459 MB)
```

### 2. Environment (conda; LightGBM/XGBoost come from conda-forge for a working libomp)
```bash
conda create -n py310 python=3.10 -y
conda install -n py310 -c conda-forge xgboost lightgbm tectonic pyarrow -y
conda activate py310
pip install -e .                       # pandas, numpy, torch, statsmodels, geopandas, mosqlient, ...
pip install pytest ipykernel nbconvert pymupdf python-dotenv
```
`requirements-freeze.txt` is the exact `pip freeze` if you need to pin versions.

**⚠️ Runtime gotcha:** set `KMP_DUPLICATE_LIB_OK=TRUE` whenever LightGBM/XGBoost and
PyTorch are imported in the same process (both link their own libomp). All `run_*`
scripts set it automatically; for ad-hoc scripts, `export KMP_DUPLICATE_LIB_OK=TRUE`.

### 3. Verify
```bash
pytest tests/ -q            # expect 47 passed (~5 min; trains small models)
```

### 4. Re-run backtests (only if you need to regenerate predictions)
```bash
python -m imdc.evaluation.run_baselines   # ~2 min
python -m imdc.models.run_ml               # LightGBM, ~8 min
python -m imdc.models.run_dl               # GRU, ~55 min (CPU/MPS)
python -m imdc.models.run_ensemble         # instant (reads saved predictions)
```
**You do not need to re-run these to continue.** Every model's scored predictions are
already committed in `results/metrics/*_scored.csv`; Phases 7–8 build directly on them.

## Two things to know

- **Model weights are NOT persisted.** Each backtest retrains from scratch; `models/`
  is empty. This is intentional and costs nothing to resume: the *predictions* in
  `results/metrics/` are the expensive outputs, and the real 2026-27 forecast will
  retrain on refreshed data anyway. (If you want the trained artifacts saved, add
  `booster.save_model()` / `torch.save(state_dict)` to the model classes and re-run.)
- **Data staleness:** the committed raw data ends 2026-03-08. The real forecast phase
  needs data through EW25 2026 — **re-pull the dataset from the FTP / Mosqlimate API
  before generating the true forecast** (Phase 7).

## Deadlines & blockers

- **Validation phase (4 backtest folds) was due 2026-07-01.** The forecasts backing it
  exist and are validated, but are **not yet packaged to the platform schema (Phase 7)
  nor uploaded** (needs the API key).
- **Forecast phase (2026-27 season) due 2026-09-10.**
- **Get a mosqlimate.org API key** — required for Phase 5 (EpiScanner) and Phase 7
  (upload). Registration/model-registration is web-UI only.
