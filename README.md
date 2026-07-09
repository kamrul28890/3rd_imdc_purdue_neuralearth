# IMDC 2026 — Team Neural Earth

Forecasting pipeline for the **3rd Infodengue–Mosqlimate Dengue Challenge (IMDC 2026)**:
weekly probabilistic dengue and chikungunya forecasts for Brazilian states and cities.

**Contact:** Md Kamruzzaman Kamrul — kamrul28890@gmail.com · Mosqlimate account: `kamrul28890`

---

## Team and contributions

The project divides into three parts. **Md Kamruzzaman Kamrul led the modeling and analysis —
the core and largest part — and drafted the manuscript**; the two co-authors led the supporting
data/validation and writing/documentation streams.

| Part | Lead | Scope |
|------|------|-------|
| **1. Modeling, methodology & software** | **Md Kamruzzaman Kamrul** · Purdue University *(team leader)* | Study design; the full forecasting pipeline (leakage-safe features, WIS/CRPS evaluation harness, baselines, LightGBM + CQR, GRU deep ensemble, mechanistic model, Vincentization ensemble, conformal recalibration); model selection & formal analysis; submission packaging; figures; original manuscript draft. |
| **2. Data curation, validation & reproducibility** | **Abdullah Al Helal** · Oklahoma State University | Dataset assembly & integrity checks; EW‑25 leakage-compliance and interval-validity verification; reproducibility and test suite; manuscript review & editing. |
| **3. Manuscript, literature & documentation** | **Eashraque Jahan Easha** · University of Denver | Literature review & references; results reporting and figure narrative; repository documentation; manuscript review & editing. |

Full CRediT-taxonomy statement: [`MODEL_CARD.md` §1](MODEL_CARD.md).

---

## IMDC documentation checklist

The information required by the challenge guidelines is documented as follows.
**[`MODEL_CARD.md`](MODEL_CARD.md) is the single compliance document** — its sections map
one-to-one to these items.

| # | Required item | Where |
|---|---------------|-------|
| 1 | Team name, main contact, members' names/affiliations/**contributions** | This README (above) · [`MODEL_CARD.md` §1](MODEL_CARD.md) |
| 2 | Repository structure | [Repository layout](#repository-layout) · [`MODEL_CARD.md` §2](MODEL_CARD.md) |
| 3 | Libraries and dependencies | [Setup](#setup-and-reproduction) · [`MODEL_CARD.md` §3](MODEL_CARD.md) · `environment.lock.yml` |
| 4 | Data and variables | [`MODEL_CARD.md` §4](MODEL_CARD.md) |
| 5 | Model training process | [`MODEL_CARD.md` §5](MODEL_CARD.md) · `reports/modeling_results_report.pdf` |
| 6 | How the **EW‑25** data-availability rule was met | [`MODEL_CARD.md` §6](MODEL_CARD.md) |
| 7 | How the prediction intervals were computed | [`MODEL_CARD.md` §7](MODEL_CARD.md) |
| 8 | DOI references | [`MODEL_CARD.md` §8](MODEL_CARD.md) |

---

## Status

| Phase | Status |
|------|--------|
| 0–1. Environment, scaffolding, data audit & EDA | ✅ Done — `reports/data_findings_report.pdf` |
| 2. Backtesting harness + baselines | ✅ Done |
| 3. Classical ML (LightGBM + CQR) | ✅ Done |
| 4. Deep learning (GRU + NegBinom) | ✅ Done |
| 5. Mechanistic model | ✅ Done (local reimplementation) |
| 6. Ensemble + conformal recalibration | ✅ Done |
| 7. Submission packaging + validation | ✅ **Validation phase submitted** (model id 83) |
| 8. Manuscript | 🟡 Draft — `paper/imdc_paper.pdf` |
| Optional tracks (chik-state, dengue/chik cities) | ✅ Done |

Automated test suite passes (`pytest tests/`); modeling write-up in
`reports/modeling_results_report.pdf`; manuscript in `paper/`. Roadmap to the September
forecast phase in `docs/FUTURE_WORK.md`.

---

## Headline result

Backtest over 26 states × 4 seasons (2022–23 … 2025–26). Mean Weighted Interval Score (WIS)
and official normalized WIS (Σ WIS / Σ cases), both lower-is-better:

| Model | WIS | normWIS | Coverage 50/80/90/95 |
|---|---|---|---|
| **Ensemble + conformal recalibration** | **1216** | **0.555** | 47/76/88/93% |
| Ensemble (Vincentization) | 1281 | 0.585 | 46/76/84/88% |
| Ensemble (inverse-WIS) | 1287 | 0.588 | 48/76/86/89% |
| LightGBM (CQR) | 1299 | 0.593 | 51/81/89/93% |
| Mechanistic | 1303 | 0.595 | 53/79/85/89% |
| Climatological | 1327 | 0.606 | 48/73/82/85% |
| GRU (NegBinom) | 1373 | 0.627 | 32/57/68/75% |
| Seasonal-naive | 1459 | 0.666 | — |
| Naive | 1664 | 0.760 | — |

**No single model wins every season** — the GRU is best on normal seasons but fails on the 2024
outlier; LightGBM is most robust to the outlier but weakest on normal seasons. The unweighted
quantile-median **ensemble is never worst on any fold**, and **conformal recalibration** of the
ensemble's tails is the largest single accuracy gain (WIS 1281 → 1216). See the modeling report
for the full analysis, including the WIS scale-dependence finding.

---

## Repository layout

```
src/imdc/                     # installable package (pip install -e .)
  config.py                   # paths, state list, target cities, quantile levels
  data/                       # loaders, fold derivation, leakage guards, aggregation
  features/panel.py           # synthetic-origin panel + leakage-safe features
  evaluation/                 # WIS/CRPS metrics, harness, baselines, postprocessing
  models/                     # LightGBM, GRU, mechanistic, ensemble + run_*.py scripts
  submission/                 # platform-schema builders + validator
results/metrics/              # scored backtest predictions & leaderboards (canonical outputs)
results/figures/              # paper-ready figures
submissions/validation/       # platform-ready forecast files per track/season/geography
tests/                        # leakage, WIS correctness, determinism, every model
notebooks/                    # 00 data audit, 01/02 EDA (executed)
reports/                      # data_findings + modeling_results (LaTeX + compiled PDF)
paper/                        # manuscript + supporting information (LaTeX + PDF)
docs/                         # PLAN, FUTURE_WORK, PAPER_PLAN, IMPROVEMENTS
data/raw/data_imdc_2026/      # official dataset (Git LFS)
data/processed/               # regenerable feature cache (gitignored; auto-rebuilds)
Makefile                      # `make reproduce` regenerates every result from raw data
```

---

## Setup and reproduction

### 1. Clone (with LFS — the raw data is ~800 MB)
```bash
git lfs install
git clone https://github.com/kamrul28890/3rd_imdc_purdue_neuralearth.git
cd 3rd_imdc_purdue_neuralearth
git lfs pull        # fetches data/raw/data_imdc_2026/
```

### 2. Environment (conda; LightGBM/XGBoost from conda-forge for a working libomp)
```bash
conda create -n py310 python=3.10 -y
conda install -n py310 -c conda-forge xgboost lightgbm tectonic pyarrow -y
conda activate py310
pip install -e .                       # pandas, numpy, torch, statsmodels, geopandas, mosqlient, ...
pip install pytest ipykernel nbconvert pymupdf python-dotenv
```
`requirements-freeze.txt` pins exact versions; `environment.lock.yml` is the full conda export.

> **Runtime note:** set `KMP_DUPLICATE_LIB_OK=TRUE` whenever LightGBM/XGBoost and PyTorch are
> imported in the same process (each links its own libomp). All `run_*` scripts set it automatically.

### 3. Reproduce
```bash
make reproduce        # all models → ensemble → optional tracks → submissions → figures → manifest
make reproduce-fast   # reuse the committed GRU predictions (skip the ~55-min GRU retrain)
make test             # full test suite, incl. determinism guards
```
Every model's scored predictions are already committed in `results/metrics/*_scored.csv`, so you
do **not** need to re-run anything to inspect or build on the results.

### Reproducibility guarantees
1. **Determinism** — baselines, LightGBM (`seed=42`), and the mechanistic model reproduce
   bit-for-bit (`tests/test_determinism.py`); the GRU is seeded but only bit-reproducible on CPU
   (Apple-MPS float ops are non-deterministic), so the committed `gru_scored.csv` is canonical.
2. **One command** — the `Makefile` runs every step in dependency order.
3. **Provenance** — `scripts/make_manifest.py` writes `RESULTS.md` with the git commit, a SHA-256
   fingerprint of the raw inputs (`data/raw/data_imdc_2026/CHECKSUMS.sha256`), and the leaderboards.
4. **Pinned environment** — `environment.lock.yml` (exact) and `environment.yml` (portable).
5. **Committed outputs** — `results/metrics/*.csv` are the canonical results the paper reads from.

---

## Deadlines

- **Validation phase (4 backtest folds):** ✅ submitted (model id 83).
- **Forecast phase (2026–27 season):** due **2026-09-10**. The committed raw data must be
  re-pulled through EW 25 2026 before generating the true forecast; the same leakage-safe
  machinery (see [`MODEL_CARD.md` §6](MODEL_CARD.md)) then produces the EW 41 2026 → EW 40 2027 horizon.
