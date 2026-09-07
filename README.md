# IMDC 2026 — Team Neural Earth

Forecasting pipeline for the **3rd Infodengue–Mosqlimate Dengue Challenge (IMDC 2026)**:
weekly probabilistic dengue and chikungunya forecasts for Brazilian states and cities.

**Main contact:** Md Kamruzzaman Kamrul — kamrul28890@gmail.com · Mosqlimate account: `kamrul28890`
**Repository:** https://github.com/kamrul28890/3rd_imdc_purdue_neuralearth

This README is the single, comprehensive documentation for the submission. It contains all the
information required by the IMDC guidelines, organized as follows:

1. [Team, contact and member contributions](#1-team-contact-and-member-contributions)
2. [Repository structure](#2-repository-structure)
3. [Libraries and dependencies](#3-libraries-and-dependencies)
4. [Data and variables](#4-data-and-variables)
5. [Data usage restriction](#5-data-usage-restriction)
6. [Model training process](#6-model-training-process)
7. [How the EW‑25 data-availability rule was met](#7-how-the-ew25-data-availability-rule-was-met)
8. [How the prediction intervals were computed](#8-how-the-prediction-intervals-were-computed)
9. [DOI references](#9-doi-references)

Followed by [results](#results), [setup & reproduction](#setup-and-reproduction), and
[status & deadlines](#status-and-deadlines).

---

## 1. Team, contact and member contributions

- **Team:** Neural Earth
- **Main contact / team leader:** Md Kamruzzaman Kamrul — kamrul28890@gmail.com
  (Mosqlimate platform account: `kamrul28890`)

The work divides into three parts. **Md Kamruzzaman Kamrul led the modeling and analysis —
the core and largest part of the project — and drafted the manuscript**, while the two co-authors
led the supporting data/validation and writing/documentation streams.

| Part | Lead | Scope |
|------|------|-------|
| **1. Modeling, methodology & software** | **Md Kamruzzaman Kamrul** · Purdue University, West Lafayette, IN, USA *(team leader)* | Study design; the full forecasting pipeline (leakage-safe features, WIS/CRPS evaluation harness, baselines, LightGBM + CQR, GRU deep ensemble, mechanistic model, Vincentization ensemble, conformal recalibration); model selection & formal analysis; submission packaging; figures; original manuscript draft. |
| **2. Data curation, validation & reproducibility** | **Abdullah Al Helal** · Oklahoma State University, Stillwater, OK, USA | Dataset assembly & integrity checks; EW‑25 leakage-compliance and interval-validity verification; reproducibility and test suite; manuscript review & editing. |
| **3. Manuscript, literature & documentation** | **Eashraque Jahan Easha** · University of Denver, Denver, CO, USA | Literature review & references; results reporting and figure narrative; repository documentation; manuscript review & editing. |

**Contributions (CRediT taxonomy):**
- **Md Kamruzzaman Kamrul** — conceptualization, methodology, software, formal analysis,
  investigation, visualization, writing (original draft), project administration.
- **Abdullah Al Helal** — data curation, validation, software (testing & reproducibility),
  writing (review & editing).
- **Eashraque Jahan Easha** — investigation (literature review), validation, visualization,
  writing (review & editing).

---

## 2. Repository structure

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

Additional documentation: `MODEL_CARD.md` (condensed model card), `reports/modeling_results_report.pdf`
(full methodology & results), `paper/imdc_paper.pdf` (manuscript), `docs/` (plans and roadmap).

---

## 3. Libraries and dependencies

Python 3.10 (conda environment). Core libraries:

- **Data & numerics:** pandas, numpy, scipy, pyarrow
- **Modeling:** **LightGBM** (quantile regression), **PyTorch** (GRU), scikit-learn, statsmodels
- **Geospatial:** geopandas
- **Epi/challenge:** epiweeks, **mosqlient** (Mosqlimate API client for registration/upload)
- **Reporting:** matplotlib, pymupdf, nbconvert; tectonic for LaTeX

Exact, pinned versions are in `environment.lock.yml` (full `conda env export`) and
`requirements-freeze.txt` (`pip freeze`); the minimal portable spec is `environment.yml`.
Setup instructions are in [Setup and reproduction](#setup-and-reproduction).

> **Runtime note:** set `KMP_DUPLICATE_LIB_OK=TRUE` whenever LightGBM/XGBoost and PyTorch are
> imported in the same process (each links its own libomp). All `run_*` scripts set it automatically.

---

## 4. Data and variables

All data are from the **official IMDC 2026 dataset** (`data/raw/data_imdc_2026/`, tracked via Git
LFS and fingerprinted in `data/raw/data_imdc_2026/CHECKSUMS.sha256`) and public sources. **No
private or additional non-shared datasets were used.** Inputs:

- **Case counts** — weekly dengue and chikungunya cases per municipality (SINAN / Infodengue),
  aggregated to the 26 mandatory states (Espírito Santo excluded, per the rules) and to the target
  cities.
- **Climate** — ERA5 reanalysis (observed) and ECMWF **seasonal forecasts** (future covariate).
- **Ocean/climate indices** — ENSO, IOD, PDO.
- **Static covariates** — population, Köppen climate / biome classification, health-region geometries.
- **Search-access signal** — Afya Whitebook.

Models work in **log1p-incidence space** (cases per population). Engineered features
(`src/imdc/features/panel.py`): autoregressive case lags, rolling statistics, a same-epiweek
seasonal anchor, calendar harmonics, population-weighted state climate summaries, ocean-index lags,
and static climate-zone composition. All inputs are redistributable within the challenge; the raw
archive is included via Git LFS for reproduction.

---

## 5. Data usage restriction

Per the challenge's Data Usage Policy, any dataset not available via the official FTP server or the
Mosqlimate API must be submitted to the organizers so it can be shared with all participants. **We
used only the official IMDC 2026 dataset** distributed via that FTP server (`data/raw/data_imdc_2026/`,
listed in §4) plus its own published `*_update_2026` increments used for the forecast-phase refresh
(`scripts/refresh_data.py`) — no additional, private, or non-shared dataset was introduced at any
point in this pipeline. All inputs are redistributable within the challenge, and the raw archive is
included via Git LFS so the full pipeline is reproducible from the same data without re-downloading.

---

## 6. Model training process

Four submission tracks; the best model is chosen **per track** by fold-1 tuning plus scale-free
relative WIS:

| Track | Model chosen |
|-------|--------------|
| **Dengue, state** | Unweighted quantile-median **ensemble** of climatological + LightGBM + GRU, with conformal recalibration |
| **Chikungunya, state** | **LightGBM** quantile regression (dominates every fold; the ensemble dilutes it) |
| **Dengue / chikungunya, cities** | **Climatological-quantile** model (geography-agnostic; strongest at city level) |

> **What was actually submitted for validation vs. now.** The dengue-state conformal
> recalibration step above was implemented and evaluated (it's what produces the Results table's
> "Ensemble + conformal recalibration" row) but was not wired into the script that generated the
> actual validation-phase submission (fold 1–4, seasons 2022–23…2025–26) — that submission used
> the plain ensemble instead. Caught after the validation phase closed and its results webinar
> (2026-07-31) had already run, so it's documented here rather than corrected on an
> already-presented submission. It **is** applied to the forecast-phase (2026–27) submission via
> `scripts/generate_forecast_2026_27.py`. See `MODEL_CARD.md` §5 and `docs/IMPROVEMENTS.md`.

**Training design.** All models train on a **synthetic-origin panel** (weekly forecast origins ×
horizons 1–67 weeks) with strict leakage-safe cutoff filtering (see §7). Backtests cover the four
official validation seasons (2022–23, 2023–24, 2024–25, 2025–26).

**Individual models.**
- **Baselines** — naive, seasonal-naive, and a climatological-quantile model (empirical quantiles
  of historical same-epiweek incidence).
- **LightGBM** — gradient-boosted quantile regression at the nine required levels, log1p-incidence
  framing, deterministic (`seed=42`), with **conformalized quantile regression (CQR)** calibration.
- **GRU** — a small global recurrent network with a negative-binomial head, run as a **5-member
  deep ensemble** (seeds averaged).
- **Mechanistic** — a compartmental/renewal-style model producing bootstrap trajectories with a
  negative-binomial observation model.
- **Ensemble** — per-quantile median (Vincentization) of the member quantiles, followed by
  **conformal recalibration** of the tails (the single largest accuracy gain; see §8).

Full methodology, per-fold analysis, and the WIS scale-dependence findings are in
`reports/modeling_results_report.pdf` and the manuscript (`paper/imdc_paper.pdf`). Every step is
reproducible with `make reproduce`.

---

## 7. How the EW‑25 data-availability rule was met

The challenge requires that a forecast covering **EW 41 of the current year through EW 40 of the
following year** use **only data available up to EW 25 of the current year**. We enforce this
mechanically, not by convention:

- **Fold boundaries come from the organizers' own flags.** Backtest windows are derived at runtime
  from the `train_N` / `target_N` columns the challenge ships in the case tables
  (`src/imdc/data/folds.py::get_folds`); the last training date is the fold's EW‑25 cutoff. Nothing
  is hardcoded, so the pipeline stays correct if the dataset is refreshed.
- **Every joined table is explicitly date-filtered to the cutoff.** Only the case tables carry fold
  flags; climate, ocean-index, and search-access tables are filtered with `cutoff_filter`
  (`date <= cutoff`, inclusive), which also excludes the ~15-week unflagged gap between each
  training cutoff and the target window. No feature can see a value dated after EW 25.
- **The ECMWF seasonal *forecast* product is filtered by issue date, not target date.** It is a
  legitimate future covariate only when the forecast was *issued* on or before the cutoff, so
  `cutoff_filter_forecasting_climate` keeps rows whose `reference_month <= cutoff month` — never a
  forecast issued later that happens to describe the target period.
- **A hard leakage assertion guards the whole pipeline.** `src/imdc/data/validate.py::assert_no_leakage`
  raises if any training row is dated beyond the cutoff, and `tests/` includes dedicated leakage
  tests so a future refactor cannot silently reintroduce it.
- **The real 2026–27 forecast uses the same machinery.** `get_forecast_origin` sets the forecast
  origin to the latest observed date in the (EW‑25-truncated) dataset, and all covariates pass
  through the same cutoff filters before the forecast horizon (EW 41 2026 → EW 40 2027) is produced.

---

## 8. How the prediction intervals were computed

Every forecast is a **full predictive distribution**: a median plus the required **50/80/90/95%
central intervals**. Uncertainty is produced natively per model and then combined:

- **Climatological** — empirical quantiles of historical same-epiweek incidence.
- **LightGBM** — quantile regression at the nine required levels, with **conformalized quantile
  regression (CQR)** calibration so nominal intervals achieve nominal coverage.
- **GRU** — a negative-binomial predictive head, ensembled over 5 seeds.
- **Mechanistic** — bootstrap trajectories with a negative-binomial observation model.
- **Ensemble** — per-quantile **median (Vincentization)** of the member quantiles, followed by a
  **conformal recalibration** step (multiplicative CQR widening factors tuned on held-out
  validation folds) that corrects residual over-confidence in the tails.

Interval **nesting** (50 ⊂ 80 ⊂ 90 ⊂ 95) and **non-negativity** are enforced, and every file is
validated against the platform schema before upload (`src/imdc/submission/validate.py`).

---

## 9. DOI references

- Araujo EC, Carvalho LM, Coelho FC, et al. Leveraging probabilistic forecasts for dengue
  preparedness and control: The 2024 Dengue Forecasting Sprint in Brazil. *PNAS*
  123(7):e2508989123 (2026). DOI: [10.1073/pnas.2508989123](https://doi.org/10.1073/pnas.2508989123)
- Bracher J, Ray EL, Gneiting T, Reich NG. Evaluating epidemic forecasts in an interval format.
  *PLoS Comput Biol* 17(2):e1008618 (2021).
  DOI: [10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)
- Romano Y, Patterson E, Candès E. Conformalized quantile regression. *NeurIPS* 32 (2019).
  DOI: [10.48550/arXiv.1905.03222](https://doi.org/10.48550/arXiv.1905.03222)

The model repository itself does not yet have an assigned DOI; a Zenodo archive DOI will be minted
for the tagged release accompanying the resulting publication.

---

## Results

Backtest over 26 states × 4 seasons (2022–23 … 2025–26). Mean Weighted Interval Score (WIS) and
official normalized WIS (Σ WIS / Σ cases), both lower-is-better:

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
ensemble's tails is the largest single accuracy gain (WIS 1281 → 1216).

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

### 3. Reproduce
```bash
make reproduce        # all models → ensemble → optional tracks → submissions → figures → manifest
make reproduce-fast   # reuse the committed GRU predictions (skip the ~55-min GRU retrain)
make test             # full test suite, incl. determinism guards
```
Every model's scored predictions are already committed in `results/metrics/*_scored.csv`, so you do
**not** need to re-run anything to inspect or build on the results.

**Reproducibility guarantees.** (1) Determinism — baselines, LightGBM (`seed=42`), and the
mechanistic model reproduce bit-for-bit (`tests/test_determinism.py`); the GRU is seeded but only
bit-reproducible on CPU, so the committed `gru_scored.csv` is canonical. (2) One command — the
`Makefile` runs every step in dependency order. (3) Provenance — `scripts/make_manifest.py` writes
`RESULTS.md` with the git commit, a SHA-256 fingerprint of the raw inputs, and the leaderboards.
(4) Pinned environment — `environment.lock.yml` (exact) and `environment.yml` (portable).
(5) Committed outputs — `results/metrics/*.csv` are the canonical results the paper reads from.

---

## Status and deadlines

| Phase | Status |
|------|--------|
| 0–1. Environment, scaffolding, data audit & EDA | ✅ Done |
| 2. Backtesting harness + baselines | ✅ Done |
| 3. Classical ML (LightGBM + CQR) | ✅ Done |
| 4. Deep learning (GRU + NegBinom) | ✅ Done |
| 5. Mechanistic model | ✅ Done |
| 6. Ensemble + conformal recalibration | ✅ Done |
| 7. Submission packaging + validation | ✅ Done |
| 8. Manuscript | 🟡 Draft |
| Optional tracks (chik-state, dengue/chik cities) | ✅ Done |

- **Validation phase (4 backtest folds):** ✅ submitted (model id 83). See the note in §5 above on
  a conformal-recalibration gap between the documented and actually-submitted dengue-state model.
- **Forecast phase (2026–27 season):** due **2026-09-10**. Raw data re-pulled through EW 25 2026
  (`scripts/refresh_data.py`) and population extrapolated to 2026/2027 (`scripts/extrapolate_population.py`).
  All 77 forecast files generated and locally validated (`scripts/generate_forecast_2026_27.py`);
  upload pending (`scripts/finish_upload.py`, needs `MOSQLIMATE_API_KEY` in `.env`).
