# Model Card — IMDC 2026 submission (Team Neural Earth)

This document is the team's response to the IMDC documentation requirements. Its
sections map one-to-one to the information requested by the Organizing Committee:
team & contributions, repository structure, libraries & dependencies, data & variables,
model training, EW-25 data-availability compliance, prediction-interval computation, and
DOI references. See `README.md` for setup/reproduction and `reports/` + `paper/` for the
full methodology and results.

## 1. Team, main contact, and member contributions
- **Team:** Neural Earth
- **Main contact / team leader:** Md Kamruzzaman Kamrul — kamrul28890@gmail.com
  (Mosqlimate platform account: `kamrul28890`)

The work divides into three parts. **Md Kamruzzaman Kamrul led the modeling and analysis —
the core and largest part of the project — and drafted the manuscript**, while the two
co-authors led the supporting data/validation and writing/documentation streams.

| Part | Lead | Scope |
|------|------|-------|
| **1. Modeling, methodology & software** | **Md Kamruzzaman Kamrul** (Purdue University) | Study design; the full forecasting pipeline — leakage-safe features, the WIS/CRPS evaluation harness, baselines, LightGBM (CQR), the GRU deep ensemble, the mechanistic model, the Vincentization ensemble, and conformal recalibration; model selection and formal analysis; submission packaging; figures; and the original manuscript draft. |
| **2. Data curation, validation & reproducibility** | **Abdullah Al Helal** (Oklahoma State University) | Assembling and integrity-checking the official dataset, verifying EW‑25 leakage compliance and interval validity, the reproducibility and test suite, and manuscript review & editing. |
| **3. Manuscript, literature & documentation** | **Eashraque Jahan Easha** (University of Denver) | Literature review and references, results reporting and figure narrative, repository documentation, and manuscript review & editing. |

CRediT taxonomy:
- **Md Kamruzzaman Kamrul** — Purdue University, West Lafayette, IN, USA —
  conceptualization, methodology, software, formal analysis, investigation, visualization,
  writing (original draft), and project administration.
- **Abdullah Al Helal** — Oklahoma State University, Stillwater, OK, USA —
  data curation, validation, software (testing & reproducibility), and writing (review & editing).
- **Eashraque Jahan Easha** — University of Denver, Denver, CO, USA —
  investigation (literature review), validation, visualization, and writing (review & editing).

## 2. Repository structure
- `src/imdc/` — installable pipeline: data loading/folds/leakage guards, features, evaluation
  harness (WIS/CRPS), models (baselines, LightGBM, GRU, mechanistic, ensemble), submission tools.
- `results/metrics/` — scored backtest predictions and leaderboards (canonical outputs).
- `submissions/validation/` — platform-ready forecast files per track/season/geography.
- `tests/` — automated tests (leakage, scoring correctness, determinism, every model).
- `reports/`, `paper/` — methodology + results (compiled PDFs); `docs/PLAN.md` — full roadmap.
- `Makefile` — `make reproduce` regenerates every result from raw data.

## 3. Libraries and dependencies
Python 3.10 (conda). Core: pandas, numpy, scipy, scikit-learn, statsmodels, **LightGBM**,
**PyTorch**, geopandas, epiweeks, **mosqlient**. Exact versions in `environment.lock.yml`;
portable spec in `environment.yml`; setup in `README.md`. Runtime note: set
`KMP_DUPLICATE_LIB_OK=TRUE` (LightGBM and PyTorch each link libomp).

## 4. Data and variables
Official IMDC 2026 dataset (`data/raw/data_imdc_2026/`, tracked via Git LFS, fingerprinted in
`data/raw/data_imdc_2026/CHECKSUMS.sha256`): weekly dengue/chikungunya case counts per
municipality (SINAN/Infodengue), ERA5 climate reanalysis + ECMWF seasonal forecasts, ENSO/IOD/PDO
ocean indices, population, Köppen/biome classification, health-region geometries, and the Afya
Whitebook search-access signal. Cases are aggregated to the 26 mandatory states (Espírito Santo
excluded); models work in log1p-incidence space. Features: autoregressive lags, rolling statistics,
a same-epiweek seasonal anchor, calendar harmonics, population-weighted state climate summaries,
ocean-index lags, and static climate-zone composition.

All data are from the official IMDC 2026 repository and public sources (Infodengue/SINAN, ERA5,
ECMWF, DATASUS, Afya). No private or additional non-shared datasets were used. All inputs are
redistributable within the challenge; the raw archive is included via Git LFS for reproduction.

## 5. Model training
Four submission tracks, best model chosen per track by fold-1 tuning + scale-free relative WIS:
- **Dengue, state:** unweighted quantile-median **ensemble** of climatological, LightGBM, and GRU.
- **Chikungunya, state:** **LightGBM** quantile regression (dominates every fold; ensemble dilutes it).
- **Dengue/chikungunya, cities:** **climatological-quantile** model (geography-agnostic; strongest at city level).

Training uses a synthetic-origin panel (weekly origins × horizons 1–67) with strict
leakage-safe cutoff filtering (see §6). Backtests cover four seasons (2022–23 … 2025–26). LightGBM
is deterministic (`seed=42`); the GRU is a 5-member deep ensemble. Full details in
`reports/modeling_results_report.pdf`.

## 6. Data-availability compliance (only data up to EW 25 used)
The challenge requires that a forecast covering EW 41 of the current year through EW 40 of the
following year use **only data available up to EW 25 of the current year**. We enforce this
mechanically, not by convention:

- **Fold boundaries come from the organizers' own flags.** Backtest windows are derived at
  runtime from the `train_N` / `target_N` columns the challenge ships in the case tables
  (`src/imdc/data/folds.py::get_folds`); the last training date is the fold's EW-25 cutoff. Nothing
  is hardcoded, so the pipeline stays correct if the dataset is refreshed.
- **Every joined table is explicitly date-filtered to the cutoff.** Only the case tables carry
  fold flags. Climate, ocean-index, and search-access tables are filtered with
  `cutoff_filter` (`date <= cutoff`, inclusive), which also excludes the ~15-week unflagged gap
  between each training cutoff and the target window. No feature can see a value dated after EW 25.
- **The ECMWF seasonal *forecast* product is filtered by issue date, not target date.** It is a
  legitimate future covariate only when the forecast was *issued* on or before the cutoff, so
  `cutoff_filter_forecasting_climate` keeps rows whose `reference_month <= cutoff month` — never a
  forecast issued later that happens to describe the target period.
- **A hard leakage assertion guards the whole pipeline.** `src/imdc/data/validate.py::assert_no_leakage`
  raises if any training row is dated beyond the cutoff, and `tests/` includes dedicated leakage
  tests so a future refactor cannot silently reintroduce it.
- **The real 2026–27 forecast uses the same machinery.** `get_forecast_origin` sets the forecast
  origin to the latest observed date in the (EW-25-truncated) dataset, and all covariates pass
  through the same cutoff filters before the forecast horizon (EW 41 2026 → EW 40 2027) is produced.

## 7. How the prediction intervals were computed
Every forecast is a full predictive distribution: a median plus the required 50/80/90/95% central
intervals. Uncertainty is produced natively per model and then combined:
- **Climatological** — empirical quantiles of historical same-epiweek incidence.
- **LightGBM** — quantile regression at the nine required levels, with **conformalized quantile
  regression (CQR)** calibration so nominal intervals achieve nominal coverage.
- **GRU** — a negative-binomial predictive head, ensembled over 5 seeds.
- **Mechanistic** — bootstrap trajectories with a negative-binomial observation model.
- **Ensemble** — per-quantile median (Vincentization) of the member quantiles, followed by a
  **conformal recalibration** step (multiplicative CQR widening factors tuned on held-out
  validation folds) that corrects residual over-confidence in the tails.

Interval nesting (50 ⊂ 80 ⊂ 90 ⊂ 95) and non-negativity are enforced, and every file is validated
against the platform schema before upload (`src/imdc/submission/validate.py`).

## 8. DOI references
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
