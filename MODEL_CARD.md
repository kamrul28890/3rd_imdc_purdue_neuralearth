# Model Card — IMDC 2026 submission

Documentation required by the IMDC rules for the model repository. See `README.md`
for setup/reproduction and `reports/` + `paper/` for the full methodology and results.

## Team and contributors
- **Team:** Neural Earth
- **Members:**
  - Abdullah Al Helal — Oklahoma State University
  - Md Kamruzzaman Kamrul — Purdue University (team leader)
  - Eashraque Jahan Easha — University of Denver
- **Contact:** kamrul28890@gmail.com (Mosqlimate platform account: `kamrul28890`)

## Repository structure
- `src/imdc/` — installable pipeline: data loading/folds/leakage guards, features, evaluation
  harness (WIS/CRPS), models (baselines, LightGBM, GRU, mechanistic, ensemble), submission tools.
- `results/metrics/` — scored backtest predictions and leaderboards (canonical outputs).
- `submissions/validation/` — platform-ready forecast files per track/season/geography.
- `tests/` — 64 automated tests (leakage, scoring correctness, determinism, every model).
- `reports/`, `paper/` — methodology + results (compiled PDFs); `docs/PLAN.md` — full roadmap.
- `Makefile` — `make reproduce` regenerates every result from raw data.

## Dependencies
Python 3.10 (conda). Core: pandas, numpy, scipy, scikit-learn, statsmodels, **LightGBM**,
**PyTorch**, geopandas, epiweeks, **mosqlient**. Exact versions in `environment.lock.yml`;
portable spec in `environment.yml`; setup in `README.md`. Runtime note: set
`KMP_DUPLICATE_LIB_OK=TRUE` (LightGBM and PyTorch each link libomp).

## Data and variables
Official IMDC 2026 dataset (`data/raw/data_imdc_2026/`, tracked via Git LFS, fingerprinted in
`data/raw/data_imdc_2026/CHECKSUMS.sha256`): weekly dengue/chikungunya case counts per
municipality (SINAN/Infodengue), ERA5 climate reanalysis + ECMWF seasonal forecasts, ENSO/IOD/PDO
ocean indices, population, Köppen/biome classification, health-region geometries, and the Afya
Whitebook search-access signal. Cases are aggregated to the 26 mandatory states (Espírito Santo
excluded); models work in log1p-incidence space. Features: autoregressive lags, rolling statistics,
a same-epiweek seasonal anchor, calendar harmonics, population-weighted state climate summaries,
ocean-index lags, and static climate-zone composition.

## Model training
Four submission tracks, best model chosen per track by fold-1 tuning + scale-free relative WIS:
- **Dengue, state:** unweighted quantile-median **ensemble** of climatological, LightGBM, and GRU.
- **Chikungunya, state:** **LightGBM** quantile regression (dominates every fold; ensemble dilutes it).
- **Dengue/chikungunya, cities:** **climatological-quantile** model (geography-agnostic; strongest at city level).

Training uses a synthetic-origin panel (weekly origins × horizons 1–67) with strict
leakage-safe cutoff filtering. Backtests cover four seasons (2022–23 … 2025–26). LightGBM is
deterministic (`seed=42`); the GRU is a 5-member deep ensemble. Full details in
`reports/modeling_results_report.pdf`.

## Data usage restrictions
All data are from the official IMDC 2026 repository and public sources (Infodengue/SINAN, ERA5,
ECMWF, DATASUS, Afya). No private or additional non-shared datasets were used. All inputs are
redistributable within the challenge; the raw archive is included via Git LFS for reproduction.

## Predictive uncertainty
Every forecast is a full predictive distribution: median plus 50/80/90/95% central intervals, as
required. Uncertainty is produced natively per model (empirical quantiles, LightGBM quantile
regression with **conformalized** calibration, negative-binomial for the GRU, bootstrap-trajectory
+ negative-binomial observation model for the mechanistic model) and combined by per-quantile
median in the ensemble. Interval nesting and non-negativity are enforced and locally validated
against the platform schema before upload (`imdc/submission/validate.py`).

## References
- Araujo EC, Carvalho LM, Coelho FC, et al. Leveraging probabilistic forecasts for dengue
  preparedness and control: The 2024 Dengue Forecasting Sprint in Brazil. *PNAS*
  123(7):e2508989123 (2026).
- Bracher J, et al. Evaluating epidemic forecasts in an interval format. *PLoS Comput Biol* (2021).
- Romano Y, Patterson E, Candès E. Conformalized quantile regression. *NeurIPS* (2019).
