# IMDC 2026 — Modeling, Ensembling, Submission & Paper (Phases 2–8)

## Context

Phase 0 (environment/scaffolding) and Phase 1 (data audit + EDA, written up in `reports/data_findings_report.pdf`) are complete and verified — see that report for the full EDA. This plan covers everything from here through a submitted PNAS-style paper: a WIS-based backtesting harness, baseline/ML/DL/mechanistic models, an ensemble, the official competition submission, and the manuscript itself.

I independently verified the technical claims below by reading the installed `mosqlient==2.5.2` source directly (not assuming from docs):
- `compute_wis`/`compute_interval_score` in `mosqlient/scoring/score.py` match the standard WIS formula exactly — our harness's numbers should agree with the platform's own scorer.
- `mosqlient.registry.models.Model` has only `.get()`, no `.post()` — **model registration is web-UI only**, not an API call. Only `Prediction` has `.post()`.
- `mosqlient.forecast.ensemble_vincentization` exists and takes the **per-quantile median across models** (not mean) — reuse it rather than reimplementing.
- `mosqlient.forecast.baseline.Arima` is a ready-made Box-Cox + `auto_arima` (via `pmdarima`) baseline wrapper whose 4 default alphas (`[0.05,0.1,0.2,0.5]`) map exactly to our required 50/80/90/95 intervals.
- `mosqlient/registry/schema.py`'s `PredictionDataRow` validator enforces exactly the bounds-nesting/non-negativity rule from the official rules, and a separate date-range check uses `epiweeks.Week(year-1,41).startdate()` → `Week(year,40).startdate()` at `freq="W-SUN"` with gap detection — confirmed to match our own fold boundaries bit-for-bit (`Week(2022,41).startdate()` = `2022-10-09`, identical to our derived fold-1 `target_start`).
- `mosqlient.datastore.get_episcanner(api_key, uf, disease, year)` exists; **its response schema is a passthrough** (untyped in the client), so the exact field set/granularity must be confirmed with a live smoke-test call, not assumed.
- No `mosqlient` API key is currently configured in this environment — **the user must register at mosqlimate.org and obtain an API key** before Phase 5 (EpiScanner) or Phase 7 (submission upload) can run; this is a blocking prerequisite outside my ability to complete.
- `pmdarima` (needed to reuse `mosqlient`'s `Arima` baseline) is not installed and its numpy-version compatibility with our installed numpy 2.2.6 is uncertain (historically pinned `numpy<2`) — flagged as a risk with a fallback below.

## Cross-cutting decisions (apply to every phase below)

**Fold roles are fixed and not interchangeable:**
- **Fold 1** (train_cutoff 2022-06-19) — dedicated **ensemble-weight-tuning** fold. Cleanest fold: both its train and target windows fully predate the 2024 outlier.
- **Folds 2 & 3** — the **headline backtest comparison** reported in the paper. Fold 2's target window contains the 2024 outlier season (a feature, not a bug — it becomes the "how robust is each model to an extreme season" case study). Fold 3 trains on 2024 as part of its history.
- **Fold 4** (target 2025-10-05 → nominal ~2026-09-27) — **not a backtest fold**. Real data currently ends 2026-03-08, so it's only partially resolved; report it separately as "prospective, outcome pending," never folded into the headline WIS table.
- **Data staleness**: the forecast phase needs training data through EW25 2026 (~mid-June); the currently-audited raw file ends 2026-03-08. **A data re-pull from the FTP/API is required before Phase 7's real forecast run** — do not assume the already-downloaded archive is sufficient by September.
- **No random row-level cross-validation anywhere.** Any internal train/validation split for hyperparameter tuning or early stopping (Phases 3–4) must be a contiguous time-block holdout (last 52 weeks of a training panel), because synthetic-origin panels are heavily autocorrelated — adjacent weekly origins share nearly all their lag history, so random splits leak and produce falsely optimistic internal validation numbers.
- **Process discipline against unconscious overfitting**: during Phase 3–5 development, iterate against fold 1 and the internal block-holdout only; compute full fold-2/fold-3 scores at defined phase-end milestones, not continuously while debugging.

## Phase 2 — Backtesting harness

### 2.1 Metric definitions (exact, verified against `mosqlient.scoring.score`)

Interval score for a `(1-α)` interval `[l,u]`, observation `y`:
`IS_α(l,u,y) = (u-l) + (2/α)·max(0,l-y) + (2/α)·max(0,y-u)`

WIS over intervals `{50,80,90,95}` → `α_k = {0.5,0.2,0.1,0.05}`, median `m`, `w_0=1/2`, `w_k=α_k/2`:
`WIS(F,y) = [w_0·|y-m| + Σ_k w_k·IS_{α_k}(l_k,u_k,y)] / (K+0.5)`

**CRPS identity** (implementation strategy, not just theory): `IS_α(l,u,y) = (2/α)·[PB_{α/2}(l,y) + PB_{1-α/2}(u,y)]` where `PB_τ(q,y)=(y-q)(τ-1{y<q})` is pinball loss. So `WIS` is exactly the mean pinball loss over the 9 `QUANTILE_LEVELS` already defined in `config.py` — build `wis()` as a thin wrapper around a `pinball_loss(y,q,tau)` primitive evaluated at those 9 levels, so the CRPS-approximation property is structural.

Also: empirical coverage per nominal level; pinball loss per level (standalone diagnostic); MASE (`m=52`, denominator from the fold's training series only); peak-timing error (`argmax_week(median) - argmax_week(observed)`, in weeks) and peak-magnitude error (`log(pred_median_at_true_peak_week / observed_peak)`, evaluated at the *true* peak week, reported as a pair not conflated); relative-WIS vs. baseline (Cramer et al. 2022 pairwise geometric-mean-ratio machinery — reduces to a simple mean-ratio here since every model forecasts every state×fold, but implement the general form since it's the literature standard and cheap).

**WIS decomposition** (dispersion/overprediction/underprediction per interval) — not in `mosqlient.scoring`, but straightforward from the components above and directly feeds the paper's 2024-outlier case-study figure.

### 2.2 Harness API (fills existing empty scaffolds in `src/imdc/evaluation/`, `src/imdc/features/`, `src/imdc/models/`, `src/imdc/submission/`)

- `imdc/evaluation/metrics.py`: `pinball_loss`, `interval_score`, `wis_from_quantiles`, `wis_decomposition`, `coverage`, `mase`, `peak_timing_error`, `peak_magnitude_error`, `relative_wis`.
- `imdc/evaluation/harness.py`: `run_backtest(model, folds, geography, disease) -> long predictions df`, `score_backtest(predictions_df, observed_df) -> scored df`, `summarize(scored_df, by=[...])`.
- `imdc/evaluation/baselines.py`: `NaiveModel`, `SeasonalNaiveModel`, `ClimatologicalQuantileModel`.
- `imdc/evaluation/postprocess.py`: `enforce_monotonicity` (sort-and-reassign, Chernozhukov/Fernández-Val/Galichon 2010), `to_submission_wide`.

**Model protocol** (every model family implements this): `fit(train_long, fold, covariates) -> self`, `predict(horizons, quantile_levels) -> long_df[uf, date, horizon, quantile, value]`. The harness — not individual models — applies `imdc.data.folds.cutoff_filter`/`cutoff_filter_forecasting_climate` to every covariate table before handing it to a model, and runs `imdc.data.validate.assert_no_leakage`/`assert_gap_weeks_absent` as a hard guard. This is the single most important harness design choice: no model implementation re-derives its own cutoff logic, since that's exactly how the confirmed 15-week-gap leak happens.

**Canonical long-format schema** (everything else builds on this): `model, disease, geography_type, uf_or_geocode, fold_id, origin_date, target_date, horizon_weeks, quantile_level, predicted_value, observed_value`. Scored table: `model, disease, geography, fold_id, target_date, horizon_weeks, wis, mae, ae_median, coverage_50/80/90/95, dispersion, overprediction, underprediction`. The wide submission format is produced exactly once, centrally, by `postprocess.to_submission_wide` — scoring never touches wide format directly.

### 2.3 Baseline models (the harness's first client — must work before anything else is trusted)

- **Naive**: median = last observed value at `train_cutoff`; quantile spread via horizon-adaptive empirical quantiles of historical `h`-step differences per state (COVIDhub-baseline style).
- **Seasonal-naive**: median = **median** (not mean — robust to the 2024 outlier automatically) of training-history observations in the same epiweek across prior years; spread from empirical quantiles of (observed − seasonal median) pooled across training epiweeks.
- **Climatological-quantile** (build/validate this first): per state and epiweek `w`, pool historical observations from epiweek `w±2` (circular) across the training window; take the 9 `QUANTILE_LEVELS` as direct order-statistic empirical quantiles on raw counts (no transform needed — empirical quantiles commute with monotonic transforms). Forecast is horizon-invariant by design. 2024 widens fold-3-onward upper quantiles appropriately — this is intended behavior, not a bug to fix.

### 2.4 Testing strategy

**Hand-computed toy test** (K=2 intervals, `pred=10, lower_50=6, upper_50=14, lower_90=2, upper_90=20, y=15`): median term `2.5`; 50%-interval `IS=12` weighted `3`; 90%-interval `IS=18` weighted `0.9`; `WIS=(2.5+3+0.9)/2.5=2.56`. **Independent cross-check via the pinball-loss identity** at quantiles `{0.05,0.25,0.5,0.75,0.95}={2,6,10,14,20}`: pinball losses `0.65,2.25,2.5,0.75,0.25`, sum `6.4`, `/2.5=2.56` — matches. Assert both in one pytest test.

Since `mosqlient.scoring` needs `scoringrules`/`altair` (not installed, not worth adding just for a cross-check), vendor a small `_reference_wis_bracher()` copy directly in the test file (cite Bracher et al. 2021 and `mosqlient.scoring.score`) instead of adding two heavy optional dependencies.

Property tests: WIS reduces to `|y-median|` with zero intervals; reflection symmetry; Monte Carlo coverage sanity (simulate `y~NegBinom`, quantile-forecast from the true distribution, verify empirical coverage converges to nominal ±3pp over 5000 draws). Integration test: `ClimatologicalQuantileModel` end-to-end on fold 1, 3-state subset — schema/shape correctness, monotonic quantiles, finite/non-negative WIS, and a coarse sanity check that it beats naive on a strongly-seasonal state.

## Phase 3 — Classical ML (XGBoost/LightGBM)

### 3.1 Synthetic-origin panel (the key prerequisite)

Only 4 true forecast origins per fold exist — far too few for gradient boosting. Build a rolling panel: every Sunday from ~2014 (after enough warmup for `lag_52`) through a fold's own `train_cutoff` becomes a candidate origin; for each origin × horizon `h=1..~67`, compute features "as of" that origin and the realized value `h` weeks later as one training row. This yields ~26 states × ~400–500 origins × ~67 horizons — enough for boosted trees, while *evaluation* stays strictly the 4 official windows. Every synthetic origin/label must respect **its own fold's** `train_cutoff` (fold 2's panel legitimately being a superset of fold 1's expanding window is fine; no origin may ever exceed its fold's own cutoff).

### 3.2 The gap subtlety (a second, distinct leakage trap beyond the one already caught)

Because of the confirmed 15-week gap, a direct-strategy model trained at one origin must forecast `h=1..~67` (15 gap weeks + ~52 target weeks), then slice out `h≈16..67` for scoring. Lag features must be computed from the **full continuous cutoff-filtered history** (the gap is a withheld-evaluation gap simulating real reporting lag, not a missing-observation gap — the data exists, it's just excluded from `train_N`/`target_N`). Using `target_window()` output to build lag features would silently produce wrong/NaN lags. **Add an explicit regression test for this**, analogous to the existing `assert_gap_weeks_absent` test.

### 3.3 Feature engineering

- **Target framing**: model `log1p(incidence per 100k)` internally (stabilizes scale across SP-vs-small-state variance, tames the EDA's extreme right-skew); convert back via `expm1` × target-year population, clip at 0 for submission.
- **Autoregressive**: `lag_1..4`, `lag_52` (encodes the seasonal anchor directly — more robust than harmonics alone given the EDA's 10-week peak-timing IQR spread), rolling mean/std (4wk, 8wk), rolling max (12wk), all on log1p(incidence) from continuous cutoff-filtered history.
- **Calendar/harmonic**: target-epiweek (of the date being forecast, not the origin — must be supplied explicitly) as raw integer + sin/cos at 1× and 2× annual frequency (2nd harmonic for the sharp single peak found in EDA), plus `horizon` as its own feature.
- **Climate**: population-weighted state ERA5 rolling means/anomalies-vs-epiweek-climatology up to `origin_date` only; `forecasting_climate.csv.gz` (genuine future signal) via the existing `cutoff_filter_forecasting_climate`, valid for `h≤26`, falling back to climatological normals beyond that — build a feature-availability-by-horizon table as a first deliverable. Rather than hard-coding the EDA's raw-correlation lags (+12/+8/+4 weeks, flagged there as confounded by shared seasonality), feed a small candidate set of lags alongside the calendar-harmonic features and let regularized tree feature selection find the residual signal; the pending deseasonalized cross-correlation re-check remains worth doing as a fast side analysis to sanity-check resulting feature importances, not to gate this phase.
- **Ocean indices**: national-level origin-anchored lags (4/8/12/26/52wk) — EDA found only weak national-annual correlation untested regionally; treat as a stretch feature, not core.
- **Access_afya**: only for `h∈{1,2}` (matches the EDA's contemporaneous-correlation finding); explicitly NaN (both LightGBM/XGBoost handle natively) for `h≥3` rather than passing stale values.
- **Static**: `log(population)`, state-level Koppen represented as a population-weighted fraction-of-state-population-per-class vector (not a single dominant label) — directly usable given the EDA's near-monotone Koppen stratification, zero leakage risk (time-invariant).

### 3.4 Multi-horizon strategy and quantile approach

**Direct, not recursive**: the real deployment forecast is genuinely single-shot with no intermediate true observations to recurse on; recursive would compound 67 steps of predicted-on-predicted error in a way that doesn't match real deployment. **One pooled model across all states and horizons** with `horizon` and target-epiweek as explicit features (not 67 independent per-horizon models) — with only ~500 origins/state, per-(state,horizon) data is too thin; pooling regularizes and learns a smooth horizon-effect curve.

**Quantiles**: primary = LightGBM `objective="quantile"`, one run per quantile level (9 total, shared features/hyperparameters). Secondary ablation = XGBoost 3.x's native multi-quantile `reg:quantileerror` (single model, `quantile_alpha` array) — worth reporting because a shared-tree model should structurally cross quantiles less often; log `frac_rows_needing_reordering` per model as a small, concrete, reportable calibration finding for the SI. **Monotonicity enforcement is centralized** in the harness's `postprocess.enforce_monotonicity`, never left to individual model code.

### 3.5 Hyperparameter tuning without overfitting to 4 folds

Do not tune against the 4 official folds. Use fold 2's synthetic panel specifically (most "normal," pre-2024-outlier training window), its own last 52 weeks as an internal contiguous holdout, for a **single, one-time, ≤20-config grid search** over a narrow space (`num_leaves` 15–31, `max_depth` 4–6, `min_data_in_leaf` 20–50, `learning_rate` 0.03–0.1 with early stopping, `subsample`/`colsample` 0.7–0.9). Freeze the result and reuse unchanged across every fold's own training panel — tuning per fold would leak fold-specific structure into what's supposed to be an unbiased comparison.

## Phase 4 — Deep learning

### 4.1 Architecture: small, custom, raw PyTorch — not a framework, not a large model

Skip NeuralForecast/Darts-style frameworks (no new heavy dependency; "torch only" is the standing decision and the scale doesn't warrant the extra surface). **Global 1–2 layer GRU encoder** (hidden size 32–64, tens of thousands of parameters), trained jointly across the 26 states, with a small learned per-state embedding (26×8) and the same static Koppen-fraction/log-population features from Phase 3 concatenated at every timestep. Decoder: MQ-RNN/DeepAR-direct style — encoder's final hidden state + horizon embedding (or the same sin/cos target-epiweek + horizon features) → small shared MLP head evaluated once per horizon, all horizons' losses summed per origin (one encoder pass, ~67 lightweight MLP evaluations — cheap even on an M1 CPU/MPS). **Explicitly not a Transformer**: the synthetic-origin panel gives on the order of 10–20k (state, origin) examples, heavily autocorrelated (smaller effective sample size); Monash/M4/M5 literature consistently shows simple statistical/GBM methods beating large nets on short, few-series panels — DL only tends to win with hundreds+ of series.

### 4.2 Loss and interval construction

**Negative Binomial** likelihood (not Poisson — underdispersed for right-skewed epidemic counts with 2024 present; not Gaussian/MSE on log-counts — breaks discrete-count interpretation). Network outputs `(log_mu, log_alpha)`; `mu = population · exp(f(x))` (standard epi GLM-offset trick, predicts log-incidence-rate while the loss is evaluated in count space). Quantiles come **analytically** from `scipy.stats.nbinom.ppf(mu, alpha)` — no Monte Carlo. Train a **small deep ensemble** (5–10 seeds, each likely under a minute on the M1 given the tiny model/data) and pool NegBinom quantile predictions — this pooled ensemble is what's actually submitted, not a single seed; deep ensembles reliably improve calibration on small noisy data at near-zero extra design cost.

### 4.3 Honest risk assessment

DL carries the **highest** risk, among all five model families, of looking good on internal validation but failing to generalize — it has the most capacity to fit the panel's autocorrelated noise. Mitigations: same block-holdout/frozen-hyperparameter discipline as Phase 3, plus dropout/weight decay/early stopping. Treat DL losing to LightGBM or even the climatological baseline as an **acceptable, expected, still-publishable** outcome — a small-data-regime DL-underperforms finding is itself a legitimate result, consistent with M4/M5 post-mortems. The real hedge is structural: Phase 6's ensemble weighting naturally down-weights or excludes a genuinely weaker family, so DL underperforming individually poses no risk to the "best possible models" goal or the paper's validity.

## Phase 5 — Mechanistic / semi-mechanistic modeling

### 5.1 Prerequisite and data-shape caveat

**Blocked on the user obtaining a `mosqlient` API key** (register at mosqlimate.org — I cannot do this). Run a Step-0 smoke test (`get_episcanner(api_key, uf="RJ", disease="dengue", year=2023)`) early to confirm response schema/granularity, since the client-side schema is an untyped passthrough — do not assume municipality-level output until confirmed live.

### 5.2 Design: reuse EpiScanner's fitted curves as a bootstrap ensemble, don't refit a fresh renewal equation

Aggregate EpiScanner outputs to state level via population-weighted means per year, using only years with a completed season before `fold.train_cutoff` (leakage-safe by construction — EpiScanner needs a *completed* season to fit). Rather than numerically integrating a fresh SIR/renewal equation (unnecessary complexity for the timebox, and this dataset lacks serotype/immunity data needed for proper susceptible-depletion at state level): for `B=500–1000` bootstrap draws, resample a historical year's state-level weekly incidence curve, epiweek-realigned using that year's EpiScanner timing estimate, optionally reweighted toward historically fast/high-R0 years if early-season signal at the forecast origin suggests a fast start. Layer a modest NegBinom observation-noise term (dispersion calibrated once from training-residual week-to-week deviation) on each bootstrap trajectory — without this, ~10–14 distinct historical curves would produce implausibly lumpy quantiles. Empirical quantiles across the `B` noised trajectories at each target week become the submission intervals.

**2024 handling, deliberate**: for folds whose training window includes the completed 2024 season (folds 3, 4), include 2024 in the resampling pool as an ordinary, equally-weighted draw — this is where realistic outbreak-year tail behavior should enter the uncertainty quantification, a genuinely attractive property relative to a tree model with no explicit outbreak indicator.

### 5.3 Hard timebox and fallback

After ~1 week (~15–20% of total modeling time): if fold-1/fold-2 relative WIS isn't at least broadly competitive with the climatological baseline, drop it from the ensemble candidate pool rather than continuing to debug — its paper value (a genuinely different, interpretable comparison) is realized by running the comparison honestly, not by winning. Note in the paper's limitations: municipality-level EpiScanner aggregated to state level loses sub-state asynchrony in epidemic timing, an acceptable simplification given the mandatory target is state-level.

## Phase 6 — Ensemble

**Primary/default**: `mosqlient.forecast.ensemble_vincentization` — per-quantile **median** across models (verified from source; note precisely: median, not mean). Forecast-hub literature (COVID Forecast Hub, FluSight) consistently finds unweighted quantile combination a strong, hard-to-beat default, and with only one fold available to validate anything fancier, extra weight-tuning carries real overfitting risk.

**Secondary comparisons**: inverse-WIS weighting (`weight_m ∝ (1/mean_WIS_m)^p`); QRA/stacking (per-quantile-level constrained non-negative sum-to-one regression, pooled across states within the tuning fold — not per-state, too thin) as an exploratory, higher-flexibility/higher-overfitting-risk alternative.

**Weight tuning discipline**: fit any weights **only on fold 1**, freeze, apply unchanged to folds 2–3 (headline) and fold 4 (real forecast). Default to unweighted Vincentization unless a fancier method shows a clear, fold-1-validated, qualitatively sensible improvement (not a single-model-dominates pattern suggesting fold-1-noise fitting). **Required SI robustness check**: refit weights using an alternative tuning source (e.g., leave-one-fold-out across 1–3) and show the headline comparison is qualitatively stable — preempts the most likely reviewer objection to a 1-fold-tuned ensemble.

## Phase 7 — Submission packaging

### 7.1 Builder/validator (`src/imdc/submission/`)

`build.py::build_submission_frame(long_quantile_df, geography, disease, season_year)` pivots to the wide format using `epiweeks.Week(year-1,41).startdate()` → `Week(year,40).startdate()` (already verified to match `mosqlient`'s own validator and our fold boundaries exactly). `validate.py` locally replicates the platform's checks (schema columns, monotonic nesting via `enforce_monotonicity` as a fail-safe re-check, non-negativity, zero date gaps, full 26/26-state or full optional-city completeness) **before any network call**, so failures are free rather than burning an API round-trip. `registry.py` wraps `mosqlient.registry.Prediction.post(...)` / `validate_prediction(...)`; captures the git commit hash via `git rev-parse HEAD` at call time and **refuses to upload if `git status --porcelain` is non-empty** — otherwise "references a commit hash" doesn't actually guarantee the hash matches the code that produced the numbers.

### 7.2 Repo packaging and registration

**Model registration has no API path in this client version — it's a one-time web-UI step at mosqlimate.org** (confirmed: `Model` class exposes only `.get()`). Because the platform keys a model by disease + adm_level + category, expect up to **4 separate registered models** for one methodology (mandatory dengue-state, optional dengue-city, chikungunya-state, chikungunya-city) — plan registration time accordingly, and note this requires the user's account/action, not something I can do.

Package the final submission as a **separate, lean public repo** (`3rd_imdc_{institution}_{team_name}`), distinct from this research monorepo: inference code only, pinned `pyproject.toml` (exact versions via `pip freeze`/`conda env export` at freeze time — **pin `mosqlient==2.5.2` exactly**, since several of this plan's guarantees were read directly from that version's source and a silent upgrade could change server-side validation behavior), README following the `sprint-template-2025` layout (`README.md`, `LICENSE`, `pyproject.toml`, `Demo Notebooks/`), a `model_card.md` transparently documenting training window/features/known limitations (2024-outlier handling, fold-4 partial resolution), and committed frozen model artifacts (small enough given the dataset size). Keeps the commit-hash guarantee meaningful and decouples the frozen competition artifact from ongoing paper-writing commits in this repo.

### 7.3 `pmdarima`/statistical-baseline risk

`pmdarima` (needed to reuse `mosqlient.forecast.baseline.Arima`) is not installed; numpy 2.2.6 compatibility is uncertain (pmdarima has historically pinned `numpy<2`). **Attempt the install early** (Phase 2 start, not deferred); if it fails, fall back to `statsmodels.tsa.statespace.SARIMAX` with a small manual `(p,d,q)×(P,D,Q,52)` grid, or Nixtla's `statsforecast.AutoARIMA` (lighter, numpy2-friendly) as a second fallback — either preserves an ARIMA-family comparison point without blocking on a single fragile dependency.

## Phase 8 — Paper (PNAS-style, following Araujo et al. 2026)

### 8.1 Structure, mapped to phase outputs

- **Significance statement** (~150w): headline relative-WIS result (Phase 6, folds 2–3).
- **Abstract** (~250w): problem → 5-model-family + ensemble approach across resolved backtests → headline quantitative + qualitative findings (ensemble robustness to the 2024 outlier; DL underperformance in the small-series regime).
- **Main text** (~3000–4000w): (1) *Introduction* — dengue burden, IMDC framing, gap addressed, Araujo et al. as precedent; (2) *Data & backtesting design* — concise, folds 1–3 roles, WIS evaluation, full detail to SI (Phase 1 EDA + Phase 2 harness); (3) *Model families* — one paragraph each, detail to SI (Phases 3–6); (4) *Results* — Fig 1 relative-WIS by model×fold small-multiples/boxplot across states (Phase 2 `relative_wis`, folds 2–3 only), Fig 2 nominal-vs-empirical coverage diagram per family (Phase 2 `coverage`), Fig 3 2024-outlier case study via WIS decomposition (dispersion/over/under-prediction, mechanistic model's outlier-inclusive bootstrap as narrative anchor), Fig 4 state-level choropleth of ensemble relative-WIS (`geopandas`/`shape_muni.gpkg`, tied to the EDA's Koppen stratification and southward-expansion finding), prospective fold-4/2026-27 results if resolved in time (flagged as a timing risk); (5) *Discussion* — DL underperformance, ensemble robustness value, mechanistic interpretability trade-off, explicit limitations (fold-4 unresolved at writing time, the 15-week operational reporting gap, two-disease/single-country scope), Ministry-of-Health-use implications; (6) *Materials & Methods* — brief, SI-pointer.
- **SI Appendix**: full WIS/CRPS math, full feature list/hyperparameters, DL architecture + training curves, EpiScanner derivation + per-state parameter tables, full ensemble weight table + the leave-one-fold-out robustness check, per-state/per-fold score tables, data/code availability (research repo + up to 4 competition submission repos), reproducibility statement.

### 8.2 Sequencing (today: 2026-07-02; forecast deadline 2026-09-10)

- **Jul 2–20**: Phase 2 harness + baselines (climatological validated first), Phase 3 classical ML, run on the 4 folds.
- **Jul 20–Aug 3**: Phase 4 DL and Phase 5 mechanistic in parallel (mutually independent); Phase 6 ensemble work starts as soon as ≥2–3 families have fold-1 results.
- **Aug 3–7**: freeze the candidate model/ensemble set for the competition submission — leave real buffer before Sept 10.
- **Aug 7–21**: Phase 7 packaging; upload folds 1–4 backtest predictions as soon as models are frozen (don't wait for Sept 10 — the platform's own scoring becomes an independent cross-check on the harness). **Re-pull raw data through EW25 2026 here** (per the data-staleness note above) before generating the true forecast.
- **Aug 21–Sept 5**: generate/validate the true forecast-phase submission — literally "fold 5" through the same harness/model code with `train_cutoff = EW25 2026`, the direct payoff of a generic harness. Target internal-complete Sept 3–5, upload by Sept 8, leave Sept 9–10 as slack.
- **Paper writing starts in July, not after Sept 10**: Introduction/Data/Methods don't depend on final results and should be drafted once Phase 2/EDA are stable (~late July). Only Results/Discussion/Abstract/Significance wait on frozen results.
- **October webinars**: target a complete first full draft before the confirmed October webinar date (forcing function + potential comparanda from discussion); internal full-draft deadline ~1 week before the webinar, revision pass after, then PNAS submission.

## Consolidated risk list

1. Data staleness (raw data ends 2026-03-08, forecast phase needs EW25 2026) — required refresh step, not assumable.
2. Fold-4 unresolvable in time — resolved by keeping it out of the headline table; must be stated plainly in the paper.
3. WIS scale-dependence across states (SP dominates any naive pooled mean) — primary cross-state metric is relative-WIS vs. baseline; compute headline WIS on raw counts (matches how the platform actually scores), incidence-scale as SI robustness only.
4. One-fold ensemble weight tuning is thin — mitigated by the required leave-one-fold-out SI check.
5. Synthetic-origin panel non-independence inflates apparent sample size — mitigated by the hard time-block-holdout-only rule.
6. EpiScanner is a network/API-key dependency unique among all phases, with an unconfirmed response schema — smoke-test and cache aggressively early; **user must obtain an API key first**.
7. `mosqlient` version drift — pin `mosqlient==2.5.2` exactly.
8. `pmdarima`/numpy2 compatibility uncertain — attempt early, have `statsmodels`/`statsforecast` fallbacks ready.

## Verification

- Every metric in `imdc/evaluation/metrics.py` gets the hand-computed toy test (§2.4) plus property tests, before any model touches the harness.
- `pytest tests/` must stay green throughout; new tests accompany each phase (gap-leakage regression test for Phase 3's lag-feature trap, submission-format validator tests for Phase 7).
- Each model family's backtest results land in `results/metrics/` as the canonical long-format table; sanity-check WIS improves (or degrades explicably) baselines → ML → DL/mechanistic → ensemble, and no model wildly underperforms seasonal-naive without a documented reason.
- Before any real `mosqlient` upload, run the local validator against a dummy prediction end-to-end (Phase 2/7) to catch format issues for free.
- Figures for the paper are regenerated from `results/metrics` + `results/figures` only, never hand-edited, so the pipeline stays reproducible end to end.
