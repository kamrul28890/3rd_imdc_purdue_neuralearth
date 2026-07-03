# Improvement Roadmap — IMDC 2026 pipeline (Team Neural Earth)

A review of the repository (code, models, tests, running) with concrete, prioritized
improvements. Every item lists **what**, **why**, **how**, and a rough **effort**.
Line references are to the state at commit `ce85b7f`.

## How the repo stands today (honest baseline)

Strong foundation: clean package layout (~2,600 LOC), a leakage-safe backtesting harness,
five model families, an ensemble, 64 passing tests, full reproducibility (`make reproduce`,
determinism, provenance manifest), and a complete, validated, mostly-uploaded submission.
The gaps below are the difference between "works and won a deadline" and "a maintainable
research platform we can extend for the September forecast and the paper."

**Top 5 things to fix first (highest value / lowest effort):**
1. ~~Cache the data loaders~~ — **DONE** (§2.1): 25× faster cached loads, mutation-safe.
2. ~~Fix the EW53 season-week collision~~ — **DONE** (§1.1): date-based `season_week_from_date`.
3. ~~Handle all-zero-median forecasts~~ — **DONE** (§1.2): mean point-estimate for degenerate series.
4. ~~Persist trained models~~ — **DONE** (§2.4): `models/persistence.py` save/load (torch CPU-safe).
5. Define a formal `Forecaster` protocol — removes duck-typing and dead params (§3.1).

**Also since implemented:** §1.3 fold-4 flagging (`summarize(exclude_folds=…)`), and §5.1 **ECMWF
seasonal-climate features** — improved LightGBM 1311→1299 and the ensemble to **1280.7** WIS.
Remaining high-value items: §3.1 protocol, §3.2 dedup, §3.3 CLI, §4.1 fast tests, §5.2 hierarchical
reconciliation, §5.3 stacking.

---

## 1. Correctness & robustness

### 1.1 EW53 season-week collision (latent bug) — **P0**
**What:** `mechanistic.py::season_week` maps both EW1 and EW53 to season-week 13
(`season_week(1)==season_week(53)==13`). Years with an ISO/epi week 53 (e.g. 2026) collide,
so EW53 case data overwrites/mis-slots into the EW1 bucket.
**Why it matters:** silently corrupts the mechanistic model's historical trajectories for
53-week seasons and the fold-4 (2025–26) forecast — exactly the season we just submitted.
**How:** make the season index handle 53-week years explicitly — either extend `SEASON_LEN`
to 53 with a proper EW53→53 mapping, or drop EW53 by convention (document it). Add a unit
test asserting the mapping is injective over EW1–EW53.
**Effort:** 1–2 h.

### 1.2 Degenerate all-zero-median forecasts — **P0**
**What:** For sparse series (small cities with ~no chikungunya) the climatological median is
exactly 0 for all 52 weeks. The platform silently rejects these (root cause of the 15 stuck
chik-city uploads).
**Why:** both a data-quality and a modeling issue — an all-zero point forecast with non-zero
intervals is degenerate, and it blocks submission.
**How:** (a) short term, floor the median at a tiny positive value (e.g. `max(pred, 0.5·lower_50_nonzero)`
or the historical mean) when the whole median series is zero; (b) better, use a proper
zero-inflated / hurdle model or the mean instead of the median for near-zero series. Add a
submission-validator check that flags all-zero-median tables *before* upload.
**Effort:** 2–4 h (short term); 1–2 d (proper model).

### 1.3 Fold-4 is prospective, not validated — **P1**
**What:** the 2025–26 season is incomplete in the data; fold-4 scores in the backtest are on a
truncated window and the submitted fold-4 forecast is unscoreable until the season resolves.
**Why:** any headline number that averages fold-4 is misleading.
**How:** already partially handled (reported separately), but codify it: exclude fold-4 from
`summarize()` headline aggregations by default, add a `resolved_only` flag. Re-score when
refreshed data arrives.
**Effort:** 1–2 h.

### 1.4 GRU is not bit-reproducible (MPS) — **P2**
**What:** documented, but the GRU still varies run-to-run on Apple MPS.
**How:** add a `deterministic=True` model flag that forces CPU + `torch.use_deterministic_algorithms(True)`
+ fixed seeds, for the canonical/paper runs; keep MPS for fast iteration. Persisting weights
(§2.4) also sidesteps this for the submission artifact.
**Effort:** half a day.

---

## 2. Performance & runtime

### 2.1 Redundant data I/O — **P0, biggest single speedup**
**What:** the gzip loaders are called **17×** across a run with **no caching** — `load_cases`
alone is re-read by the harness, then again inside each model's `fit` (`build_panel`,
`_state_incidence`), then *again* in `predict` (`build_prediction_features`). Each is a
multi-hundred-MB gunzip+parse.
**Why:** dominates wall-clock for every model; the GRU's ~55 min and the 10-min test suite are
largely I/O + re-derivation.
**How:** put `@functools.lru_cache` (or a small in-memory registry) on the loaders in
`data/loaders.py`, and pass already-loaded/aggregated frames into models instead of having
each model reload. The parquet cache (`data/processed/`) covers climate but not
cases/ocean/population — extend the same pattern.
**Effort:** 2–4 h. **Expected:** 3–5× faster across the board.

### 2.2 Model `fit` and `predict` re-derive everything — **P1**
**What:** `LGBMQuantileModel.predict` calls `build_prediction_features(self._fold, …)`, which
rebuilds the whole origin-anchored series that `fit` already built; same in the GRU/mechanistic.
**How:** cache the per-fold origin series on the model instance during `fit` and reuse it in
`predict`. Combine with §2.1.
**Effort:** 2–3 h.

### 2.3 Mechanistic `predict` is a Python triple-loop — **P2**
**What:** `mechanistic.py::predict` loops states × weeks × rows and calls `nbinom.rvs`
per cell.
**How:** vectorize — draw all bootstrap trajectories at once (`traj[boot_idx]` is already an
array), compute `mu` as a matrix, and call `nbinom.rvs` once on the full `(n_boot, n_weeks)`
array, then take quantiles along the boot axis.
**Effort:** half a day. **Expected:** ~10× faster mechanistic runs.

### 2.4 No model-weight persistence — **P1**
**What:** nothing saves trained boosters / GRU state / mechanistic trajectories; every
backtest and every forecast retrains from scratch (why regenerating fold-4 needed a full GRU
refit).
**How:** add `save(path)` / `load(path)` to each model (`booster.save_model`,
`torch.save(state_dict)`, `pickle` the trajectory arrays) and a `models/` artifact layout keyed
by (model, disease, fold, commit). Then the September forecast and any re-prediction are
seconds, not an hour.
**Effort:** half a day. High leverage.

### 2.5 No parallelism — **P2**
**What:** 9 LightGBM quantile fits and 4 folds run sequentially; the GRU's 5-member ensemble is
sequential.
**How:** parallelize the per-quantile fits (`joblib.Parallel`) and/or folds. LightGBM already
uses threads, so parallelize at the fold/quantile level with processes carefully (respect
`KMP_DUPLICATE_LIB_OK`).
**Effort:** half a day.

---

## 3. Software engineering & maintainability

### 3.1 No formal model interface; dead params — **P1**
**What:** the fit/predict "protocol" is duck-typed and documented only in a docstring; there
are **3 unused `covariates=None`** params.
**How:** define `class Forecaster(typing.Protocol)` with `fit(train_df, fold) -> Self` and
`predict(target_grid) -> long_df`, type-annotate all models to it, delete the dead params, and
run `mypy` in CI. Makes the contract enforceable and the codebase navigable.
**Effort:** half a day.

### 3.2 Duplicated logic (state vs city, ensemble vs forecast) — **P1**
**What:** `harness.py` (state) and `evaluation/city.py` (city) largely duplicate run/score
logic; `ensemble.py::vincentization` and `submission/forecast.py::_vincentize_wide` both do
per-quantile median (the split exists only because `_KEYS` includes `observed_value`, which
drops NaN groups on the prediction-only path).
**How:** introduce a `Geography` abstraction (state / city) so one harness handles both; unify
the two Vincentization implementations into one that takes explicit `index_cols` and never
assumes `observed_value`.
**Effort:** 1 day.

### 3.3 Seven near-duplicate `run_*.py` scripts — **P2**
**What:** `run_baselines/run_ml/run_dl/run_mechanistic/run_ensemble/run_chikungunya/run_cities`
share ~80% boilerplate.
**How:** one parametrized CLI: `python -m imdc.run --stage backtest --model lgbm --disease dengue`
(argparse or Typer), with a registry mapping model names → factories. Collapses 7 files into
one + a table.
**Effort:** half a day.

### 3.4 Scattered configuration — **P2**
**What:** hyperparameters (`DEFAULT_PARAMS`), feature lists (`FEATURE_COLS`), ensemble members
(`MEMBERS`), and track configs live in different modules as literals.
**How:** centralize experiment config in dataclasses (or one `configs/*.yaml`), so a run is
fully described by a config object that also gets stamped into the provenance manifest.
**Effort:** 1 day.

### 3.5 No CI, no linting, no logging — **P2**
**How:** add a GitHub Actions workflow (`pytest -m "not slow"`, `ruff`, `mypy` on push); adopt
`ruff` for formatting/lint; replace `print` in scripts with the `logging` module.
**Effort:** half a day (CI + ruff), ongoing.

---

## 4. Testing

Current suite is solid on **leakage, WIS correctness, determinism, and per-model smoke**, but:

### 4.1 Tests are integration-heavy and slow (10 min) — **P1**
**What:** most tests hit real gzip data and train real models.
**How:** add a `tests/fixtures/` synthetic mini-dataset (a few states, ~150 weeks) and unit
tests that run in milliseconds; mark the real-data ones `@pytest.mark.slow` and run only fast
ones on every push, slow ones nightly. Pairs with §2.1 to cut time further.
**Effort:** 1 day.

### 4.2 Coverage gaps — **P2**
Missing tests for: the submission **upload** path (mock `mosqlient`), the **full-season
forecast** path (`forecast.py`), the **city ensemble**, and a **regression guard** asserting the
ensemble's headline WIS stays within a band (catches accidental degradation). Add `pytest-cov`
and a coverage floor.
**Effort:** 1 day.

### 4.3 Property-based tests — **P3**
Use `hypothesis` for metrics (WIS ≥ 0, monotone in error), monotonicity enforcement (output
always sorted), and submission validation (round-trip).
**Effort:** half a day.

---

## 5. Modeling & forecast accuracy (for September + the paper)

### 5.1 Use the ECMWF seasonal climate forecast — **P1**
**What:** `forecasting_climate.csv.gz` (genuine future climate, ≤6 months) is loaded nowhere in
the feature pipeline; only origin-anchored reanalysis is used.
**Why:** it's the one legitimate source of *future* covariate signal for horizons ≤26 — likely
the biggest available accuracy lever we haven't pulled.
**How:** add population-weighted state ECMWF features keyed by (reference_month ≤ origin, target
month), with the leakage-safe `cutoff_filter_forecasting_climate` already in `folds.py`; NaN
beyond 6 months (trees handle it).
**Effort:** 1–2 d.

### 5.2 Hierarchical coherence (state ↔ city ↔ national) — **P2**
**What:** state, city, and national forecasts are produced independently and may be incoherent
(cities don't sum to their state).
**How:** forecast reconciliation (MinT / bottom-up) across the geography hierarchy — a clean
methodological contribution for the paper and a likely accuracy gain.
**Effort:** 2–3 d.

### 5.3 Better ensemble than unweighted median — **P2**
**What:** the shipped ensemble is unweighted Vincentization; inverse-WIS and QRA/stacking were
only sketched.
**How:** implement per-quantile constrained stacking (QRA) tuned on fold 1, with the
leave-one-fold-out robustness check the plan specified; compare honestly.
**Effort:** 1–2 d.

### 5.4 Disease-specific modeling — **P3**
Chikungunya has biennial dynamics and far sparser city series; give it its own tuned features
(longer memory, epidemic-year indicator) rather than reusing the dengue recipe. Ties into §1.2.
**Effort:** 1–2 d.

### 5.5 Deseasonalized climate-lag analysis — **P3**
The EDA's raw lag estimates (+12/+8/+4 wk) are confounded by shared annual seasonality; run the
STL-residual cross-correlation to confirm the lag windows feeding the panel.
**Effort:** half a day.

---

## 6. Reproducibility & ops (already strong — polish)

### 6.1 Data-refresh workflow for the September forecast — **P1**
The committed raw data ends 2026-03-08; the forecast phase needs EW25 2026. Script the FTP/API
re-pull + checksum update + `make reproduce`, so producing the real 2026–27 forecast is one
command on fresh data.
**Effort:** half a day.

### 6.2 Experiment tracking — **P3**
With many model/feature variants coming for the paper, add lightweight tracking (MLflow or a CSV
run-log keyed by config hash) so results are attributable to exact configs.
**Effort:** half a day.

---

## 7. Suggested sequencing

**Sprint 1 — correctness & speed (≈2–3 days):** §1.1 EW53, §1.2 zero-median, §2.1 loader cache,
§2.4 weight persistence, §1.3 fold-4 flagging. *Unblocks the stuck uploads, removes a real bug,
and makes everything faster before the September push.*

**Sprint 2 — engineering hygiene (≈2–3 days):** §3.1 Forecaster protocol, §3.2 dedup logic,
§3.3 unified CLI, §4.1 fast tests, §3.5 CI/ruff. *Makes the codebase safe to extend.*

**Sprint 3 — accuracy for September + paper (≈1 week):** §5.1 ECMWF features, §6.1 data refresh,
§5.3 stacking ensemble, §5.2 hierarchical reconciliation. *Directly targets the real 2026–27
forecast and the paper's methods.*

Each sprint is independently valuable and leaves the repo in a shippable state.
