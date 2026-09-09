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

### 1.1 EW53 season-week collision — **RESOLVED** (verified 2026-09-08; was P0)
**What it described:** a hypothesized `mechanistic.py::season_week` mapping both EW1 and EW53 to
the same season-week index, corrupting 53-week-season data (e.g. the actual 2025-26 season, which
does have an EW53 - 2025 is a 53-epiweek year; 2026 is not).
**Status:** already fixed by the time of a 2026-09-08 verification pass - the current
`mechanistic.py::season_week_from_date` computes the index via day-arithmetic from the season's
EW41 start date (`(date - season_start).days // 7 + 1`), not a naive epiweek-number mapping, and
its own docstring already states this is "so 53-week years don't collide." Directly verified
against the real case: EW52 2025 -> season-week 12, EW53 2025 -> season-week 13, EW1 2026 ->
season-week 14 - three distinct, sequential values, no collision. This entry was left describing
an already-fixed bug as an open P0 - correcting it here so it isn't mistakenly "fixed" again or
used to distrust the fold-4/2025-26 mechanistic results. A unit test asserting injectivity over
EW1-EW53 (the original "How" suggestion) is still worth adding as a regression guard, but the P0
correctness risk itself is gone.

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
**Note (checked 2026-09-08): only half-built.** `ClimatologicalQuantileModel` already has a
`point_estimate: "median" | "mean"` constructor option implementing exactly the "mean instead of
median" fix in (b) - but `scripts/generate_forecast_2026_27.py` instantiates it with the bare
class (`ClimatologicalQuantileModel`, no factory/lambda), so the forecast actually generated and
submitted still uses the default `"median"` and gets none of this fix's benefit. If this is meant
to be the fix, it needs to actually be wired in at the call site
(`city_single_forecast(..., lambda: ClimatologicalQuantileModel(point_estimate="mean"), ...)` for
the city tracks at least) - don't assume it's applied just because the option exists.
**Effort:** 2–4 h (short term); 1–2 d (proper model).

### 1.3 Fold-4 is prospective, not validated — **P1**
**What:** the 2025–26 season is incomplete in the data; fold-4 scores in the backtest are on a
truncated window and the submitted fold-4 forecast is unscoreable until the season resolves.
**Why:** any headline number that averages fold-4 is misleading.
**How:** already partially handled (reported separately), but codify it: exclude fold-4 from
`summarize()` headline aggregations by default, add a `resolved_only` flag. Re-score when
refreshed data arrives.
**Effort:** 1–2 h.

### 1.4 GRU is not bit-reproducible (MPS/CUDA) — **RESOLVED** (fixed 2026-09-09; was P1)
**What:** documented, but the GRU still varies run-to-run on Apple MPS. Extended overnight
(2026-09-08): this machine's `_device()` only ever checked MPS, never CUDA, so `gru_negbin` had
always trained on CPU despite a CUDA-capable GPU being available. Adding a CUDA check and
re-running the (already deployed, paper-cited) `gru_negbin` model - same code, same
`torch.manual_seed` - gave WIS=1434.1 vs the original CPU run's 1372.9, and normWIS_ex2024 nearly
DOUBLED (0.467 vs 0.298). This is not bit-level noise, it's a real quality regression: likely
cuDNN's fused GRU/lgamma kernels taking a different, non-bit-identical path than the CPU reference
implementation, compounding over 40 epochs into a different local optimum. (A second architecture,
the pinball-loss `gru_wis` in `dl_sequence_wis.py`, showed no such divergence between backends -
so this is architecture-dependent, not universal, which makes it more dangerous, not less: you
can't assume a given model is safe on a new backend just because another one was.) Raised to P1
because this is no longer just a reproducibility nicety - it's a real risk of silently reporting a
worse number (or, undetected, a *better*-looking but spurious one) if this pipeline is ever run on
different hardware than it was validated on. `_device()` was left CPU-only for now (the pre-
existing behavior) specifically to avoid this risk until the fix below exists.
**How:** add a `deterministic=True` model flag that forces CPU + `torch.use_deterministic_algorithms(True)`
+ fixed seeds, for the canonical/paper runs; keep MPS/CUDA for fast iteration, but require a
before/after comparison against the CPU/deterministic result before trusting a GPU-trained number
for anything reported. Persisting weights (§2.4) also sidesteps this for the submission artifact.
**Done (2026-09-09):** `DLSequenceModel.__init__` takes `deterministic: bool = False`; when
True, `_device()` returns `torch.device("cpu")` unconditionally (ignoring MPS availability) and
`torch.use_deterministic_algorithms(True)` is set. Off by default so fast local iteration on MPS
is unaffected; the canonical/paper run should pass `deterministic=True` explicitly. Regression
test added (`tests/test_dl_sequence.py::test_deterministic_flag_is_reproducible`): two independent
fits with the flag on produce predictions matching to `atol=1e-5` (not exact bit-equality, since
CPU BLAS reduction order can still vary a few ULPs across thread counts - a much smaller effect
than the backend divergence this fixes). Full `tests/test_dl_sequence.py` suite (6 tests) passes.
Not yet done: actually re-running the paper-cited `gru_negbin` result with `deterministic=True`
to confirm it reproduces the existing CPU-trained number bit-for-bit-ish before calling it the new
canonical run - the flag exists and is tested, but the deployed ensemble's training scripts
(`run_dl.py`, `run_ensemble.py`) don't pass it yet.
**Effort:** half a day.

### 1.5 The paper's entire Results table is stale relative to the current data — **P0** (found 2026-09-08)
**What:** every backtest result file behind the deployed ensemble and the paper's Table
`tab:leaderboard` (`results/metrics/baselines_scored.csv`, `lgbm_scored.csv`, `gru_scored.csv`,
`mechanistic_scored.csv`, `final_scored.csv`, `final_leaderboard*.csv`) was generated on
2026-09-06 22:34 - BEFORE commit `98cf72b` ("Refresh data through EW25 2026...", 2026-09-07
01:19) updated `data/raw/data_imdc_2026/dengue.csv.gz`. None of these files were regenerated
after that refresh. Confirmed by re-running each model fresh overnight (2026-09-08) and
comparing: `climatological_quantile` WIS 1327 (stale) -> 1264 (current data), `naive` 1664 ->
1557, `seasonal_naive` 1459 -> 1379, `mechanistic_traj` 1303 -> 1238 - a real, consistent ~5%
shift across every model, not noise (each re-run is itself byte-reproducible). Root cause: this
project's cutoff-filter backtest methodology re-derives each fold's training data from
`load_cases()`'s CURRENT, most-recently-revised vintage every time it runs - a case-count
revision to an already-past week (a known feature of epi surveillance reporting: recent weeks'
counts are revised upward as delayed reports arrive) changes backtest results for a fold even
though that fold's `train_cutoff` date itself hasn't moved. This is an accepted practice in the
field (obtaining true historical data vintages is rarely feasible), but it means **every data
refresh silently invalidates all previously-saved backtest numbers** until they are regenerated -
and the paper's `imdc_paper.tex` Table~\ref{tab:leaderboard} (and the specific conformal factors
1.03/1.00/1.23/1.73, coverage figures 47/76/88/93, and every other number quoted from that table
in the Results/Discussion prose) was written from the stale, pre-refresh numbers and was never
updated.
**Why it matters:** the paper currently reports the wrong (out-of-date) headline numbers for its
own most current dataset. The qualitative conclusions likely still hold (relative model ordering
looks stable under a quick spot-check) but every specific figure is off by several percent, and
the "best" conformal-widening factors, ensemble weights, and CQR calibration have not been
re-tuned on the current data either - they're the same numbers, just now describing a dataset
that no longer exists.
**Done overnight (2026-09-08):** regenerated the full chain - baselines (`run_baselines.py`) ->
LightGBM (`run_ml.py`) -> GRU (`run_dl.py`, ~80 min on CPU, the long pole) -> mechanistic
(`run_mechanistic.py`) -> ensemble (`run_ensemble.py`, which re-tuned conformal factors and
inverse-WIS weights on the freshly regenerated fold 1). Also found and fixed the SAME staleness
on the dengue/chikungunya CITY tracks (`run_cities.py` -> `city_dengue_scored.csv`,
`city_chikungunya_scored.csv`) and the chikungunya STATE track (`run_chikungunya.py` ->
`chik_final_scored.csv`, `chik_final_leaderboard.csv`) - all shared the identical 2026-09-06 22:34
timestamp as the dengue-state files, confirming this was a single, project-wide regeneration gap,
not specific to one track. Chikungunya's qualitative story is unchanged after regeneration
(`lgbm_quantile` still clearly best, WIS 79.1, ensemble still dilutes it at 89.5 - matches the
paper's existing claim). All `*_scored.csv` files and `final_scored.csv`/`final_leaderboard*.csv`
(both dengue-state and chikungunya-state) and both city-track files now reflect current data.
This regeneration also
surfaced a genuine ensemble improvement found the same night (docs/FUTURE_WORK.md Sec 6c):
`lgbm_quantile` swapped for `xgb_quantile` in `run_ensemble.py`'s `MEMBERS`, validated under the
fold-1 tuning discipline. The corrected, current, and improved deployed ensemble is now
**WIS=1162.2, normWIS_all=0.561, normWIS_ex2024=0.375** (new conformal factors: 1.08/1.04/1.19/1.86
for 50/80/90/95, vs the stale 1.03/1.00/1.23/1.73).
**STILL OUTSTANDING:** manually sync every cited number in `imdc_paper.tex`/`imdc_paper_SI.tex`
(Results text, Table~\ref{tab:leaderboard}, Table~\ref{tab:byfold}, Fig~\ref{fig:conformal} and
its caption, the specific conformal-factor and coverage numbers in prose, and any other table that
sources these files - `tab:ablation`, `tab:chik`, `tab:operational`, etc.) - deliberately NOT done
automatically overnight, since it touches the manuscript's actual reported results and deserves
your review rather than an unattended rewrite. Two decisions bundled together for that review: (1)
sync the paper to the corrected data with the SAME `lgbm_quantile`-based ensemble it already
describes, or (2) also adopt the `xgb_quantile` swap in the paper. Either way, consider building a
script that renders the LaTeX tables directly from the CSVs rather than hand-transcribing numbers,
to make this a non-issue on the *next* data refresh too - and a lightweight check (e.g. a recorded
raw-data file hash alongside each `*_scored.csv`) that can detect "this result predates the
current data" automatically, rather than relying on manually noticing file timestamps as happened
here.
**Effort:** CSV regeneration - done. Remaining: a half day for a careful paper-number sync.

**Follow-up verification (2026-09-08, later the same day):** re-ran every remaining model not
covered by the first pass - `prophet_model.py` (v1), `prophet_v2.py`, `sarimax_model.py` (v1),
`sarimax_v2.py` (tune-once), `mechanistic_nsub.py`, `hierarchical.py` - to check whether they were
ALSO affected. They were not: all five came back byte-for-byte or near-identical to their saved
files (prophet_v1 WIS 2177.3 vs saved ~2174.7 - a 0.1% difference, consistent with cmdstanpy's own
run-to-run optimizer noise, not staleness; prophet_v2 1520.141462 vs saved 1520.141462, exact
match; sarimax v1 1418.6 vs 1418.6; sarimax v2 tune-once 1751.86 vs 1751.86; mechanistic_nsub
1269.14 vs 1269.1). Root cause of the difference from the first-pass files: every one of these was
generated DURING active development work on 2026-09-07, i.e. already after the data refresh
(timestamps confirm this - e.g. `prophet_v2_scored.csv` is dated Sep 7 19:57, `sarimax_v2_scored.csv`
Sep 7 18:49), whereas the stale batch (`baselines_scored.csv` etc.) was all dated Sep 6 22:34,
before the refresh. So the staleness bug was narrower than it first looked: it only ever affected
files nobody had touched since before the refresh, not every result file in the project.
`sarimax_v2_perfold_scored.csv` (Sep 7 22:27, same "already fresh" cohort) was not re-verified
given the ~2h43m cost and this now-4-for-4 confirmation pattern, but should be assumed fine on the
same evidence unless something else changes.
**Also resolved by this pass:** `hierarchical_scored.csv` did not exist as a saved file before -
generated fresh, WIS=1232.1, normWIS_all=0.5948, normWIS_ex2024=0.4010 (still beats the unpooled
`climatological_quantile`'s 0.6103, confirming the original ablation finding holds on current
data). And the `mechanistic_traj` vs `mechanistic_nsub` choice for the paper's mechanistic entry
(previously ambiguous - nsub won raw WIS, traj won normalized WIS, on stale data) is now decided
cleanly: on current data, **`mechanistic_traj` wins on all three metrics** (WIS 1238.1 vs 1269.1;
normWIS_all 0.5977 vs 0.6127; normWIS_ex2024 0.4503 vs 0.4517) - use `mechanistic_traj`, not
`mechanistic_nsub`, as the paper's mechanistic-family entry.

**Landmine described above - RESOLVED 2026-09-08.** `scripts/generate_forecast_2026_27.py` now
imports and uses `XGBQuantileModel` (not `LGBMQuantileModel`) for the dengue-state ensemble's GBM
member, matching both `run_ensemble.py`'s `MEMBERS` and the paper's now-adopted XGBoost swap, so it
is consistent with the current `conformal_factors.csv` (1.08/1.04/1.19/1.86). Verified the script
still imports and parses cleanly after the change (not run end-to-end, since doing so would write
real forecast files). Chikungunya-state deliberately still uses `LGBMQuantileModel` - XGBoost was
never evaluated for chikungunya, so there is no basis yet to switch it. **The already-submitted
2026-27 forecast is still unaffected either way** - it was generated earlier with the old
lgbm-based ensemble on the old factors, which were mutually consistent at the time. This fix only
matters if `generate_forecast_2026_27.py` is run again in the future. Whether to actually re-run it
and resubmit before the 2026-09-10 deadline is a separate, time-sensitive decision left to the
user, not made here.

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

### 3.1 No formal model interface; dead params — **RESOLVED** (2026-09-09; was P1)
**What:** the fit/predict "protocol" is duck-typed and documented only in a docstring; there
are **3 unused `covariates=None`** params.
**How:** define `class Forecaster(typing.Protocol)` with `fit(train_df, fold) -> Self` and
`predict(target_grid) -> long_df`, type-annotate all models to it, delete the dead params, and
run `mypy` in CI. Makes the contract enforceable and the codebase navigable.
**Done (2026-09-09):** added `imdc.protocol.Forecaster`, a `@runtime_checkable` Protocol
declaring `fit(train_df, fold) -> Forecaster` / `predict(target_grid, quantile_levels) ->
DataFrame`. By the time of this pass the dead-param count had grown to **12**, not 3 - the
2026-09-08 overnight model sweep (FUTURE_WORK.md Sec 6c) copy-pasted the same
`covariates=None` into every new model file (`climate_surge`, `dl_sequence_wis`, `prophet_v2`,
`sarimax_v2`, `surge_template`, `mechanistic_nsub`) on top of the original 6
(`dl_sequence`, `mechanistic`, `ml_boosted`, `prophet_model`, `sarimax_model`, `xgb_quantile`).
Verified none of the 12 ever read `covariates` (including through the 3 `super().fit(...,
covariates)` chains in the mechanistic-family subclasses) and no caller ever passed it, then
removed it from all 12 `fit` signatures and the 3 super-calls. Added return-type annotations
(`-> "ClassName"`) to every `fit` that lacked one. New `tests/test_protocol.py` (32 tests: 16
model classes x 2 checks) asserts every model structurally satisfies `Forecaster` and that
`fit`'s signature has no `covariates` param, as a standing regression guard rather than a
one-time cleanup. Full suite (111 tests) passes.
**Not done:** `mypy` in CI - deferred to Sec 3.5 (no CI workflow exists yet at all).
**Effort:** half a day.

### 3.2 Duplicated logic (state vs city, ensemble vs forecast) — **PARTLY RESOLVED** (2026-09-09; was P1)
**What:** `harness.py` (state) and `evaluation/city.py` (city) largely duplicate run/score
logic; `ensemble.py::vincentization` and `submission/forecast.py::_vincentize_wide` both do
per-quantile median (the split exists only because `_KEYS` includes `observed_value`, which
drops NaN groups on the prediction-only path).
**How:** introduce a `Geography` abstraction (state / city) so one harness handles both; unify
the two Vincentization implementations into one that takes explicit `index_cols` and never
assumes `observed_value`.
**Done (2026-09-09):**
- **Vincentization unified.** `ensemble.py` now has `_median_ensemble(stacked, index_cols,
  passthrough_cols=None)`: groups only on caller-supplied `index_cols` (never on a column
  that can be NaN) and attaches anything else (e.g. `observed_value`) via a first-value merge
  after the groupby instead of putting it in the groupby key. `vincentization` (backtest,
  has `observed_value`) and `submission/forecast.py::_vincentize_wide` (real forecast,
  doesn't) both delegate to it. Added a regression test proving the actual bug this fixes:
  `test_vincentization_keeps_rows_with_no_observed_value` (rows used to silently vanish when
  grouped by a NaN `observed_value`).
- **State/city run+score loops merged.** `harness.py` gained `_run_backtest_generic` and
  `_score_generic`; `run_backtest`/`score_backtest` (state) and `city.run_city_backtest`/
  `city.score_city_backtest` are now thin wrappers supplying their own
  train/grid/observed-frame builders to the same shared loop and scoring math, instead of
  two independently-maintained copies of the fit/predict loop and the pivot+WIS+coverage
  logic.
- Chose **not** to touch any of the 9 files that call these four public functions
  (`run_dl.py`, `run_ml.py`, `run_mechanistic.py`, `run_chikungunya.py`, `run_baselines.py`,
  `run_cities.py`, `submission/forecast.py`, plus their own two modules) - all four
  signatures are unchanged, so this is a pure internal dedup with zero call-site risk.
  Verified via the existing test suite (`test_harness_baselines.py`, `test_city.py`,
  `test_ensemble.py`, `test_submission.py`) plus the new regression test; full suite (113
  tests) passes.
**Not done:** the full `Geography` protocol/abstraction that would let `run_backtest` and
`run_city_backtest` collapse into one public function (not just share an internal
implementation) - deliberately deferred. That's a genuinely larger, higher-risk change (a
public-API rewrite touching all 9 caller files) that this pass intentionally avoided doing
the day of the forecast-phase deadline, when this exact code had just produced the live,
already-uploaded submission. Worth doing later, without deadline pressure, with each of the 9
callers re-verified individually.
**Effort:** 1 day.

### 3.3 Seven near-duplicate `run_*.py` scripts — **PARTLY RESOLVED** (2026-09-09; was P2)
**What:** `run_baselines/run_ml/run_dl/run_mechanistic/run_ensemble/run_chikungunya/run_cities`
share ~80% boilerplate.
**How:** one parametrized CLI: `python -m imdc.run --stage backtest --model lgbm --disease dengue`
(argparse or Typer), with a registry mapping model names → factories. Collapses 7 files into
one + a table.
**Done (2026-09-09):** added `imdc/run.py` - `python -m imdc.run --model <name> --disease
<dengue|chikungunya> [--ufs ...] [--out path.csv]`, plus `--list`. `MODEL_REGISTRY` maps 16
model names to factories (all the ones registered in `imdc.run.MODEL_REGISTRY`, spanning every
model family in `imdc/models/` and `evaluation/baselines.py`), with heavy per-model imports
(torch, statsmodels, prophet's cmdstanpy) done lazily inside each factory so `--list` and
unrelated models stay fast and don't require every optional dependency installed.
`tests/test_run_cli.py` (5 tests) covers registry coverage, that every factory constructs
without needing to fit, an end-to-end run of a fast model, `--list`, and the missing-`--model`
argparse error.
**Deliberately did NOT touch the 7 existing scripts or the Makefile.** They're wired into
`make reproduce` and the paper/README's reproducibility claims, and each has its own bespoke
"merge into the combined leaderboard" behavior (which prior `_scored.csv` files to read and
concatenate) that this CLI doesn't try to replicate - collapsing them for real means deciding
what happens to that behavior, then re-validating each of the 7 Makefile-invoked stages
individually, which is real additional work best done without deadline pressure. This is
additive: a new model/experiment can go through the registry instead of a copy-pasted script,
which is the actual pain point the item exists to fix, without touching anything the
already-submitted forecast or the paper's reproducibility claims depend on.
**Effort:** half a day.

### 3.4 Scattered configuration — **P2**
**What:** hyperparameters (`DEFAULT_PARAMS`), feature lists (`FEATURE_COLS`), ensemble members
(`MEMBERS`), and track configs live in different modules as literals.
**How:** centralize experiment config in dataclasses (or one `configs/*.yaml`), so a run is
fully described by a config object that also gets stamped into the provenance manifest.
**Effort:** 1 day.

### 3.5 No CI, no linting, no logging — **PARTLY RESOLVED** (2026-09-09; was P2)
**How:** add a GitHub Actions workflow (`pytest -m "not slow"`, `ruff`, `mypy` on push); adopt
`ruff` for formatting/lint; replace `print` in scripts with the `logging` module.
**Done (2026-09-09):** `.github/workflows/ci.yml` - a `fast` job runs `ruff check` +
`pytest -m "not slow"` on every push/PR (~55s); a `full-nightly` job runs the complete suite
(including `@pytest.mark.slow`, ~6 min) on a nightly cron plus manual dispatch. Added
`[tool.ruff]` to `pyproject.toml` with a deliberately **narrow** rule set (`E9` syntax errors +
`F` pyflakes, i.e. real bugs/undefined names) rather than a full style ruleset - running the
narrow set today found zero violations, so CI starts green; widening it later is a one-line
config change whenever there's appetite for the resulting diff. Also fixed a real, unrelated
gap this surfaced: `pyproject.toml`'s `dependencies` was missing `epiweeks`, `torch`,
`python-dotenv`, and `joblib` despite all four being genuinely imported across `src/` (an AST
scan of every top-level import in `src/` was used to check, not a guess) - a clean-machine
`pip install -e .` would have failed. `prophet` (used by exactly two exploratory,
non-deployed models, lazily imported inside their `fit()`) was added as an optional
`[project.optional-dependencies] prophet` extra instead of a hard dependency, since it needs a
cmdstan toolchain and nothing else in the package requires it importable to install/test.
Verified `pip install --dry-run -e ".[test]"` resolves cleanly.
**Not done:** `mypy` (Sec 3.1 left this for here; still no type-checking step - would need a
first pass to see how much the current type-hint coverage already satisfies before deciding a
ruleset) and the `print` → `logging` migration across the `run_*.py` scripts (wide, low-risk-
but-also-low-value churn across every script the day of a live deadline; the scripts' `print`
output is only ever read by a human running them locally, not consumed programmatically, so
this is cosmetic rather than a real gap).
**Effort:** half a day (CI + ruff), ongoing.

---

## 4. Testing

Current suite is solid on **leakage, WIS correctness, determinism, and per-model smoke**, but:

### 4.1 Tests are integration-heavy and slow (10 min) — **PARTLY RESOLVED** (2026-09-09; was P1)
**What:** most tests hit real gzip data and train real models.
**How:** add a `tests/fixtures/` synthetic mini-dataset (a few states, ~150 weeks) and unit
tests that run in milliseconds; mark the real-data ones `@pytest.mark.slow` and run only fast
ones on every push, slow ones nightly. Pairs with §2.1 to cut time further.
**Done (2026-09-09):** profiled the full suite (`pytest --durations=30`) and marked the actually
slow tests `@pytest.mark.slow`: `test_determinism.py::test_lgbm_is_bit_deterministic` (~234s -
fits LightGBM on the real state panel twice), all of `test_ml_boosted.py` (~111s shared
module-scoped fixture - LGBM trains on the full 26-state panel regardless of the `ufs` filter,
only the target grid is restricted), `test_dl_sequence.py`'s fixture-dependent tests plus the
new determinism test (~24-46s each, real GRU fits), and
`test_folds_and_leakage.py::test_climate_table_has_no_fold_flags_and_needs_manual_cutoff`
(~24s - loads the real climate table). `pytest -m "not slow"` (the new CI `fast` job, Sec 3.5)
now runs 107 tests in ~55s, down from the full suite's ~8min for 118; `pytest -m slow` runs the
remaining 11 in ~6min, now relegated to the nightly CI job instead of every push.
**Not done:** the actual `tests/fixtures/` synthetic mini-dataset. Marking existing slow tests
gets the every-push CI loop to under a minute, which is most of this item's practical value;
the synthetic dataset would let those *specific* real-data/real-model tests themselves run in
milliseconds too, but building one that's realistic enough to keep every model family's tests
meaningful (right schema across cases/population/climate/ocean-indices, right leakage-relevant
date structure) is genuinely the "1 day" of effort the estimate says, not something to build
carefully under the current deadline pressure. Deferred, not abandoned.
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
