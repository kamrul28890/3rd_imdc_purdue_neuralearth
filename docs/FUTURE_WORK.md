# Future Work — IMDC 2026 (Team Neural Earth)

The project's calendar-anchored roadmap: the competition **forecast phase**, the webinars, and the
**journal paper**, plus how improvements sequence against those milestones. This is the
higher-level companion to:
- `docs/IMPROVEMENTS.md` — the code/engineering/accuracy backlog (line-level, prioritized).
- `docs/PLAN.md` — the original phase 2–8 implementation plan (history).

## 0. Where we are

Validation phase **submitted and verified** (308/308 predictions across dengue/chikungunya ×
state/city; 78 tests passing; fully reproducible public repo). The next hard deadline is the
**forecast phase, 2026-09-10**. Everything below is oriented around that and the paper.

**Read this first (found overnight, 2026-09-08):** the paper's Table~\ref{tab:leaderboard} and
every number sourced from it are stale relative to the current data (the September refresh
updated `dengue.csv.gz` but the backtest was never regenerated afterward) - full writeup and fix
plan in IMPROVEMENTS.md §1.5. The result CSVs behind the deployed ensemble were regenerated
overnight; the paper's actual LaTeX text/tables were deliberately NOT touched, pending your review.

## 1. Key dates
*(Confirm against https://sprint.mosqlimate.org/calendar — dates shift.)*

| Date | Milestone |
|------|-----------|
| 2026-07-31 | Validation-round results webinar |
| **2026-09-10** | **Forecast-phase submission — true 2026–27 season** ← primary deadline |
| 2026-09-22 | Team methodology presentations |
| 2026-10-15 | Technical results webinar |
| 2026-10-30 | Public results webinar |
| (rolling) | Journal paper — PNAS-style, following Araujo et al. 2026 |

## 2. Workstream A — Forecast phase (the real 2026–27 forecast) · highest priority

The pipeline already generalizes — fold 4 was effectively a dry run for this. Steps:

1. **Data refresh.** Re-pull `data_imdc_2026` from the FTP / Mosqlimate API through **EW25 2026**
   (current data ends 2026-03-08); regenerate `data/raw/data_imdc_2026/CHECKSUMS.sha256`. Script it
   (IMPROVEMENTS §6.1) so it is one command.
2. **Population extrapolation** to 2026/2027 for incidence normalization and state aggregation
   (population data ends 2025; currently clipped to the last available year). Add a documented
   extrapolation (e.g. last-observed growth rate).
3. **Re-run** `make reproduce` with `train_cutoff = EW25 2026`; produce the full 2026–27 season
   forecast for all four tracks via `imdc/submission/forecast.py`.
4. **Validate + upload** to the *forecast* phase using the same tooling (`submission/validate.py`,
   `submission/upload.py`, `scripts/finish_upload.py`); reuse registered model **id 83**.
5. **Buffer.** Internal-complete by ~Sept 5, upload by Sept 8 — never the deadline day (the API was
   intermittently timing out during the validation upload).

## 3. Workstream B — Modeling improvements to land *before* the forecast phase

*(Status update, 2026-09-08: the three items below marked "DONE" were originally written as open
TODOs; by the time of a September pass through the codebase all three were already implemented
and their outcomes already known. Left here so the historical prioritization stays legible, but
don't re-do them. Note: the specific WIS/normWIS figures quoted below for `hierarchical.py` and
the invwis/Vincentization comparison predate the 2026-09-07 data refresh and are now stale in the
same way described in Sec 6c/IMPROVEMENTS.md Sec 1.5 - `hierarchical.py` itself was not
regenerated overnight, so its exact numbers aren't yet corrected, but its qualitative conclusion
(hurts small states, hurts the ensemble) is unlikely to flip. The invwis-vs-Vincentization
comparison's CORRECTED numbers are in Sec 6c's "Conclusion.")*

- **ECMWF seasonal-climate features** (§5.1) — **DONE.** `panel.py`'s `ECMWF_COLS` are wired into
  `FEATURE_COLS` and consumed by every ML model via `build_panel`/`build_prediction_features`.
  Coverage is real but partial (~44% of training-panel rows have a non-null match, since ECMWF's
  own seasonal forecast only extends ~6 months ahead while our horizon runs 52-67 weeks) - LightGBM
  and XGBoost handle the missingness natively via NaN-aware splits.
- **Hierarchical reconciliation** (state ↔ macroregion partial pooling, §5.2) — **DONE, rejected.**
  `models/hierarchical.py` implements James-Stein-style shrinkage of each state's seasonal
  incidence distribution toward its macroregion. Standalone it slightly beats the unpooled
  climatological baseline (normWIS 0.606 -> 0.590), but pooling small states (RR, AP) toward a
  macroregion dominated by a differently-behaved large state (AM) makes small states worse, and
  adding it to the ensemble hurts (1216 -> 1241). Kept as a paper ablation, not in the ensemble.
- **Stacking / QRA ensemble** (§5.3) — **DONE, mixed.** `models/ensemble.py`'s `inverse_wis_weights`
  is a working (if simple) stacking scheme, already compared against equal-weight Vincentization in
  `run_ensemble.py`'s leaderboard: invwis's raw WIS (1287) is worse than conformal-calibrated
  Vincentization's (1216), though invwis edges it slightly on normWIS_ex2024 (0.321 vs 0.333). A
  further attempt at a horizon-bucketed *dynamic* weighting scheme (inspired by CERI's "bridge"
  model - see Sec 6c) also underperformed static Vincentization (WIS 1288). The deployed choice
  (conformal-calibrated equal-weight Vincentization) remains the best found across all variants tried.
- **Sparse-series handling.** The degenerate zero-median fix (mean point estimate) already shipped
  for the city tracks — generalize it into a principled zero-inflated / hurdle option and unit-test it.
- **Deseasonalized climate-lag re-check** (§5.5) and **disease-specific chikungunya** features
  (§5.4, biennial dynamics) — chikungunya currently reuses the dengue recipe.

## 4. Workstream C — Engineering hardening (enables fast, safe iteration before September)

From IMPROVEMENTS.md, in order:
- **Loader caching** (§2.1) — ~3–5× faster everywhere; the single biggest speedup.
- **Model-weight persistence** (§2.4) — makes the forecast run seconds, not an hour.
- **EW53 season-week bug** (§1.1) — a real latent bug affecting 53-week years like 2026.
- **`Forecaster` protocol + remove dead params** (§3.1); **unify state/city + ensemble duplication**
  (§3.2); **one parametrized CLI** (§3.3); **fast synthetic-fixture tests + CI/ruff** (§4.1, §3.5).

## 5. Workstream D — The journal paper (PNAS-style)

Build on `paper/imdc_paper.tex`:
- **Main text:** finalize the 5-family + ensemble comparison; the headline **metric-disagreement**
  finding (raw-mean WIS vs scale-free relative-WIS pick different winners); calibration (CQR); the
  **hyperparameter-tuning negative result**; and the newly-discovered **degenerate-median /
  episodic-sparsity** finding as a methods note.
- **Figures:** keep `paper_wis_by_fold`, `paper_coverage`, `paper_relative_wis`; add a state-level
  **relative-WIS choropleth** (geopandas + `shape_muni.gpkg`) tied to the EDA's Köppen gradient.
- **SI appendix:** full WIS/CRPS math, feature list + frozen hyperparameters, DL architecture,
  mechanistic derivation, per-state/per-fold tables, ensemble-weight robustness, reproducibility
  statement (public repo + `RESULTS.md` provenance).
- **Sequencing:** draft Intro/Data/Methods now (data-independent); fill Results/Discussion after the
  forecast phase and once the 2025–26 season resolves enough to score fold 4; fold in comparanda
  from the July/October webinars; submit after the October results.

## 6. Workstream E — Evaluation & ops

- When organizers publish validation scores, compare our WIS/coverage to other teams and record it.
- **Re-score fold 4** when the 2025–26 season resolves (currently prospective/partial).
- Lightweight **experiment tracking** (IMPROVEMENTS §6.2) for the many upcoming model/feature variants.
- Keep the **EpiScanner** caveat visible: the API endpoint is still down (HTTP 500); the mechanistic
  model uses the local Richards reimplementation — revisit if the endpoint returns.

## 6b. Competitor architecture notes: Prophet and SARIMAX (Epidemáticos, FGV EMAp)

Our first-pass Prophet and SARIMAX baselines (2026-09) both scored far worse than naive - not
because those model families are unsuited here, but because of specific, identifiable
implementation gaps. The same team (Epidemáticos, FGV EMAp) placed a Prophet model and a SARIMAX
model among the challenge's best-overall dengue-state models (2026-07-31 validation webinar); their
repos are public per the challenge rules
(`github.com/EzequielEBS/3rd_imdc_emap_epidematicos_prophet`,
`.../3rd_imdc_emap_epidematicos_sarimax_state`). Reviewed for lessons, not copied - our rebuild is
an independent reimplementation. What they did that we initially didn't:

**Prophet:**
- `growth="flat"` (no trend) - dengue has no secular trend, only seasonality; Prophet's default
  linear trend + changepoints can extrapolate spurious drift over a 67-week horizon.
- Real regressors: temperature/precipitation lags, ENSO, PDO, and a same-week-last-year case-count
  lag - not run purely univariate as our first pass was.
- A **custom seasonally-stratified residual bootstrap** for intervals (resample training residuals
  from a +/-4-week window around each target week's calendar position) instead of Prophet's
  built-in `predictive_samples`, whose generic Gaussian noise model isn't built for right-skewed
  count data and pinned our lower quantiles to zero.

**SARIMAX:**
- Actual exogenous regressors (the "X"): climate (PCA-reduced) + ocean indices. Our first pass ran
  a bare SARIMA with no exogenous inputs at all.
- **Per-state, WIS-ranked grid search** over `(p,d,q)(P,D,Q)`, not one fixed order forced onto all
  26 states.
- **Bootstrap-simulated forecast paths** (1000 simulated paths, empirical quantiles) instead of a
  Gaussian analytic confidence interval back-transformed through `expm1` - the latter is exactly
  what produced our first pass's absurd interval (~[-56000, 60000] for SP): a symmetric log-space
  Gaussian becomes wildly asymmetric and explosive once exponentiated at long horizons.

Rebuild status and results (final, 2026-09-08): both rebuilds substantially improved on their own
first pass but neither beat the ensemble (1216) or even the simpler v1 baselines on every metric.
**Prophet v2** (`models/prophet_v2.py`): WIS 2175 -> 1520 (a 30% reduction) after real regressors +
the seasonally-stratified residual bootstrap; six further structural experiments (documented in
the file's own docstring) all regressed and were reverted, converging on a genuine structural
limitation (severe under-prediction in the two largest states' outbreak fold) shared with this
project's GRU model - neither has a mechanism for "this season is unlike any prior season."
**SARIMAX v2** (`models/sarimax_v2.py`): the naive "tune order once, freeze" version regressed
below v1 (1752 vs 1419) because an order selected on fold 1 doesn't generalize to fold 3's
different regime; a full per-fold order search fixed most of that gap (1528) but is a tradeoff
against v1 rather than a clean win (better on normWIS_ex2024, worse on raw/normWIS_all) and is
kept as an opt-in (`select_orders=True`), not the default. This is a concrete, sourced explanation
for why two "should be able to replicate a competitor's success" attempts didn't fully close the
gap even after fixing every identified implementation gap, and a template (real regressors,
path-simulated intervals, per-unit model selection ranked by the actual target metric) for any
future classical-statistical baseline in this pipeline.

## 6c. Overnight model sweep (2026-09-08): reviewing all 24 competitor repos + new model families

Following up on Sec 6b, reviewed the READMEs/methods of all 24 other teams in
`Mosqlimate-project/3rd_IMDC_results` (not just Epidemáticos) for techniques worth independently
reimplementing. Full technique notes kept in this session's transcript; durable outcomes below.

**Mid-session correction, important - read before the rest of this section.** Partway through
this sweep, a comparison of `mechanistic_traj`'s fresh-data result against its saved
`mechanistic_scored.csv` surfaced a real discrepancy (1238 vs 1303), which traced to a much
bigger issue: every reference file behind the deployed ensemble (`baselines_scored.csv`,
`lgbm_scored.csv`, `gru_scored.csv`, `mechanistic_scored.csv`, and everything derived from them,
including the paper's Table~\ref{tab:leaderboard}) predated the 2026-09-07 data refresh and was
never regenerated afterward - full root-cause writeup in IMPROVEMENTS.md Sec 1.5. This means
every comparison EARLIER in this sweep (and the "Conclusion" originally written for this section)
was measuring tonight's new, current-data-trained models against a STALE reference ensemble. All
four stale files were regenerated overnight (`baselines_scored.csv`, `lgbm_scored.csv`,
`gru_scored.csv`, `mechanistic_scored.csv` all now reflect current data), and every ensemble
comparison below was re-run against the corrected numbers before being finalized. The corrected,
current-data deployed-ensemble baseline is **WIS=1173.4, normWIS_all=0.5664, normWIS_ex2024=0.3807,
fold-1 nWIS=0.3091** (climatological_quantile + lgbm_quantile + gru_negbin) - use THIS as the
reference point for everything below, not any number that looks like the old 1215.9/0.555/0.333
figures the paper currently cites.

One consequence of the correction changes this section's bottom line: unlike every other
comparison below (which still shows the same raw-vs-normalized disagreement under corrected data),
**swapping `xgb_quantile` in for `lgbm_quantile` in the deployed ensemble is a genuine, clean win**
under corrected data - see the `xgb_quantile` entry below and the revised Conclusion at the end.
This has already been applied: `run_ensemble.py`'s `MEMBERS` now reads
`["climatological_quantile", "xgb_quantile", "gru_negbin"]`, and `results/metrics/final_*.csv`
have been regenerated to reflect it. The paper's LaTeX was NOT touched.

**Kept (positive/interesting result):**
- **`models/surge_template.py`** (inspired by "LNCC SURGE"'s curve-alignment idea): circular-shift
  each historical season's curve to align peaks, average into a normalized template shape, forecast
  via a stochastic (peak-week offset, log-normal gain) draw against that template, layered with the
  same NegBinom observation noise as `mechanistic.py`. Standalone result: WIS=1168.1 (reproducible,
  sane per-state breakdown) - the single best individual model in the project by raw WIS,
  narrowly beating the CORRECTED deployed ensemble's 1173.4, and also narrowly best individual
  model by normWIS_all (0.564 vs 0.566). NOT added to the ensemble, though: it fails the fold-1
  tuning check decisively (fold-1 nWIS 0.444 vs the deployed ensemble's 0.309) - every member
  combination tried improves the full-aggregate raw WIS further but makes normWIS_all/ex2024
  worse, and standalone `surge_template` itself is much worse than the ensemble on fold 1
  specifically despite its close full-aggregate numbers - the same raw-WIS-vs-normalized-WIS
  disagreement this project's paper already reports as a headline finding, now reproduced by an
  independently-discovered model. Reported standalone, like `mechanistic_traj`.
- **`models/xgb_quantile.py`** (inspired by "XGBSillas"'s use of XGBoost in this challenge): a
  second classical-ML family alongside LightGBM, using XGBoost's native multi-quantile objective
  (`reg:quantileerror`, one model for all 9 quantile levels via `multi_strategy="multi_output_tree"`)
  on the identical leakage-safe panel features. Standalone (current data both sides): WIS=1190.1 vs
  `lgbm_quantile`'s 1301.2, and a win on both all-fold normalized metrics too (normWIS_all 0.575 vs
  0.628; normWIS_ex2024 0.434 vs 0.549 - `lgbm_quantile`'s own ex-2024 number moved a lot in the
  correction, from a stale 0.443). On fold 1 alone, standalone XGBoost is still marginally worse
  than standalone LightGBM (375.9 vs 351.6 WIS) - the standalone aggregate win is concentrated in
  fold 2. **ADOPTED: swapped into the deployed ensemble in place of `lgbm_quantile`.** At the
  ensemble level (combined with climatological_quantile + gru_negbin, conformal-calibrated) the
  swap wins on every metric, INCLUDING fold 1: fold-1 wis 337.3 vs 340.8, fold-1 nWIS 0.306 vs
  0.309, all-fold WIS 1162.2 vs 1173.4, normWIS_all 0.561 vs 0.566, normWIS_ex2024 0.375 vs 0.381.
  Unlike every other ensemble-composition idea tried tonight, this passes the project's own
  tuning-fold discipline cleanly - it's a natural, pre-specified single substitution (same GBM
  role, same "one baseline + one GBM + one DL" ensemble structure), not a result mined from
  searching many candidate combinations, which is what makes it trustworthy where the others
  weren't. `run_ensemble.py`'s `MEMBERS` and `results/metrics/final_*.csv` now reflect this.
  Training cost: ~37 minutes for the full 4-fold backtest (`multi_output_tree` is a markedly more
  expensive code path than LightGBM's independent per-quantile boosters - budget for this before
  re-running).
  Also tried: the `sillas_weights` option (recency-to-cutoff decay + extreme-value up-weighting,
  both techniques from XGBSillas) - a clean, uniform REGRESSION (WIS 1190.1 -> 1394.7, worse on
  every metric), driven mostly by fold 4 (467.6 -> 1176.7 WIS, with overprediction blowing up from
  8.7 to 785.2) - the extreme-value up-weighting appears to bias the model toward systematically
  over-predicting in a fold that didn't have as extreme an outbreak as the up-weighted historical
  rows implied. A follow-up ablation (`sillas_recency=True, sillas_extreme=False`) isolated the two
  components: recency-weighting alone is only a mild regression (WIS 1229.2, normWIS_all 0.593),
  confirming the extreme-value up-weighting was the dominant cause of the bigger regression, not
  recency. Neither component individually improves on the plain unweighted model, though - both
  kept as opt-in flags (`sillas_weights=False` default, i.e. off); not deleted since the code path
  is cheap to keep and the negative result itself is informative.
- **`models/dl_sequence_wis.py`** (inspired by "Dengue Oracle"'s LSTM trained directly against
  WIS): same GRU encoder/embedding architecture as `dl_sequence.py`, but the head outputs all 9
  quantile levels directly (log1p-incidence space) and trains with pinball loss instead of a
  Negative-Binomial NLL. Standalone (CPU, apples-to-apples with the deployed `gru_negbin`'s own
  CPU-trained, current-data result of WIS=1393.5): WIS=1290.9 (7% better), normWIS_all 0.623 vs
  0.673 (also better) - but normWIS_ex2024 0.387 vs 0.385 (about the same, unlike earlier stale-data
  numbers that showed a clear regression here). A GPU run gave a similar result (WIS=1273.6)
  confirming this is a genuine effect of the pinball-loss training, not device noise (contrast with
  the GPU investigation below). Despite being a clearly better STANDALONE model than `gru_negbin`
  on every metric, swapping it into the (pre-xgb-swap, lgbm-based) ensemble in place of
  `gru_negbin` still LOSES on every ensemble-level metric (current-data check): fold-1 wis 349.1
  vs that ensemble's 340.8, fold-1 nWIS 0.317 vs 0.309, all-fold WIS 1182.3 vs 1173.4, normWIS_all
  0.571 vs 0.566, normWIS_ex2024 0.386 vs 0.381 - the same loss holds swapping it into the
  now-adopted xgb-based ensemble instead (checked directly: WIS 1173.2 vs the xgb-ensemble's
  1162.2). A real, if slightly counterintuitive, finding: ensemble value
  isn't the same as standalone quality - `gru_wis` is the better individual model, but its errors
  are apparently more correlated with the other two ensemble members than `gru_negbin`'s are, so it
  adds less complementary information despite being more accurate alone. Reported standalone; not
  adopted as an ensemble member (contrast with `xgb_quantile`, which won at both the standalone AND
  ensemble level).
- **`models/climate_surge.py`** (inspired by "LNCC CLIDENGO"'s climate-modulated growth-rate ODE,
  reimplemented as a much simpler linear conditioning rather than their Brière-curve compartmental
  model): extends `surge_template.py` by regressing each historical season's log peak-height
  ("gain") against that season's own mean temperature anomaly, then centering the forecast season's
  gain distribution on the regression's prediction for the *target* season's anomaly instead of the
  unconditional historical mean - using the average anomaly over the 8 weeks before the fold's
  cutoff as a leakage-safe proxy for the target season's climate (we cannot observe 12-15 months of
  future weather). Result: WIS=1406.3, normWIS_all=0.679, normWIS_ex2024=0.677 - a clean REGRESSION
  vs. the unconditional `surge_template`'s 1168.1/0.564/0.447 on every metric, driven mostly by fold
  3 (WIS 1053.8, dispersion 897.6 - badly mis-centered/widened intervals). Most likely cause: an
  8-week pre-cutoff anomaly reading is too weak and noisy a proxy for a full season's climate more
  than a year away (unlike ENSO/PDO, which persist over many months and are already used as-is by
  other models here, a single state's temperature anomaly has much shorter-lived persistence) - the
  conditioning adds noise to the gain distribution's center without adding real signal. Kept
  (unlike the deleted ETS/RF above) because it tests a specific, competitor-inspired hypothesis
  with an identifiable, informative failure mode, matching how `prophet_model.py`/`sarimax_model.py`
  v1 are kept for the same reason.
- **GPU investigated, NOT adopted as the default.** `dl_sequence.py`'s `_device()` checked only
  Apple MPS, never CUDA, so every GRU model in this project (including the deployed `gru_negbin`)
  had been training on CPU despite this machine having a CUDA-capable GPU (RTX 3070). Fixing
  `_device()` to prefer CUDA cut `gru_wis`'s training time from 81 minutes (CPU) to 5.8 minutes
  (confirmed via `nvidia-smi` utilization) - a large, genuine speedup. But re-running the ALREADY-
  DEPLOYED `gru_negbin` on GPU (same code, same seed, current data both sides - the raw data file
  was already refreshed on disk, only the saved result CSV was stale, so this specific comparison
  is not confounded by the data-staleness issue found later the same night) gave a meaningfully
  WORSE result than the CPU run (WIS 1434.1 vs 1393.5; normWIS_ex2024 0.467 vs 0.385), while
  `gru_wis` showed no such divergence (1273.6 GPU vs 1290.9 CPU). This
  sharpens the existing known issue ("GRU is not bit-reproducible (MPS)", IMPROVEMENTS.md 1.4) from
  a reproducibility nicety into a real quality risk for at least the NegBinom-NLL architecture
  specifically (likely cuDNN's fused GRU/lgamma kernels taking a different, non-bit-identical
  optimization path than the CPU reference implementation, compounding over 40 epochs into a
  different local optimum) - torch.manual_seed seeds both CPU and CUDA generators, so this is a
  genuine backend divergence, not a seeding bug. `_device()` was reverted to CPU-only (its
  original behavior) rather than left preferring CUDA, specifically to avoid silently changing the
  already-reported, paper-cited `gru_negbin` number if `run_dl.py` is ever re-run on a CUDA
  machine. If GPU speed becomes worth it for future DL work in this project, re-validate per-
  architecture (as done here for `gru_wis`) before trusting a GPU-trained result against a CPU-
  trained baseline - don't assume they're interchangeable.

**Methodological check (no code change, a due-diligence result worth recording):** with 7 scored
models in hand tonight, ran an exhaustive search over all 2-4-member ensemble subsets, evaluated on
the "headline" folds (2-3) the paper actually reports on. The top result, `climatological_quantile
+ mechanistic_traj` (just 2 members, dropping both lgbm and gru entirely), looked like a clear win
(headline nWIS 0.565 vs the deployed ensemble's 0.601). But checking that same combination on
fold 1 - the ONLY fold this project's own tuning discipline allows using to pick ensemble members -
it is actually WORSE than the deployed ensemble (nWIS 0.390 vs 0.308). This is exactly the
overfitting-to-reporting-folds trap the fold-1-only tuning rule exists to prevent: searching ~120
combinations against just 2 evaluation folds all but guarantees some combination will look good on
noise. Confirms the deployed ensemble composition (and `run_ensemble.py`'s original reasoning for
excluding `mechanistic_traj`) is robust, not an oversight - recorded here so this exact search
isn't mistakenly re-run and mistakenly acted on in the future.

**Tried and rejected** *(all re-checked against the corrected, current-data reference files -
climatological_quantile + xgb_quantile + gru_negbin, conformal, WIS=1162.2/normWIS_all=0.561/
fold-1 nWIS=0.306, unless noted otherwise):*
- **Per-state hard model selection** (pick each state's single best model - among
  climatological/lgbm/xgb/gru_negbin - by its own fold-1 WIS, instead of one global member set for
  all 26 states). There IS real per-state heterogeneity in which model wins on fold 1 (gru_negbin
  11/26 states, climatological 7, xgb 4, lgbm 4 - not degenerate), but the selection doesn't
  generalize: headline-fold (2-3) nWIS is 0.672 vs the deployed ensemble's 0.591 - worse, not
  better. 26 independent per-state decisions each from a handful of fold-1 weeks is too
  high-variance a selection to trust; a global member set well-calibrated by Vincentization + a
  single conformal factor beats 26 separate fold-1-driven point decisions.
- **Kitchen-sink Vincentization** (NUS-CERM's philosophy: average ALL 7 available models, no
  curation, rather than the deployed 3-member set). Fails even the fold-1 check (nWIS 0.341 vs the
  deployed ensemble's 0.306) - diluting the ensemble with weaker/more-correlated members (both
  GRU variants, both GBM variants, mechanistic, surge) hurts rather than helps; curation matters
  more than raw model count here.
- **Fixed-weight blend inspired by "Universidad del Valle"'s 0.4 XGB + 0.4 climatology + 0.2
  ARIMA-family recipe** (substituting our own `sarimax_v2` for their ARIMA - note `sarimax_v2`
  itself was not regenerated after the data refresh, so this specific comparison still mixes a
  stale component): WIS=1187.3 (worse than the deployed ensemble's 1162.2), normWIS_all=0.573
  (worse than 0.561), normWIS_ex2024=0.373 (about tied with 0.375) - not a win on the metrics that
  matter, not evaluated further.
- **Per-state-size-tier conformal calibration** (instead of one global multiplicative widening
  factor across all 26 states, compute a separate factor per population tercile/tier, on the
  theory that small and large states may need different interval widths and normalized WIS
  specifically rewards good small-state calibration). Looks even more promising than originally
  found on the reporting metrics with corrected data (2-tier: WIS 1075.8, nWIS_all 0.519, both
  clearly better than the deployed ensemble's 1162.2/0.561) - but fold 1 itself (the only fold
  this project's discipline allows for making the decision) is WORSE (nWIS 0.314 vs the deployed
  global-factor calibration's 0.306). With only ~1300 fold-1 calibration rows split across tiers,
  each tier's empirical quantile (especially the 95%/97.5% tails) is estimated from too few points
  to generalize - the same overfitting-to-reporting-folds pattern as the ensemble-composition
  search above, just applied to calibration instead of model choice. Global (non-tiered) conformal
  calibration remains the deployed choice.
- **Horizon-bucketed dynamic ensemble weighting** (a lightweight version of CERI's "bridge" model:
  static inverse-WIS weights computed separately per horizon bucket instead of one global weight
  per model, using the tuning fold). WIS=1232.2 (conformal-calibrated), worse than the deployed
  static equal-weight Vincentization's 1162.2, and normWIS_all also worse (0.595 vs 0.561). Likely
  cause: the per-bucket weights turned out fairly similar to equal weighting anyway, so the added
  complexity mainly cost the robustness of Vincentization's per-quantile *median* (insensitive to
  any one bad member) by replacing it with a weighted *mean* (sensitive to any one member's
  misbehavior), without enough real heterogeneity to pay for that tradeoff.

**Tried and deleted (clearly non-competitive, no diagnostic value worth keeping a file for):**
- **Random Forest quantile regression** (bagging rather than boosting, cross-tree quantile spread
  instead of a proper leaf-sample Quantile Regression Forest) - a third classical-ML family, tried
  on our own initiative (no competitor was found using a standalone RF). WIS=1303.1, normWIS_all
  0.629, normWIS_ex2024 0.491 - worse than BOTH `lgbm_quantile` (1299.4/0.593/0.443) and
  `xgb_quantile` (1190.1/0.575/0.434) on every single metric, an unambiguous loss rather than a
  raw-vs-normalized tradeoff. Consistent with the general tabular-ML finding that boosting usually
  beats bagging when a reasonably strong feature set is already available; not a surprising or
  diagnostically interesting result, so not kept as a file.
- **Holt-Winters exponential smoothing (ETS)**, additive seasonal (52-week), per-state, bootstrap-
  simulated intervals - a classical family no competitor was found using, tried on our own
  initiative rather than from a specific reviewed technique. Both a damped-trend and a trend-free
  variant produced excessive prediction-interval dispersion at long horizons (WIS 1969 and 1525
  respectively; normWIS_ex2024 1.94 and 1.19 - both *worse than the naive baseline's 0.73*). A
  known Holt-Winters weakness (unbounded interval growth with horizon) that this project's own
  60+ week forecast horizon triggers badly; not worth further tuning given SARIMAX already covers
  the ARIMA-family classical-statistical niche more successfully.

**Reviewed, not pursued (see full transcript for the complete 24-team writeup):** several teams
used spatial/mobility-graph architectures (GNN-LSTM, SEIR with mosquito dynamics, R-INLA BYM2 CAR
spatial models) that would require infrastructure (adjacency graphs, compartmental ODE fitting)
this pipeline doesn't have and that this project's own prior spatial attempt (`hhh4_spatial.py`,
deleted) already found counterproductive for our per-state panel. Pretrained time-series foundation
models (Chronos, PatchTST, TinyTimeMixer) were used by 3+ other teams; this project already tried
Chronos zero-shot and rejected it (WIS 1467, deleted) for the same reason cited in the general
DL-forecasting literature (Monash/M4/M5): these panels are too small and autocorrelated for large
pretrained models to beat simple statistical/GBM baselines.

**Conclusion (revised after the data-staleness correction).** One genuine improvement came out of
tonight's sweep and has been ADOPTED: swapping `xgb_quantile` in for `lgbm_quantile` in the
deployed ensemble (see above) - validated under the project's own tuning-fold discipline (wins on
fold 1 AND the full aggregate), not a search artifact. Everything else tried - `surge_template`,
`gru_wis`, `climate_surge` standalone; every OTHER ensemble recombination (an exhaustive 2-4-member
subset search, a fixed-weight blend, per-state hard selection, kitchen-sink Vincentization,
per-tier conformal calibration, horizon-bucketed dynamic weighting) - still shows the same pattern
under corrected data: either better raw WIS but worse scale-free normalized WIS, or an outright
fail on the fold-1 tuning check. That pattern is robust and repeatedly confirmed, not noise or a
staleness artifact - but it is NOT universal, and the one exception (a natural, non-mined
single-substitution test) is exactly the kind of candidate worth checking even after ten
confirmations of the general pattern, which is why "try every reasonable idea" was the right
instruction rather than stopping once the pattern looked established. The deployed ensemble is now
`climatological_quantile` + `xgb_quantile` + `gru_negbin` (conformal-calibrated Vincentization),
WIS=1162.2, normWIS_all=0.561, normWIS_ex2024=0.375, on current data. `surge_template` and
`gru_wis` are kept and reported standalone in the paper's model-comparison table, the same way
`mechanistic_traj` already is, but don't replace the ensemble. If a future season's data shifts the
balance, this whole comparison is worth re-running rather than assumed to hold forever - it's a
property of the data actually observed, not a law. **Separately and more urgently:** the paper's
Table~\ref{tab:leaderboard} and every number derived from it now need to be resynced to the
corrected, current-data results (IMPROVEMENTS.md Sec 1.5) - this matters regardless of whether the
xgb_quantile swap is adopted in the paper, since the OLD lgbm-based ensemble's own numbers were
already wrong (1216/0.555/0.333 stale vs 1173/0.566/0.381 correct) before any of tonight's new
models are even considered.

## 7. Suggested calendar

- **Now → ~Jul 25:** engineering hardening (Workstream C) + start ECMWF features; prep the Jul 31 webinar.
- **Late Jul → mid-Aug:** modeling improvements (ECMWF, reconciliation, stacking); write the data-refresh script.
- **Mid-Aug → Sep 5:** data refresh → re-run → generate/validate the 2026–27 forecast; freeze the model.
- **Sep 5–8:** upload the forecast phase (buffer before Sep 10). **Sep 22:** methodology presentation.
- **Sep → Oct:** paper Results/Discussion; **Oct 15 / Oct 30** webinars; finalize + submit the paper after.

## 8. Risks & dependencies

- Data-refresh availability (FTP/API) before the September run.
- API flakiness at upload time — mitigated by the idempotent `scripts/finish_upload.py`.
- Fold-4 resolution timing for the paper's headline numbers.
- EpiScanner endpoint still down.
- Population-extrapolation assumptions for 2026/2027.
