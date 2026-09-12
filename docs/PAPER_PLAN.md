# Paper plan: probabilistic dengue and chikungunya forecasting in Brazil (IMDC 2026)

Planning document for the journal paper. Built after studying the direct precedent (the 2024
sprint paper in PNAS), the reporting standard (EPIFORGE 2020, PLOS Medicine), and the
forecast-hub evaluation literature (US/EU COVID and Flu hubs, Bracher et al. on interval scoring).
This plan is the contract for section-by-section drafting; it exists so the draft is right by
construction rather than by revision.

## 0. Writing-style rules (apply to every section of the draft)

- No em dashes. Use commas, colons, parentheses, or two sentences instead.
- No AI-cliche vocabulary: avoid "delve", "leverage" (as filler), "underscore", "crucial/pivotal",
  "testament", "landscape", "realm", "tapestry", "navigate", "in the ever-evolving", "it is worth
  noting", "plays a vital role", "rich", "robustly" as filler.
- Declarative, quantitative sentences. Every claim carries a number or a figure/table reference.
- Active voice where natural. Past tense for what we did, present tense for what is true.
- Hedge only where the data hedges. State verified results plainly.
- **Dense and compact, not verbose, but never lossy** (user directive, 2026-09-12): the target
  venue is not yet decided, so do not cut content or findings to satisfy any one journal's word
  limit. Tighten prose instead: shorter sentences, no throat-clearing or filler, one idea per
  sentence, cut redundancy, keep every distinct finding, number, and nuance. A numeric word/figure
  limit only becomes a real constraint once a venue is actually chosen, at the final formatting
  stage, not during drafting or auditing. When reviewing a section against a specific journal's
  limit before that point, reframe an "over the limit" finding as a compaction opportunity, not a
  content-cutting mandate.

## 1. Contribution statement (what makes this top-journal material)

A single-team, fully reproducible, leakage-controlled comparison of five probabilistic
forecasting model families on the official IMDC 2026 backtest, yielding four generalizable
findings plus two informative negative results:

1. **Season-dependent rankings under regime shift.** No model is best across seasons. The deep
   network is the most accurate model in normal seasons yet the worst on the 2024 record outbreak
   (a natural experiment in distribution shift). Ensembles never incur any member's worst-season
   failure.
2. **The winner depends on the evaluation metric.** Magnitude-weighted mean WIS and scale-free
   normalized/relative WIS pick different winners, driven by real cross-state skew. Leaderboards
   reporting a single aggregate can mislead. We report both.
3. **Conformal recalibration of the ensemble is a cheap, transferable accuracy and calibration
   gain** (WIS 1281 to 1216, normalized WIS 0.585 to 0.555, out-of-sample -5.5%). It is asymmetric
   insurance against catastrophic under-prediction in an extreme season.
4. **Simple methods are hard to beat; complexity did not pay.** Two clean negative results: adding
   observed-climate covariates to the boosted model did not help (the forecast-relevant signal is
   the seasonal climate forecast and ENSO, not observed weather), and hierarchical macroregion
   pooling of seasonal climatology did not help (states are too heterogeneous within a region).
   A hyperparameter search that improved an internal holdout worsened the true backtest.
5. **The best model is disease-specific** (chikungunya: gradient boosting alone beats the ensemble).
6. **Reproducibility as a first-class result:** public code, a leakage-safe harness, ~74 automated
   tests, provenance manifest, and validated submissions across four official tracks.

## 2. Target venue (decision required before drafting; see recommendation)

| Venue | Fit | Note |
|---|---|---|
| **PLOS Computational Biology (recommended)** | Rigorous computational-methods comparison, probabilistic forecasting, reproducibility-first, allows detailed Methods + unlimited SI, open access, EPIFORGE-friendly. | Best home for a single-team methods paper of this depth. |
| Epidemics (Elsevier) | Specialist epidemic-modeling journal; natural community fit; detailed methods welcome. | Strong targeted fallback. |
| PNAS | The 2024 collective sprint paper is here, but that was the organizers' multi-team result. A single-team methods paper is a harder significance sell, and the organizers will likely publish the 2026 collective results here again. | Aspirational, higher risk for one team. |
| Lancet Digital Health / Lancet Reg Health Americas | High impact, but wants a stronger real-time deployment / health-systems-impact angle than a retrospective comparison provides. | Needs reframing toward operational impact. |
| PLOS Neglected Tropical Diseases | Fits the diseases, but is more epidemiology than methods. | Secondary. |

Recommendation: **PLOS Computational Biology**, keeping the rigorous PNAS-style structure so the
manuscript stays convertible. The content below is ~90 percent venue-independent; only length,
the lay-summary format, and reference style differ.

## 3. Annotated outline (sections, subsections, key claim, evidence)

Format follows Results-first (Intro, Results, Discussion, Methods, SI), standard for PLOS Comp Biol
and PNAS.

### Title (working)
"Probabilistic dengue and chikungunya forecasting in Brazil: metric-dependent model rankings,
regime-shift fragility, and conformal ensembles." Refine at draft time.

### Author summary / Significance (lay, ~120 words)
Why forecasting dengue matters in Brazil after 2024; the practical question of which model to
trust; the headline that no model wins everywhere, that the metric changes the answer, and that a
calibrated ensemble is the robust choice.

### Abstract (structured, ~250 words)
Background, data and challenge, methods (five families, leakage-safe backtest, WIS), results
(the four findings with the key numbers), conclusion (ensembles plus dual-metric reporting plus
conformal calibration).

### 1. Introduction
- 1.1 Arbovirus burden in Brazil and the 2024 record season (440,538 peak weekly cases; ~6.67 M
  annual, ~4x the prior maximum). Climate and El Nino context.
- 1.2 Why probabilistic, calibrated forecasts are what preparedness needs (not point estimates).
- 1.3 The IMDC challenge and the contested modeling landscape (statistical vs ML vs DL vs
  mechanistic); the open question of robustness under regime shift.
- 1.4 Contributions, as an itemized list (the six above).

### 2. Results
- 2.1 Study design and data at a glance (Figure 1). Four official seasons, 26 states, the 15-week
  reporting-gap discipline, the WIS target. Points forward to Methods.
- 2.2 No single model dominates across seasons (Figure 2, Table 2). The regime shift: GRU best in
  folds 1 and 3, worst on fold 2 (2024); mechanistic and boosting most robust on the outlier;
  fold 2 ~10x harder than any other.
- 2.3 The winner depends on the metric (Table 1, Figure 3). Ensemble best by normalized WIS
  (0.555); GRU best on normal seasons (normalized WIS ex-2024 0.298); mechanistic/boosting look
  strong only because the raw mean is outlier-dominated. Rank-reordering visual.
- 2.4 Calibration and its correction (Figure 4). GRU covers 32 percent at nominal 50; CQR fixes
  the boosted model; sharpness/calibration trade-off.
- 2.5 Conformal recalibration of the ensemble (Figure 5, the WIN). Factors {50:1.03, 80:1.00,
  90:1.23, 95:1.73}; gentle tail widening; 1281 to 1216, coverage to 47/76/88/93; out-of-sample
  -5.5 percent; asymmetric-insurance interpretation; ensemble-output beats member recalibration.
- 2.6 What did not help (Table 3, ablations). Observed-climate covariates (1299 to 1342; 5.3
  percent importance; the ECMWF-forecast-plus-ENSO explanation), hierarchical macroregion pooling
  (1216 to 1241; small-state bias), and the tuning negative result.
- 2.7 Disease-specificity (Table 4). Chikungunya: LightGBM alone (82.3) beats the ensemble (91.6).
- 2.8 Optional-track summary (city-level dengue and chikungunya) in one paragraph, details to SI.

### 3. Discussion
- 3.1 "Best on average" is dangerous under regime shift; ensembles and appropriately widening
  uncertainty; the mechanistic model's bootstrap intervals as a positive example.
- 3.2 Metric choice is a reporting-standards issue for the whole field; recommend dual-metric
  reporting; connect to forecast-hub practice.
- 3.3 Why simple methods resist deep learning in small-sample, few-series epidemic forecasting;
  consistency with M-competition and COVID/Flu-hub evidence.
- 3.4 Conformal calibration as cheap transferable insurance; link to adaptive conformal and
  regime shift (Mexico online-recalibration precedent).
- 3.5 Negative results as content: forecast-relevant vs observed climate signal; heterogeneity
  limits naive pooling. Where a Bayesian INLA-DLNM member would be expected to help and why we
  could not include one here.
- 3.6 Limitations: partially-resolved fold 4; state aggregation discards sub-state asynchrony;
  two diseases, one country; single-team scope; retrospective (no true prospective score yet).
- 3.7 Future work: the September forecast phase; hierarchical reconciliation; a Bayesian
  spatiotemporal member; municipal features; operational deployment.

### 4. Materials and Methods (EPIFORGE-compliant, detailed)
- 4.1 Data and targets (sources, provenance, aggregation, ES exclusion, covariates).
- 4.2 Backtest design and leakage control (fold derivation, 15-week gap, cutoff filtering, tests).
- 4.3 Scoring (WIS definition and pinball/CRPS identity, relative WIS, normalized WIS, coverage,
  decomposition; verification against the platform scorer).
- 4.4 Models (baselines; LightGBM quantile + CQR on the synthetic-origin panel; global GRU with
  per-state embeddings + negative-binomial head as a deep ensemble; semi-mechanistic Richards
  bootstrap; ensembles by Vincentization and inverse-WIS).
- 4.5 Conformal recalibration (multiplicative CQR, calibration-fold discipline, forecast-phase
  protocol).
- 4.6 Reproducibility (code, harness, tests, provenance, submissions).

### Supporting Information
S1 WIS/CRPS math and verification; S2 full feature list and frozen hyperparameters; S3 GRU
architecture; S4 mechanistic derivation; S5 per-state and per-fold tables; S6 ensemble-weight and
member-selection robustness (the subset sweep); S7 EPIFORGE 2020 checklist item-by-item; S8 EDA
figures (climate lags, Koppen gradient, ENSO, peak-week, small multiples); S9 provenance and
reproducibility statement.

## 4. Figure and table plan

Main figures (target 5, plus 1 optional):
- **Fig 1 (build):** study design and data overview. National weekly trend with the 2024 outlier
  annotated; the four-fold timeline with train-cutoff/gap/target windows; a state map. Compose from
  `national_weekly_trends.png` plus a new folds-timeline panel.
- **Fig 2 (have, refresh):** WIS by model x season, log scale. `paper_wis_by_fold.png`. Update to
  current numbers and add `ensemble_conformal`.
- **Fig 3 (build):** metric-disagreement. Slope/dumbbell chart of model rank under raw mean WIS vs
  normalized WIS, showing the reordering. Extend `paper_relative_wis.png`.
- **Fig 4 (have, extend):** calibration reliability, empirical vs nominal coverage, before/after
  conformal. `paper_coverage.png`.
- **Fig 5 (build):** conformal recalibration effect. WIS by fold before/after, plus coverage
  before/after, plus the widening factors. New.
- **Fig 6 (optional, build):** state-level normalized-WIS choropleth tied to the Koppen/climate
  gradient (geopandas + `shape_muni.gpkg`), linking skill to the EDA climate zones.

Main tables (4):
- **Table 1:** dengue leaderboard, all models, raw WIS + normalized WIS (all and ex-2024) +
  coverage 50/80/90/95. Lead with `ensemble_conformal`.
- **Table 2:** dengue WIS by fold (the regime shift, numeric).
- **Table 3:** ablations (climate covariates, hierarchical pooling, tuning), each with the WIS
  delta and the one-line reason.
- **Table 4:** chikungunya leaderboard.

Numbers to embed are the current committed values (dengue raw WIS: conformal 1216, vincent 1281,
invwis 1287, lgbm 1299, mechanistic 1303, climatological 1327, gru 1373, seasonal 1459, naive
1664; normalized all/ex-2024 per `final_leaderboard_normalized.csv`; chik per
`chik_final_leaderboard.csv`).

## 5. References plan (~30-40)

Dengue/arbovirus burden and Brazil; Infodengue; the 2024 sprint paper (Araujo et al., PNAS 2026);
WIS and interval scoring (Bracher et al. 2021; Gneiting and Raftery CRPS); quantile regression;
conformalized quantile regression (Romano et al. 2019) and adaptive conformal (Gibbs and Candes);
LightGBM (Ke et al.); GRU (Cho et al.) and deep ensembles (Lakshminarayanan et al.); DLNM
(Gasparrini) and climate-dengue (Lowe et al.); forecast hubs (Cramer et al.; Reich et al.;
European hub); Vincentization/quantile averaging; ENSO-dengue; EPIFORGE 2020 (Pollett et al.);
Richards growth curve; negative-binomial count models.

## 6. Criteria checklist (must all pass before submission)

Top-journal readiness:
- [ ] Clear, generalizable contribution beyond one challenge (findings 1-4 are field-level).
- [ ] Rigorous, leakage-controlled evaluation with a pre-registered-style protocol.
- [ ] Honest negative results and limitations.
- [ ] Complete, reproducible methods; public code and data; tests.
- [ ] Publication-quality figures and tables; every claim sourced.
- [ ] Complete reference set; correct positioning vs prior work.
- [ ] Style rules in section 0 satisfied.

EPIFORGE 2020 mapping (each item to a manuscript location):
- Purpose/objectives -> Intro 1.3-1.4. Forecast targets -> 2.1, Methods 4.1.
- Data sources and preprocessing -> Methods 4.1; covariates -> 4.1, SI S2.
- Model descriptions and assumptions -> Methods 4.4, SI S2-S4.
- Uncertainty quantification -> Methods 4.3-4.5 (quantiles, CQR, conformal), Results 2.4-2.5.
- Validation approach -> Methods 4.2 (backtest, leakage), 4.3.
- Performance metrics -> Methods 4.3; baselines -> Results 2.2, Table 1.
- Results with uncertainty and calibration -> Results 2.2-2.7, Figs 2-5.
- Interpretation, generalizability, limitations -> Discussion 3.1-3.7.
- Data/code availability -> Methods 4.6, SI S9.

## 7. Drafting sequence

1. Lock Tables 1-4 and refresh/build Figures 2, 4 (data-ready now).
2. Methods (data-independent, EPIFORGE backbone).
3. Results 2.1-2.7 against the locked tables/figures.
4. Introduction and Discussion.
5. Abstract, Author summary, Significance.
6. Build Figures 1, 3, 5 (and optional 6); SI sections; references.
7. Full criteria and EPIFORGE pass; compile PDF; consistency check on every number.
