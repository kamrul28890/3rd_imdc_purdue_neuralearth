# EPIFORGE 2020 compliance — item-by-item (2026-09-12)

The real 19-item checklist (Pollett SM, et al. Recommended reporting items for epidemic
forecasting and prediction research: the EPIFORGE 2020 guidelines. *PLOS Medicine* 18(10):
e1003793, 2021 — already cited in `imdc_paper.tex` as `pollett2021`), fetched verbatim from the
publisher 2026-09-12, mapped against the current draft. Supersedes the informal 12-category
paraphrase currently in `imdc_paper_SI.tex` S5 Text — see `docs/MANUSCRIPT_GAP_AUDIT.md` §5 for
why the upgrade matters. Use this table to replace S5 Text's table directly (Workflow step 4);
`Status` is what to write in the SI's own status column once the two MET-with-caveat rows are
closed.

| # | Item (verbatim) | Manuscript location | Status |
|---|---|---|---|
| 1 | Describe the study as forecast or prediction research in at least the title or abstract. | Title ("...forecasting in Brazil"); Abstract, first sentence | MET |
| 2 | Define the purpose of study and forecasting targets. | Intro §1.3 (`sec:` contributions list); Results §2.1 `sec:design` | MET |
| 3 | Fully document the methods. | Methods §4 (`sec:methods`), all subsections | MET |
| 4 | Identify whether the forecast was performed prospectively, in real time, and/or retrospectively. | Explicit for the 4 backtest folds (retrospective) and the submitted 2026-27 forecast (prospective, unresolved) in Results §2.1 and Limitations | MET |
| 5 | Explicitly describe the origin of input source data, with references. | Methods §4.2 `sec:methods-data`; SI S2 Text | MET |
| 6 | Provide source data with publication, or document reasons as to why this was not possible. | `Data and code availability`; public repo, provenance manifest | MET |
| 7 | Describe input data processing procedures in detail. | Methods §4.2; SI S2 Text feature list | MET |
| 8 | State and describe the model type, and document model assumptions, including references. | Methods §4.4 `sec:methods` (Models); SI S2 Text, S1 Table | MET |
| 9 | Make the model code available, or document the reasons why this is not possible. | `Data and code availability`; full public repo with tests | MET |
| 10 | Describe the model validation, and justify the approach. | Methods §4.3 `sec:methods-backtest` (leakage-safe backtest design, 15-week reporting gap, fold derivation) | MET |
| 11 | Describe the forecast accuracy evaluation method used, with justification. | Methods §4.3 `sec:methods-scoring` (WIS, decomposition); SI S1 Text (formula, worked example, verification against platform scorer) | MET |
| 12 | Where possible, compare results to a benchmark or other comparator model, with justification. | Naive and seasonal-naive baselines throughout; Table 1/2 | MET |
| 13 | Describe the forecast horizon, with justification of its length. | Methods §4.2 (16-67 week horizon from the 15-week reporting gap, tied to the challenge's own EW41-EW40 target window) | MET |
| 14 | Present and explain uncertainty of forecasting results. | Results §2.4-2.6 (calibration, conformal recalibration); coverage tables throughout | MET |
| 15 | Briefly summarize the results in nontechnical terms, including a nontechnical interpretation of forecast uncertainty. | `Author summary` section | MET. **Correction (2026-09-12):** an earlier draft of this row claimed the summary "leans on named jargon (normalized WIS, conformal recalibration)"; that was wrong, checked against the actual text, which already uses plain language throughout ("a full range of plausible outcomes with calibrated uncertainty," "a light statistical recalibration that stretched the ensemble's prediction intervals"). Length (220 words after the 2026-09-12 density pass) is not a compliance issue for this item, and venue word limits are deferred until a venue is chosen. |
| 16 | If results are published as a data object, encourage a time-stamped version number. | Provenance manifest (SI S7 Text) records data checksums + code revision per reported number | MET (the manifest *is* the time-stamped version record; not a separate "data object" release, but satisfies the item's intent) |
| 17 | Describe the weaknesses of the forecast, including weaknesses specific to data quality and methods. | Discussion, `Limitations` §3.6 (fold-4 partial resolution, municipal aggregation, single-team scope, Oropouche-contamination caveat added 2026-09-12) | MET |
| 18 | If applicable to a specific epidemic, comment on potential implications for public health action and decision-making. | Discussion §3.1 (regime-shift robustness → operational model choice); mentions of the challenge's Ministry-of-Health-facing purpose in Intro | MET, could be strengthened with one explicit sentence tying back to the challenge's stated MoH decision-support purpose (see `araujo2026`'s own framing) — MINOR, not required |
| 19 | If applicable to a specific epidemic, comment on how generalizable it may be across populations. | Limitations §3.6 (two diseases, one country); the 2026-09-12 scope-contrast addition (26 states vs. `araujo2026`'s 5) | MET |

## Summary

**18 of 19 fully MET as of 2026-09-12** (item 15 reclassified from PARTIAL to MET after checking
the actual Author Summary text, see the correction in that row). Item 16 is met by a reasonable
interpretation (the provenance manifest serves as the time-stamped version record) rather than
literally; state that interpretation explicitly in the SI text rather than leaving it implicit, so
a reviewer doesn't read it as skipped. Item 18 is met but could be strengthened by one sentence
tying results back to the challenge's Ministry-of-Health decision-support purpose.

Neither remaining improvement is a compliance failure. Both are one-sentence additions, folded
into Workflow step 3 (the SI EPIFORGE upgrade).

## What to write in the manuscript

Add one sentence near the SI's EPIFORGE table (or in the main text's Methods/Reproducibility
subsection) stating the compliance claim explicitly and citing this checklist by name and item
count, e.g.: "This manuscript follows the EPIFORGE 2020 guidelines (Pollett et al. 2021); Table
S3 maps all 19 recommended items to their location in this manuscript." Naming the exact item
count (19) rather than a vague "we follow EPIFORGE" is itself part of doing this rigorously —
it's checkable, which is the entire point of a reporting checklist.
