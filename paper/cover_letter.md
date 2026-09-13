# Cover letter

**Status:** drafted 2026-09-12, venue-neutral. Three slots marked `[[ ]]` need filling once a venue
is chosen (see `docs/MANUSCRIPT_GAP_AUDIT.md` §1: the venue is deliberately not yet fixed). Every
number below is verified against the committed results (workflow step 7). Keep it to one page.

---

Dear [[Editor name / "Editors"]],

We submit **"Probabilistic dengue and chikungunya forecasting in Brazil: metric-dependent model
rankings, regime-shift fragility, and conformal ensembles"** for consideration as a Research Article
in *[[Journal]]*.

Brazil's 2024 dengue season broke the national record by roughly fourfold. Forecasts that support
preparedness must therefore be reliable in exactly the season that resembles nothing in the training
record, and it is unsettled which modeling paradigm delivers that. We report a reproducible,
leakage-controlled comparison of five probabilistic forecasting model families on the 3rd
Infodengue-Mosqlimate Dengue Challenge, covering four retrospective seasons and all 26 mandatory
Brazilian states, with every model fit and scored through a single backtesting harness.

Four results generalize beyond this challenge:

1. **Model rankings invert under regime shift.** A global recurrent network is the most accurate
   model in ordinary seasons yet the worst of all on the 2024 outlier, worse than a naive baseline.
   A model chosen on average performance carries hidden risk in precisely the season that matters.
2. **The identity of the best model depends on the evaluation metric.** Magnitude-weighted mean WIS
   and scale-free normalized WIS reward accuracy on different subsets of a highly skewed set of
   states and select different winners. We argue epidemic-forecasting leaderboards should report
   both; a single aggregate can mislead.
3. **Conformal recalibration of the ensemble intervals is cheap, transferable insurance.** Factors
   estimated on one tuning season and applied unchanged elsewhere improve mean WIS from 1234 to
   1162 and normalized WIS from 0.595 to 0.561, a 6.2 percent out-of-sample reduction. The gain is
   asymmetric by design: 277 WIS units recovered in the outbreak season against costs of 5 and 9
   units in ordinary ones.
4. **Added complexity did not pay.** Observed-climate covariates, hierarchical spatial pooling, a
   hyperparameter search, and an alternative ensemble-combination rule each failed to improve
   forecasts where it mattered. We report these as findings, including one case where our own
   validation rule, applied mechanically, would have recommended a change that traded away the
   outbreak-season robustness the design exists to provide.

The best model is also disease-specific: for chikungunya, gradient boosting alone outperforms the
ensemble that wins for dengue.

We think this suits *[[Journal]]* because [[venue-specific fit: for a computational-methods venue,
emphasize the leakage-audited harness, the automated test suite, and full reproducibility; for an
epidemiology or global-health venue, emphasize the Ministry-of-Health decision-support framing and
the operational onset, peak, and burden metrics]]. Relative to recent multi-country ensemble
evaluations, this study trades breadth for depth: one country, two diseases, a full-season 16 to 67
week horizon, and one public pipeline in which every reported number can be regenerated.

The complete pipeline, backtesting harness, per-state scored predictions, provenance manifest, and
validated submissions for all four official tracks are openly available at
https://github.com/kamrul28890/3rd_imdc_purdue_neuralearth, with an automated test suite covering
leakage safety, scoring correctness, and every model family. The manuscript follows the EPIFORGE
2020 reporting guidelines, mapped item by item in the Supporting Information.

This manuscript is original, is not under consideration elsewhere, and all authors have read and
approved it. The authors declare no competing interests and received no specific funding. The study
used only aggregate, publicly available surveillance data and required no ethical approval.

[[Suggested reviewers, if the submission system asks. A decision for the authors, not a default.
Candidates should be independent of the challenge organizers to avoid a conflict, since the
organizers are cited throughout as the source of the data and the predecessor sprint.]]

Thank you for considering our work.

Sincerely,
Md Kamruzzaman Kamrul, on behalf of all authors
Purdue University, West Lafayette, IN, USA
kamrul28890@gmail.com
