# SARIMAX-augmented, size-aware 2026-27 dengue forecast (prospective comparison)

Not a challenge submission. The submitted forecast is in `submissions/validation/dengue/season_2027/` (three-member ensemble, global conformal recalibration); a size-aware version of it is in `submissions/prospective/size_aware_2026_27/`. This one adds the per-fold-order SARIMAX v2 as a fourth member (per-quantile median), which changes the median as well as the intervals, and applies the size-aware scale frozen with a = 1.00, b = 0.10 (tuned on the 2022-23 season only). It uses no data after the forecast origin and was generated on 2026-11-09, inside the target window but before any 2026-27 observation was examined; the git commit that adds it timestamps it. Scoring all three once the season resolves is the out-of-sample test of the post-hoc variants (strictly: on the target weeks after the commit date).

- generator: `scripts/generate_forecast_2026_27_sarimax_size_aware.py`
- training data through: 2026-06-21
- target window: 2026-10-11 to 2027-10-03
- national median total: 838,033 (submitted: 1,088,267; SARIMAX member alone: 396,495)
- national sum of 90% bounds, season total: 134,122 to 5,170,193 (submitted: 76,099 to 5,924,061)
- `sarimax_v2_member.csv`: the SARIMAX member's own forecast, for decomposing the result
- SHA-256 of the 26 state files, concatenated in MANDATORY_UFS order: `88839b7d69aa3734574c32d11adfea0c5327411cb1dbb4ac591e28219a9ecf9c`
