# Size-aware 2026-27 dengue forecast (prospective comparison)

Not a challenge submission. The submitted forecast is in `submissions/validation/dengue/season_2027/` (global conformal recalibration). This is the same ensemble (identical medians) with the size-aware recalibration instead, frozen with a = 1.00, b = 0.10 (tuned on the 2022-23 season only) and generated using no data after the forecast origin. It was committed on 2026-11-09, four weeks into the target window, before any 2026-27 observation was examined. Scoring both once the season resolves is the out-of-sample test of the size-aware rule (strictly: on the target weeks after the commit date).

- generator: `scripts/generate_forecast_2026_27_size_aware.py`
- training data through: 2026-06-21
- target window: 2026-10-11 to 2027-10-03
- national median total: 1,088,267 (submitted: 1,088,267)
- national sum of 90% bounds, season total: 190,205 to 6,474,629 (submitted: 76,099 to 5,924,061)
- SHA-256 of the 26 state files, concatenated in MANDATORY_UFS order: `fddffdcf0b091e41a145d7bb9cc9340e261cb45f0cf7d78bf23e2ba1d41999b8`
