# IMDC 2026 — reproducible pipeline orchestration.
#
# `make reproduce` regenerates every result, figure, submission and manifest from
# the raw data, in dependency order. Deterministic models (baselines, LightGBM,
# mechanistic) reproduce bit-for-bit; the GRU up to Apple-MPS float non-determinism.
#
# Assumes the `py310` conda env is active (see README). KMP_DUPLICATE_LIB_OK is set
# because LightGBM/XGBoost and PyTorch each link their own libomp.

export KMP_DUPLICATE_LIB_OK = TRUE
PY = python

.PHONY: help test baselines ml dl mechanistic ensemble chikungunya cities \
        submissions figures reports manifest reproduce reproduce-fast clean

help:
	@echo "Targets:"
	@echo "  test          run the 62-test suite"
	@echo "  reproduce     full pipeline: all models -> ensemble -> submissions -> figures -> manifest"
	@echo "  reproduce-fast same, but skip the ~55-min GRU (reuses committed gru_scored.csv)"
	@echo "  <step>        baselines|ml|dl|mechanistic|ensemble|chikungunya|cities|submissions|figures|reports|manifest"

test:
	$(PY) -m pytest tests/ -q

# --- dengue state-level model runs (each writes results/metrics/<model>_scored.csv) ---
baselines:
	$(PY) -m imdc.evaluation.run_baselines
ml:
	$(PY) -m imdc.models.run_ml
dl:
	$(PY) -m imdc.models.run_dl
mechanistic:
	$(PY) -m imdc.models.run_mechanistic

# --- ensemble + optional tracks ---
ensemble:
	$(PY) -m imdc.models.run_ensemble
chikungunya:
	$(PY) -m imdc.models.run_chikungunya
cities:
	$(PY) -m imdc.evaluation.run_cities

# --- artifacts ---
submissions:
	$(PY) -m imdc.submission.generate
figures:
	$(PY) scripts/make_figures.py
reports:
	cd reports && tectonic data_findings_report.tex && tectonic modeling_results_report.tex
	cd paper && tectonic imdc_paper.tex
manifest:
	$(PY) scripts/make_manifest.py

# Full reproduction (GRU is the long pole, ~55 min on M1).
reproduce: baselines ml dl mechanistic ensemble chikungunya cities submissions figures manifest
	@echo "Reproduction complete. See RESULTS.md for the provenance manifest."

# Faster path: reuse the committed GRU predictions (bit-reproducibility of the GRU is
# MPS-limited anyway), regenerate everything else deterministically.
reproduce-fast: baselines ml mechanistic ensemble chikungunya cities submissions figures manifest
	@echo "Fast reproduction complete (GRU predictions reused). See RESULTS.md."

clean:
	rm -f results/metrics/*_scored.csv results/metrics/*leaderboard*.csv results/metrics/*.log
	rm -rf submissions/validation
	@echo "Cleaned generated metrics and submissions (raw data and committed figures untouched)."
