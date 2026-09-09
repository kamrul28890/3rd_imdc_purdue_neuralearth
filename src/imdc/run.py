"""One parametrized CLI for running any model's backtest (IMPROVEMENTS.md Sec 3.3).

    KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.run --model xgb --disease dengue
    python -m imdc.run --model naive --disease chikungunya --ufs SP RJ --out /tmp/naive.csv
    python -m imdc.run --list

Deliberately ADDITIVE, not a replacement for the seven existing run_*.py scripts
(run_baselines/run_ml/run_dl/run_mechanistic/run_ensemble/run_chikungunya/run_cities):
those are wired into the Makefile's `make reproduce` target and the paper/README's
reproducibility claims, and each has its own bespoke "merge into the combined
leaderboard" behavior this CLI doesn't try to replicate. This is the "add one new
model without copy-pasting a whole script" fast path the doc asks for; consolidating
the seven scripts onto it (and repointing the Makefile) is future work once this has
been validated over time, not something to do the same week as a live competition
submission that depends on those exact scripts.
"""
import argparse
import time

from imdc.config import MANDATORY_UFS
from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel, NaiveModel, SeasonalNaiveModel
from imdc.evaluation.harness import run_backtest, score_backtest, summarize

# name -> zero-arg factory. Each factory takes `disease` and returns a fresh Forecaster;
# hyperparameters match the defaults each model's own run_*.py script uses today.
MODEL_REGISTRY = {
    "naive": lambda disease: NaiveModel(),
    "seasonal_naive": lambda disease: SeasonalNaiveModel(),
    "climatological": lambda disease: ClimatologicalQuantileModel(),
    "lgbm": lambda disease: _import("imdc.models.ml_boosted", "LGBMQuantileModel")(disease=disease),
    "xgb": lambda disease: _import("imdc.models.xgb_quantile", "XGBQuantileModel")(disease=disease),
    "gru": lambda disease: _import("imdc.models.dl_sequence", "DLSequenceModel")(
        disease=disease, n_ensemble=5, epochs=30),
    "gru_wis": lambda disease: _import("imdc.models.dl_sequence_wis", "DLSequenceWISModel")(
        disease=disease, n_ensemble=5, epochs=30),
    "mechanistic": lambda disease: _import("imdc.models.mechanistic", "MechanisticTrajectoryModel")(disease=disease),
    "mechanistic_nsub": lambda disease: _import(
        "imdc.models.mechanistic_nsub", "NSubEpidemicMechanisticModel")(disease=disease),
    "hierarchical": lambda disease: _import(
        "imdc.models.hierarchical", "HierarchicalClimatologicalModel")(),
    "prophet": lambda disease: _import("imdc.models.prophet_model", "ProphetModel")(disease=disease),
    "prophet_v2": lambda disease: _import("imdc.models.prophet_v2", "ProphetV2Model")(disease=disease),
    "sarimax": lambda disease: _import("imdc.models.sarimax_model", "SarimaxModel")(disease=disease),
    "sarimax_v2": lambda disease: _import("imdc.models.sarimax_v2", "SarimaxV2Model")(disease=disease),
    "surge_template": lambda disease: _import("imdc.models.surge_template", "SurgeTemplateModel")(disease=disease),
    "climate_surge": lambda disease: _import("imdc.models.climate_surge", "ClimateSurgeModel")(disease=disease),
}


def _import(module: str, name: str):
    """Lazy import so `--list` and unrelated models don't pay for torch/statsmodels/prophet
    import time or require every optional dependency to be installed.
    """
    import importlib
    return getattr(importlib.import_module(module), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), help="model to backtest")
    parser.add_argument("--disease", default="dengue", choices=["dengue", "chikungunya"])
    parser.add_argument("--ufs", nargs="+", default=MANDATORY_UFS, help="states to include (default: all 26)")
    parser.add_argument("--out", default=None, help="write per-unit scored rows to this CSV path")
    parser.add_argument("--list", action="store_true", help="list registered model names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(MODEL_REGISTRY):
            print(name)
        return

    if not args.model:
        parser.error("--model is required (or pass --list to see available models)")

    folds = get_folds(args.disease)
    factory = lambda: MODEL_REGISTRY[args.model](args.disease)

    t0 = time.time()
    preds = run_backtest(factory, folds, disease=args.disease, ufs=args.ufs)
    scored = score_backtest(preds, disease=args.disease, folds=folds)
    print(f"{args.model} ({args.disease}): {len(scored)} scored rows in {time.time() - t0:.1f}s")

    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    by_model = summarize(scored, by=["model"])
    by_fold = summarize(scored, by=["model", "fold_id"])
    print("\n=== Leaderboard (mean over all states/folds/horizons) ===")
    print(by_model.to_string(index=False))
    print("\n=== By fold ===")
    print(by_fold.sort_values("fold_id").to_string(index=False))


if __name__ == "__main__":
    main()
