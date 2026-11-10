"""Backtest SARIMAX v2 with its ARIMA orders re-selected on every fold (the "per-fold" variant).

This is the generator of results/metrics/sarimax_v2_perfold_scored.csv, which run_ensemble.py
now loads as the fourth member of the SARIMAX-augmented ensemble (paper Sec. "sizeaware").
Orders are selected per state from each fold's own training data only (held-out tail of the
training series), so no fold sees its target window. Folds are independent, so they can run as
separate processes and be concatenated:

    python scripts/run_sarimax_v2_perfold.py --fold 1 --out part1.csv   (... 2, 3, 4)
    python scripts/run_sarimax_v2_perfold.py --merge part1.csv part2.csv part3.csv part4.csv \
        --out results/metrics/sarimax_v2_perfold_scored.csv

About 40 minutes per fold for 26 states.

Reruns are close but not bit-identical: the ARIMA optimizer (and hence some per-state order choices)
is sensitive to floating-point details such as BLAS threading. The committed file is the September
2026 run (mean WIS 1527.6); a November 2026 rerun with this script (four folds in parallel,
OMP_NUM_THREADS=2) gave 1525.9, with the largest change on fold 1 (484.0 -> 469.9). Rebuilt from the
rerun, the SARIMAX-augmented size-aware ensemble scores Rv 0.882 instead of 0.884 and mean WIS
1160.6 instead of 1160.7, with the same size-aware parameters.
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest
from imdc.models.sarimax_v2 import SarimaxV2Model

BACKTEST_FOLDS = (1, 2, 3, 4)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, nargs="+", default=list(BACKTEST_FOLDS))
    p.add_argument("--merge", nargs="+", help="concatenate already-scored per-fold CSVs instead")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if args.merge:
        out = pd.concat([pd.read_csv(f) for f in args.merge], ignore_index=True)
        out = out.sort_values(["fold_id", "uf", "date"]).reset_index(drop=True)
    else:
        folds = [f for f in get_folds("dengue") if f.id in args.fold]
        preds = run_backtest(lambda: SarimaxV2Model(disease="dengue", select_orders=True), folds,
                             disease="dengue")
        out = score_backtest(preds, disease="dengue", folds=folds)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}; mean WIS {out['wis'].mean():.1f}")


if __name__ == "__main__":
    main()
