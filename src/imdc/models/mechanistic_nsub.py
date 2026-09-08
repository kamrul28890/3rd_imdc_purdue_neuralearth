"""n-sub-epidemic-enhanced mechanistic bootstrap (Chowell et al. 2022, PLOS Comp Biol).

The literature framework fits a season as a sum of several overlapping generalized-logistic
"sub-epidemic" curves (capturing plateaus, resurgences, multi-peak seasons that a single Richards
curve can't) and is normally used to extrapolate a *currently unfolding, partially observed*
outbreak forward. That doesn't map cleanly onto our task: we forecast a full season ahead with zero
visibility into it (by design, via the 15-week reporting gap), so there's no partial current-season
curve to fit and extrapolate from at forecast time.

Adaptation used here: fit a 2-sub-epidemic curve to each *historical* (already-complete) season,
and add its smoothed curve as an *additional* entry in `MechanisticTrajectoryModel`'s bootstrap pool
alongside the raw historical curve, but only when the two sub-epidemics are genuinely separate waves
(see `_is_genuinely_bimodal` - an AICc-based "which fits better" comparison was tried first and
discarded, since a more flexible 8-parameter curve fits *any* single-peaked season at least as well
as a 4-parameter one and so wins essentially always regardless of whether the season is actually
multi-wave; see that function's docstring for the regression this caused and how it was diagnosed).
The result is richer, shape-diverse synthetic trajectories for seasons with real multi-wave
structure, without abandoning the base model's deliberate choice to bootstrap real historical
variability rather than only model-fitted curves (docs/PLAN.md Sec 5.2), and without duplicating
weight onto single-peaked seasons (including extreme ones) that a raw curve already represents.
"""
import numpy as np
from scipy.optimize import curve_fit

from imdc.models.mechanistic import MechanisticTrajectoryModel, richards_cumulative


def _two_sub_epidemic_cumulative(t, K1, r1, tp1, a1, K2, r2, tp2, a2):
    return richards_cumulative(t, K1, r1, tp1, a1) + richards_cumulative(t, K2, r2, tp2, a2)


def _fit_two_sub(y, t):
    K0 = y[-1] * 0.7
    mid = len(t) // 2
    p0 = [K0, 0.3, float(t[max(1, np.argmax(np.diff(y[:mid], prepend=0)))]), 1.0,
          y[-1] * 0.5, 0.3, float(t[mid + max(1, np.argmax(np.diff(y[mid:], prepend=0)))]), 1.0]
    lo = [y[-1] * 0.1, 1e-3, 1, 1e-3] * 2
    hi = [y[-1] * 5, 5.0, len(y), 50] * 2
    popt, _ = curve_fit(_two_sub_epidemic_cumulative, t, y, p0=p0, bounds=(lo, hi), maxfev=8000)
    fitted = _two_sub_epidemic_cumulative(t, *popt)
    rss = float(np.sum((y - fitted) ** 2))
    return fitted, rss, 8, popt


_MIN_PEAK_SEPARATION_WEEKS = 8.0
_MIN_SUBEPIDEMIC_SHARE = 0.15


def _is_genuinely_bimodal(popt) -> bool:
    """True only if the two sub-epidemics are well-separated in time and both contribute
    materially - i.e. this is an actual multi-wave season, not just a more-flexible single peak.

    AICc alone cannot make this distinction: an 8-parameter two-sub-epidemic curve fits *any*
    single-peaked season at least as well as a 4-parameter Richards curve simply by having more
    freedom (confirmed empirically - it won for every one of SP's 14 historical seasons in fold 4,
    including the unmistakably single-peaked 2024 outbreak). Augmenting the bootstrap pool with
    such a "fancier-fit single peak" adds a near-duplicate of the raw curve, not real shape
    diversity - and for an already-extreme outbreak curve, that duplicate quietly doubles its
    weight in the resampling pool, which is what caused a real fold-4 regression (WIS 216 -> 307)
    tracked down to exactly this curve. Requiring genuine separation plus a minimum share for each
    sub-epidemic keeps the augmentation to seasons where it adds something the raw curve doesn't
    already have.
    """
    K1, _, tp1, _, K2, _, tp2, _ = popt
    if abs(tp1 - tp2) < _MIN_PEAK_SEPARATION_WEEKS:
        return False
    share1 = K1 / max(K1 + K2, 1e-9)
    return _MIN_SUBEPIDEMIC_SHARE <= share1 <= (1 - _MIN_SUBEPIDEMIC_SHARE)


def smoothed_incidence_curve(incidence_curve: np.ndarray) -> "np.ndarray | None":
    """Smoothed two-sub-epidemic curve, but only when genuinely bimodal (see
    `_is_genuinely_bimodal`); None otherwise, so single-peaked seasons are left unaugmented rather
    than duplicated by a same-shaped, more-flexibly-fit curve.
    """
    y = np.cumsum(np.nan_to_num(incidence_curve))
    t = np.arange(1, len(y) + 1, dtype=float)
    if y[-1] <= 0:
        return None
    try:
        fitted2, _, _, popt = _fit_two_sub(y, t)
    except Exception:
        return None
    if not _is_genuinely_bimodal(popt):
        return None
    weekly = np.diff(fitted2, prepend=0.0)
    return np.maximum(weekly, 0.0)


class NSubEpidemicMechanisticModel(MechanisticTrajectoryModel):
    """MechanisticTrajectoryModel with each historical season's bootstrap pool augmented by a
    smoothed 1-or-2-sub-epidemic fitted curve, alongside (not replacing) the raw historical curve."""

    name = "mechanistic_nsub"

    def fit(self, train_df, fold, covariates=None):
        super().fit(train_df, fold, covariates)
        for uf, mats in list(self._trajectories.items()):
            smoothed = []
            for curve in mats:
                s = smoothed_incidence_curve(curve)
                if s is not None:
                    smoothed.append(s)
            if smoothed:
                self._trajectories[uf] = np.vstack([mats, np.vstack(smoothed)])
        return self
