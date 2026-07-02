"""Global GRU sequence forecaster with a Negative-Binomial head (DL model family).

Deliberately small (plan Sec 4.1): a 1-layer GRU encoder (hidden 48) shared
across all 26 states, a learned per-state embedding, static climate-zone
covariates, and a per-horizon MLP head that outputs Negative-Binomial (mu, alpha).
Trained with the NB negative log-likelihood in count space via a population
offset (mu = pop/1e5 * exp(f)), so the network predicts a log-incidence-rate.
Quantiles are analytic (scipy.stats.nbinom.ppf); a small deep ensemble of
independently-seeded runs is pooled by quantile averaging.

Why small: the synthetic-origin panel is few-series and heavily autocorrelated;
the M4/M5 literature shows large nets overfit this regime. This model is a
genuine comparison point, expected to be competitive-not-dominant.

Requires KMP_DUPLICATE_LIB_OK=TRUE (torch + lightgbm both link libomp).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from epiweeks import Week
from scipy.stats import nbinom

from imdc.config import INTERVAL_LEVELS, MANDATORY_UFS, QUANTILE_LEVELS

# quantile-level index pairs bounding each central interval (order matches QUANTILE_LEVELS)
_INTERVAL_BOUND_IDX = {95: (0, 8), 90: (1, 7), 80: (2, 6), 50: (3, 5)}
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_no_leakage
from imdc.features.panel import INCIDENCE_SCALE, state_climate_full, state_population, state_static_features

SEQ_LEN = 104
HORIZONS = np.arange(1, 68)
_SEQ_NUMERIC = ["log_inc", "temp_med", "precip_med", "rel_humid_med"]


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def _state_weekly_frame(fold, disease: str) -> pd.DataFrame:
    """Per (uf, date) weekly features for sequences: log_inc + climate + epiweek harmonics."""
    cases = cutoff_filter(load_cases(disease), fold.train_cutoff)
    assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} dl cases")
    state = aggregate_cases_to_state(cases)
    pop = state_population()
    df = state.copy()
    df["year"] = df["date"].dt.year
    ymin, ymax = pop["year"].min(), pop["year"].max()
    df["pop_year"] = df["year"].clip(ymin, ymax)
    df = df.merge(pop.rename(columns={"year": "pop_year"}), on=["uf", "pop_year"], how="left")
    df["log_inc"] = np.log1p(df["casos"] / df["population"] * INCIDENCE_SCALE)

    climate = cutoff_filter(state_climate_full(), fold.train_cutoff)
    df = df.merge(climate[["uf", "date", "temp_med", "precip_med", "rel_humid_med"]],
                  on=["uf", "date"], how="left")
    df["epiweek"] = np.array([Week.fromdate(d).week for d in df["date"]])
    df = df.sort_values(["uf", "date"]).reset_index(drop=True)
    return df


def _epiweek_harmonics(epiweeks: np.ndarray) -> np.ndarray:
    w = epiweeks.astype(float)
    return np.column_stack([
        np.sin(2 * np.pi * w / 52), np.cos(2 * np.pi * w / 52),
        np.sin(4 * np.pi * w / 52), np.cos(4 * np.pi * w / 52),
    ])


class GRUForecaster(nn.Module):
    def __init__(self, n_seq_feat: int, n_static: int, n_states: int, hidden: int = 48,
                 emb_dim: int = 8, horizon_feat: int = 5, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_seq_feat, hidden, batch_first=True)
        self.state_emb = nn.Embedding(n_states, emb_dim)
        dec_in = hidden + emb_dim + n_static + horizon_feat
        self.head = nn.Sequential(
            nn.Linear(dec_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),  # -> (log-incidence-rate f, log_alpha)
        )

    def forward(self, seq, state_idx, static, horizon_feats):
        # seq: (B,L,Fs)  static: (B,Fst)  horizon_feats: (B,H,hf)
        _, h = self.gru(seq)
        h = h[-1]  # (B, hidden)
        emb = self.state_emb(state_idx)  # (B, emb)
        ctx = torch.cat([h, emb, static], dim=1)  # (B, C)
        B, H, hf = horizon_feats.shape
        ctx_rep = ctx.unsqueeze(1).expand(-1, H, -1)
        dec_in = torch.cat([ctx_rep, horizon_feats], dim=2)  # (B,H,C+hf)
        out = self.head(dec_in)  # (B,H,2)
        f = out[..., 0]
        log_alpha = out[..., 1]
        return f, log_alpha


def _negbin_nll(y, mu, log_alpha):
    """NB2 negative log-likelihood; mu>0, alpha=exp(log_alpha)."""
    alpha = torch.exp(log_alpha).clamp(min=1e-4, max=1e4)
    r = 1.0 / alpha
    eps = 1e-8
    log_r_over = r * (torch.log(r + eps) - torch.log(r + mu + eps))
    log_y_over = y * (torch.log(mu + eps) - torch.log(r + mu + eps))
    ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
          + log_r_over + log_y_over)
    return -ll


class DLSequenceModel:
    """Deep-ensemble GRU forecaster implementing the harness fit/predict protocol."""

    name = "gru_negbin"

    def __init__(self, disease: str = "dengue", n_ensemble: int = 5, hidden: int = 48,
                 seq_len: int = SEQ_LEN, epochs: int = 40, lr: float = 1e-3,
                 min_origin: str = "2014-01-01", quantile_levels: list = QUANTILE_LEVELS,
                 seed: int = 0, calibrate: bool = False, calib_weeks: int = 78):
        self.disease = disease
        self.n_ensemble = n_ensemble
        self.hidden = hidden
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.min_origin = pd.Timestamp(min_origin)
        self.quantile_levels = quantile_levels
        self.seed = seed
        self.calibrate = calibrate
        self.calib_weeks = calib_weeks
        self._models = []
        self._fold = None
        # per-interval additive widening in log1p-count space (CQR); 0 = no adjustment
        self._cqr_logadjust = {level: 0.0 for level in INTERVAL_LEVELS}

    # -- assembly shared by fit/predict --
    def _assemble(self, fold, disease):
        weekly = _state_weekly_frame(fold, disease)
        self._ufs = MANDATORY_UFS
        self._uf_to_idx = {uf: i for i, uf in enumerate(self._ufs)}
        static = state_static_features().set_index("uf")
        static_cols = [c for c in static.columns]
        self._static_cols = static_cols

        # per-state arrays
        series = {}
        for uf, g in weekly[weekly["uf"].isin(self._ufs)].groupby("uf"):
            g = g.sort_values("date").reset_index(drop=True)
            series[uf] = g
        return weekly, series, static

    def _standardizer(self, series):
        allrows = pd.concat(series.values(), ignore_index=True)
        mean = allrows[_SEQ_NUMERIC].mean()
        std = allrows[_SEQ_NUMERIC].std().replace(0, 1.0)
        return mean, std

    def fit(self, train_df, fold, covariates=None):
        self._fold = fold
        weekly, series, static = self._assemble(fold, self.disease)
        self._mean, self._std = self._standardizer(series)
        self._static = static
        L = self.seq_len
        n_h = len(HORIZONS)

        # Build training tensors: for each origin, a sequence + per-horizon labels/mask.
        # Training targets are always inside the series, so their epiweeks come from the
        # precomputed epiweek array (no per-origin Week.fromdate - the old hot loop).
        seqs, state_idx, statics, hz_feats, labels, masks, pops, origin_dates = [], [], [], [], [], [], [], []
        for uf, g in series.items():
            vals = g.copy()
            vals[_SEQ_NUMERIC] = (vals[_SEQ_NUMERIC] - self._mean) / self._std
            harm = _epiweek_harmonics(g["epiweek"].to_numpy())
            feat = np.column_stack([vals[_SEQ_NUMERIC].to_numpy(), harm[:, :2]])
            counts = g["casos"].to_numpy().astype(float)
            dates = g["date"].to_numpy()
            pop_arr = g["population"].to_numpy()
            ew_arr = g["epiweek"].to_numpy()
            n = len(g)
            static_vec = static.loc[uf, self._static_cols].to_numpy(dtype=float)
            for t in range(L, n):
                if pd.Timestamp(dates[t]) < self.min_origin:
                    continue
                target_idx = t + HORIZONS
                valid = target_idx < n
                tew = np.ones(n_h, dtype=float)
                tew[valid] = ew_arr[target_idx[valid]]
                hz = np.column_stack([HORIZONS / 52.0, _epiweek_harmonics(tew)])
                lab = np.zeros(n_h)
                lab[valid] = counts[target_idx[valid]]
                pv = np.full(n_h, pop_arr[t])
                pv[valid] = pop_arr[target_idx[valid]]

                seqs.append(feat[t - L:t]); state_idx.append(self._uf_to_idx[uf])
                statics.append(static_vec); hz_feats.append(hz)
                labels.append(lab); masks.append(valid.astype(float)); pops.append(pv)
                origin_dates.append(dates[t])

        dev = _device()
        seqs = torch.tensor(np.array(seqs), dtype=torch.float32, device=dev)
        state_idx = torch.tensor(np.array(state_idx), dtype=torch.long, device=dev)
        statics_t = torch.tensor(np.array(statics), dtype=torch.float32, device=dev)
        hz_feats = torch.tensor(np.array(hz_feats), dtype=torch.float32, device=dev)
        labels_t = torch.tensor(np.array(labels), dtype=torch.float32, device=dev)
        masks_t = torch.tensor(np.array(masks), dtype=torch.float32, device=dev)
        pops_t = torch.tensor(np.array(pops), dtype=torch.float32, device=dev)
        origin_dates = np.array(origin_dates)

        # contiguous time-block split for CQR calibration (no random split - autocorrelated panel)
        if self.calibrate:
            split = np.datetime64(self._fold.train_cutoff - pd.Timedelta(weeks=self.calib_weeks))
            proper = torch.tensor(origin_dates < split, device=dev)
            calib = ~proper
        else:
            proper = torch.ones(len(seqs), dtype=torch.bool, device=dev)
            calib = torch.zeros(len(seqs), dtype=torch.bool, device=dev)

        n_seq_feat = seqs.shape[2]
        n_static = statics_t.shape[1]
        tensors = (seqs, state_idx, statics_t, hz_feats, labels_t, masks_t, pops_t)
        for m in range(self.n_ensemble):
            torch.manual_seed(self.seed + m)
            model = GRUForecaster(n_seq_feat, n_static, len(self._ufs), hidden=self.hidden).to(dev)
            self._train_one(model, *(t[proper] for t in tensors))
            self._models.append(model)

        if self.calibrate and int(calib.sum()) > 200:
            self._compute_cqr(*(t[calib] for t in tensors))
        return self

    def _compute_cqr(self, seqs, state_idx, statics_t, hz_feats, labels_t, masks_t, pops_t):
        """Per-interval additive widening (log1p-count space) from the calibration holdout."""
        offset = (torch.log(pops_t / INCIDENCE_SCALE + 1e-8)).cpu().numpy()
        mask = masks_t.cpu().numpy().astype(bool)
        y = labels_t.cpu().numpy()[mask]
        # pooled mu/alpha across ensemble members on the calibration set
        mus, alphas = [], []
        for model in self._models:
            model.eval()
            with torch.no_grad():
                f, log_alpha = model(seqs, state_idx, statics_t, hz_feats)
            mus.append(np.exp(offset + f.cpu().numpy())[mask])
            alphas.append(np.clip(np.exp(log_alpha.cpu().numpy()), 1e-4, 1e4)[mask])
        logy = np.log1p(y)
        n = len(y)
        for level, (lo_i, hi_i) in _INTERVAL_BOUND_IDX.items():
            lo_tau, hi_tau = self.quantile_levels[lo_i], self.quantile_levels[hi_i]
            qlo = np.mean([nbinom.ppf(lo_tau, 1.0 / a, (1.0 / a) / (1.0 / a + m)) for m, a in zip(mus, alphas)], axis=0)
            qhi = np.mean([nbinom.ppf(hi_tau, 1.0 / a, (1.0 / a) / (1.0 / a + m)) for m, a in zip(mus, alphas)], axis=0)
            scores = np.maximum(np.log1p(qlo) - logy, logy - np.log1p(qhi))
            k = int(np.ceil((n + 1) * (level / 100)))
            self._cqr_logadjust[level] = float(np.sort(scores)[min(k, n) - 1])

    def _train_one(self, model, seqs, state_idx, statics_t, hz_feats, labels_t, masks_t, pops_t):
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        n = seqs.shape[0]
        batch = 128
        offset = torch.log(pops_t / INCIDENCE_SCALE + 1e-8)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=seqs.device)
            model.train()
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                f, log_alpha = model(seqs[idx], state_idx[idx], statics_t[idx], hz_feats[idx])
                mu = torch.exp(offset[idx] + f).clamp(min=1e-6, max=1e9)
                nll = _negbin_nll(labels_t[idx], mu, log_alpha)
                loss = (nll * masks_t[idx]).sum() / masks_t[idx].sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

    def predict(self, target_grid, quantile_levels=None):
        quantile_levels = quantile_levels or self.quantile_levels
        fold = self._fold
        weekly, series, static = self._assemble(fold, self.disease)
        L = self.seq_len
        dev = _device()

        # one sequence per state (ending at cutoff), decode the target grid's dates
        rows = []
        grid = target_grid.copy()
        grid["date"] = pd.to_datetime(grid["date"])
        for uf, gg in grid.groupby("uf"):
            g = series[uf]
            vals = g.copy()
            vals[_SEQ_NUMERIC] = (vals[_SEQ_NUMERIC] - self._mean) / self._std
            harm = _epiweek_harmonics(g["epiweek"].to_numpy())
            feat = np.column_stack([vals[_SEQ_NUMERIC].to_numpy(), harm[:, :2]])
            seq = torch.tensor(feat[-L:][None, :, :], dtype=torch.float32, device=dev)
            sidx = torch.tensor([self._uf_to_idx[uf]], dtype=torch.long, device=dev)
            svec = torch.tensor(static.loc[uf, self._static_cols].to_numpy(dtype=float)[None, :],
                                dtype=torch.float32, device=dev)

            tdates = gg["date"].to_numpy()
            tew = np.array([Week.fromdate(pd.Timestamp(d)).week for d in tdates])
            horizons = ((pd.to_datetime(tdates) - fold.train_cutoff).days // 7).to_numpy()
            hz = np.column_stack([horizons / 52.0, _epiweek_harmonics(tew)])
            hz_t = torch.tensor(hz[None, :, :], dtype=torch.float32, device=dev)

            pop_target = self._target_population(uf, tdates)
            offset = np.log(pop_target / INCIDENCE_SCALE + 1e-8)

            member_quantiles = []
            for model in self._models:
                model.eval()
                with torch.no_grad():
                    f, log_alpha = model(seq, sidx, svec, hz_t)
                f = f.cpu().numpy()[0]
                log_alpha = log_alpha.cpu().numpy()[0]
                mu = np.exp(offset + f)
                alpha = np.clip(np.exp(log_alpha), 1e-4, 1e4)
                r = 1.0 / alpha
                p = r / (r + mu)
                q = np.array([nbinom.ppf(tau, r, p) for tau in quantile_levels])  # (Q, H)
                member_quantiles.append(q)
            pooled = np.mean(member_quantiles, axis=0)  # (Q, H), quantile averaging
            pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0)
            # CQR widening: widen each interval's bounds in log1p-count space, then re-sort
            for level, (lo_i, hi_i) in _INTERVAL_BOUND_IDX.items():
                q = self._cqr_logadjust.get(level, 0.0)
                if q != 0.0:
                    pooled[lo_i] = np.expm1(np.maximum(0.0, np.log1p(pooled[lo_i]) - q))
                    pooled[hi_i] = np.expm1(np.log1p(pooled[hi_i]) + q)
            pooled = np.sort(pooled, axis=0)  # monotonicity across quantiles
            pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0)

            for qi, tau in enumerate(quantile_levels):
                for hj, d in enumerate(tdates):
                    rows.append({"uf": uf, "date": pd.Timestamp(d), "quantile_level": tau,
                                 "predicted_value": max(0.0, pooled[qi, hj])})
        return pd.DataFrame(rows)

    def _target_population(self, uf, tdates) -> np.ndarray:
        pop = state_population()
        ymin, ymax = pop["year"].min(), pop["year"].max()
        years = pd.to_datetime(tdates).year.to_numpy().clip(ymin, ymax)
        pmap = pop[pop["uf"] == uf].set_index("year")["population"]
        return np.array([pmap.get(int(y), pmap.iloc[-1]) for y in years], dtype=float)
