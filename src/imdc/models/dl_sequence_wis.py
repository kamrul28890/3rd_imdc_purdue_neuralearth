"""GRU sequence forecaster trained directly against a pinball/WIS-consistent loss, instead of a
Negative-Binomial NLL (dl_sequence.py). Motivated by "Dengue Oracle" (FGV EMAp)'s LSTM, reviewed
for training directly against WIS rather than a generic likelihood (see docs/FUTURE_WORK.md Sec
6c) - reimplemented independently as a pinball-loss GRU, not their code.

Architecture is otherwise identical to dl_sequence.py (shared encoder, per-state embedding, static
covariates, per-horizon decoder head) - only the head's output and the loss function differ:
- dl_sequence.py: head outputs 2 values (log-rate, log-dispersion) per horizon; trained with NB2
  negative log-likelihood; quantiles are analytic (scipy.stats.nbinom.ppf) after training.
- this file: head outputs one value per quantile level (9) per horizon directly, in log1p(incidence)
  space (same target representation as ml_boosted.py/xgb_quantile.py); trained with the pinball
  (quantile/check) loss summed over all 9 levels with weights matching this project's own WIS
  definition (imdc.evaluation.metrics), so the training objective is the actual scoring metric
  rather than a proxy for it. No CQR calibration - a model trained directly on the quantile loss
  has no obvious analog of a calibration holdout the way a likelihood-based model does, and
  intervals are already the direct optimization target rather than a post-hoc derived one.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from epiweeks import Week

from imdc.config import MANDATORY_UFS, QUANTILE_LEVELS
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_no_leakage
from imdc.features.panel import INCIDENCE_SCALE, state_climate_full, state_population, state_static_features
from imdc.models.dl_sequence import _device, _epiweek_harmonics, _SEQ_NUMERIC, SEQ_LEN, HORIZONS

# WIS weights: w_0=1/2 for the median, w_k=alpha_k/2 for each interval - see metrics.py. Applied
# per-quantile here as a symmetric pinball weighting (each of the two tails of an interval carries
# half the interval's weight, which is exactly the pinball loss's own tau/(1-tau) weighting, so no
# extra reweighting beyond standard pinball loss is needed for the outer quantiles; only the
# median needs its usual doubled relative weight, matched by pinball loss automatically since
# tau=0.5 already weights both sides equally at 0.5).
_TAU = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)


def _pinball_loss(pred, target, tau):
    """pred, target: (...,), tau: (Q,) broadcastable over the last new dim."""
    diff = target.unsqueeze(-1) - pred  # (..., Q)
    return torch.maximum(tau * diff, (tau - 1) * diff)


class GRUQuantileHead(nn.Module):
    def __init__(self, n_seq_feat: int, n_static: int, n_states: int, n_quantiles: int,
                 hidden: int = 48, emb_dim: int = 8, horizon_feat: int = 5, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_seq_feat, hidden, batch_first=True)
        self.state_emb = nn.Embedding(n_states, emb_dim)
        dec_in = hidden + emb_dim + n_static + horizon_feat
        self.head = nn.Sequential(
            nn.Linear(dec_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_quantiles),
        )

    def forward(self, seq, state_idx, static, horizon_feats):
        _, h = self.gru(seq)
        h = h[-1]
        emb = self.state_emb(state_idx)
        ctx = torch.cat([h, emb, static], dim=1)
        B, H, hf = horizon_feats.shape
        ctx_rep = ctx.unsqueeze(1).expand(-1, H, -1)
        dec_in = torch.cat([ctx_rep, horizon_feats], dim=2)
        return self.head(dec_in)  # (B, H, Q) - log1p(incidence) quantile predictions


class DLSequenceWISModel:
    """Deep-ensemble GRU forecaster trained with pinball loss, same fit/predict protocol."""

    name = "gru_wis"

    def __init__(self, disease: str = "dengue", n_ensemble: int = 5, hidden: int = 48,
                 seq_len: int = SEQ_LEN, epochs: int = 40, lr: float = 1e-3,
                 min_origin: str = "2014-01-01", quantile_levels: list = QUANTILE_LEVELS, seed: int = 0):
        self.disease = disease
        self.n_ensemble = n_ensemble
        self.hidden = hidden
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.min_origin = pd.Timestamp(min_origin)
        self.quantile_levels = list(quantile_levels)
        self.seed = seed
        self._models = []
        self._fold = None

    def _assemble(self, fold, disease):
        cases = cutoff_filter(load_cases(disease), fold.train_cutoff)
        assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} dl-wis cases")
        state = aggregate_cases_to_state(cases)
        pop = state_population()
        df = state.copy()
        df["year"] = df["date"].dt.year
        ymin, ymax = pop["year"].min(), pop["year"].max()
        df["pop_year"] = df["year"].clip(ymin, ymax)
        df = df.merge(pop.rename(columns={"year": "pop_year"}), on=["uf", "pop_year"], how="left")
        df["incidence"] = df["casos"] / df["population"] * INCIDENCE_SCALE
        df["log_inc"] = np.log1p(df["incidence"])
        climate = cutoff_filter(state_climate_full(), fold.train_cutoff)
        df = df.merge(climate[["uf", "date", "temp_med", "precip_med", "rel_humid_med"]],
                      on=["uf", "date"], how="left")
        df["epiweek"] = np.array([Week.fromdate(d).week for d in df["date"]])
        df = df.sort_values(["uf", "date"]).reset_index(drop=True)

        self._ufs = MANDATORY_UFS
        self._uf_to_idx = {uf: i for i, uf in enumerate(self._ufs)}
        static = state_static_features().set_index("uf")
        self._static_cols = list(static.columns)
        series = {uf: g.sort_values("date").reset_index(drop=True)
                  for uf, g in df[df["uf"].isin(self._ufs)].groupby("uf")}
        return series, static

    def fit(self, train_df, fold, covariates=None):
        self._fold = fold
        series, static = self._assemble(fold, self.disease)
        self._static = static
        allrows = pd.concat(series.values(), ignore_index=True)
        self._mean = allrows[_SEQ_NUMERIC].mean()
        self._std = allrows[_SEQ_NUMERIC].std().replace(0, 1.0)
        L = self.seq_len
        n_h = len(HORIZONS)

        seqs, state_idx, statics, hz_feats, labels, masks = [], [], [], [], [], []
        for uf, g in series.items():
            vals = g.copy()
            vals[_SEQ_NUMERIC] = (vals[_SEQ_NUMERIC] - self._mean) / self._std
            harm = _epiweek_harmonics(g["epiweek"].to_numpy())
            feat = np.column_stack([vals[_SEQ_NUMERIC].to_numpy(), harm[:, :2]])
            log_inc = g["log_inc"].to_numpy()
            dates = g["date"].to_numpy()
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
                lab[valid] = log_inc[target_idx[valid]]

                seqs.append(feat[t - L:t]); state_idx.append(self._uf_to_idx[uf])
                statics.append(static_vec); hz_feats.append(hz)
                labels.append(lab); masks.append(valid.astype(float))

        dev = _device()
        seqs = torch.tensor(np.array(seqs), dtype=torch.float32, device=dev)
        state_idx = torch.tensor(np.array(state_idx), dtype=torch.long, device=dev)
        statics_t = torch.tensor(np.array(statics), dtype=torch.float32, device=dev)
        hz_feats = torch.tensor(np.array(hz_feats), dtype=torch.float32, device=dev)
        labels_t = torch.tensor(np.array(labels), dtype=torch.float32, device=dev)
        masks_t = torch.tensor(np.array(masks), dtype=torch.float32, device=dev)
        tau = _TAU.to(dev)

        n_seq_feat = seqs.shape[2]
        n_static = statics_t.shape[1]
        n_q = len(self.quantile_levels)
        for m in range(self.n_ensemble):
            torch.manual_seed(self.seed + m)
            model = GRUQuantileHead(n_seq_feat, n_static, len(self._ufs), n_q, hidden=self.hidden).to(dev)
            self._train_one(model, seqs, state_idx, statics_t, hz_feats, labels_t, masks_t, tau)
            self._models.append(model)
        return self

    def _train_one(self, model, seqs, state_idx, statics_t, hz_feats, labels_t, masks_t, tau):
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        n = seqs.shape[0]
        batch = 128
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=seqs.device)
            model.train()
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                pred = model(seqs[idx], state_idx[idx], statics_t[idx], hz_feats[idx])  # (B,H,Q)
                pl = _pinball_loss(pred, labels_t[idx], tau)  # (B,H,Q)
                loss = (pl.mean(dim=-1) * masks_t[idx]).sum() / masks_t[idx].sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

    def predict(self, target_grid, quantile_levels=None):
        quantile_levels = quantile_levels or self.quantile_levels
        fold = self._fold
        series, static = self._assemble(fold, self.disease)
        L = self.seq_len
        dev = _device()

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

            member_preds = []
            for model in self._models:
                model.eval()
                with torch.no_grad():
                    pred = model(seq, sidx, svec, hz_t)  # (1,H,Q)
                member_preds.append(pred.cpu().numpy()[0])
            pooled_log = np.mean(member_preds, axis=0)  # (H, Q)
            pooled_log = np.sort(pooled_log, axis=1)  # enforce monotonicity
            counts = np.maximum(0.0, np.expm1(pooled_log) * pop_target[:, None] / INCIDENCE_SCALE)

            for hj, d in enumerate(tdates):
                for qi, tau in enumerate(quantile_levels):
                    rows.append({"uf": uf, "date": pd.Timestamp(d), "quantile_level": tau,
                                 "predicted_value": float(counts[hj, qi])})
        return pd.DataFrame(rows)

    def _target_population(self, uf, tdates) -> np.ndarray:
        pop = state_population()
        ymin, ymax = pop["year"].min(), pop["year"].max()
        years = pd.to_datetime(tdates).year.to_numpy().clip(ymin, ymax)
        pmap = pop[pop["uf"] == uf].set_index("year")["population"]
        return np.array([pmap.get(int(y), pmap.iloc[-1]) for y in years], dtype=float)
