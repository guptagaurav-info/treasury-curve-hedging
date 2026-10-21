"""Hedge-ratio rules and the walk-forward engine.

A hedge rule maps a training window (data strictly before the rebalance date) to hedge
weights w_j: the DV01 held in hedge instrument j as a fraction of the book's DV01
(negative = short). Weights are fixed for the month; notionals are re-sized daily from the
previous close's DV01, so the hedge always uses information available at the prior close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGIES = {
    "unhedged": "Unhedged",
    "dv01_10y": "Duration-matched, 10-year only",
    "beta_10y": "Yield-beta 10-year (regression on yield changes)",
    "pca2": "Two-factor curve hedge (2y + 10y)",
    "pca3": "Three-factor curve hedge (2y + 5y + 10y)",
    "minvar": "Minimum-variance regression (2y + 5y + 10y)",
}


def pca_loadings(dy: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-k principal components of daily yield changes (bp). Sign: level/slope positive at long end."""
    x = dy.to_numpy() * 1e4
    x = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:k]
    L = vecs[:, order]
    L = L * np.sign(L[-1, :])[None, :]   # long-end loading positive, for readability only
    return pd.DataFrame(L, index=dy.columns, columns=[f"pc{i+1}" for i in range(k)])


def fit_weights(rule: str, dy: pd.DataFrame, xr: pd.DataFrame, dv01: pd.DataFrame,
                book: float, hedges: list) -> dict:
    """Return {tenor: weight} for one rule from a training window (all inputs strictly past)."""
    w = {h: 0.0 for h in hedges}
    if rule == "unhedged":
        return w
    if rule == "dv01_10y":
        w[10] = -1.0
        return w
    if rule == "beta_10y":
        a, b = dy[book].to_numpy(), dy[10].to_numpy()
        w[10] = -float(np.cov(a, b)[0, 1] / np.var(b, ddof=1))
        return w
    if rule in ("pca2", "pca3"):
        inst = [2, 10] if rule == "pca2" else [2, 5, 10]
        L = pca_loadings(dy, len(inst))
        A = L.loc[inst].to_numpy().T            # factors x instruments
        bvec = -L.loc[book].to_numpy()          # neutralise the book's factor exposure
        sol = np.linalg.solve(A, bvec)
        for t, s in zip(inst, sol):
            w[t] = float(s)
        return w
    if rule == "minvar":
        # OLS of the book's excess return on hedge excess returns -> notional ratios -> DV01 weights
        X = xr[hedges].to_numpy()
        yv = xr[book].to_numpy()
        X1 = np.c_[np.ones(len(X)), X]
        beta = np.linalg.lstsq(X1, yv, rcond=None)[0][1:]
        d_book = dv01[book].mean()
        for t, bt in zip(hedges, beta):
            w[t] = float(-bt * dv01[t].mean() / d_book)
        return w
    raise ValueError(rule)


def rebalance_dates(idx: pd.DatetimeIndex, start: str) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    s = s[s >= pd.Timestamp(start)]
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).min().values)


def walk_forward(curve: pd.DataFrame, xr: pd.DataFrame, dv01: pd.DataFrame, cfg: dict,
                 rules=None, lookahead: bool = False):
    """Run every rule walk-forward. Returns (daily $ P&L frame, weights log frame).

    lookahead=True fits each rule once on the full test sample: a deliberately leaky
    benchmark used only to show how much in-sample fitting flatters a hedge.
    """
    rules = rules or list(STRATEGIES)
    book = cfg["book"]["tenor"]
    hedges = cfg["hedge_tenors"]
    pts = cfg["pca_tenors"]
    mv = cfg["book"]["market_value_usd"]
    wf = cfg["walk_forward"]

    dy_all = curve[pts].diff()
    valid = xr[[book] + hedges].notna().all(axis=1) & dy_all.notna().all(axis=1)
    days = curve.index[valid]
    rdates = rebalance_dates(days, wf["test_start"])

    book_dv01_prev = (dv01[book].shift(1) * mv)
    pnl = {r: pd.Series(np.nan, index=curve.index) for r in rules}
    cost = {r: pd.Series(0.0, index=curve.index) for r in rules}
    logs = []
    prev_w = {r: {h: 0.0 for h in hedges} for r in rules}

    if lookahead:
        span = days[days >= pd.Timestamp(wf["test_start"])]
        tr = span
        fixed = {r: fit_weights(r, dy_all.loc[tr], xr.loc[tr], dv01.loc[tr], book, hedges) for r in rules}

    for i, d0 in enumerate(rdates):
        d1 = rdates[i + 1] if i + 1 < len(rdates) else curve.index[-1] + pd.Timedelta(days=1)
        train = days[days < d0][-wf["train_days"]:]
        train = train[train >= d0 - pd.DateOffset(years=wf.get("max_train_years", 4))]
        if len(train) < wf["min_train_days"]:
            continue
        assert train.max() < d0, "training data must end before the rebalance date"
        hold = days[(days >= d0) & (days < d1)]
        for r in rules:
            w = fixed[r] if lookahead else fit_weights(r, dy_all.loc[train], xr.loc[train],
                                                        dv01.loc[train], book, hedges)
            bd = book_dv01_prev.loc[hold]
            leg = mv * xr.loc[hold, book]
            for h in hedges:
                notional = w[h] * bd / dv01[h].shift(1).loc[hold]
                leg = leg + notional * xr.loc[hold, h]
            pnl[r].loc[hold] = leg
            turn = sum(abs(w[h] - prev_w[r][h]) for h in hedges)
            if len(hold):
                cost[r].loc[hold[0]] = turn * bd.iloc[0]
            prev_w[r] = w
            logs.append({"date": d0, "rule": r, "train_start": train.min(), "train_end": train.max(),
                         **{f"w{h}": w[h] for h in hedges}})
    P = pd.DataFrame(pnl).dropna(how="all")
    C = pd.DataFrame(cost).loc[P.index]   # $ DV01 traded; multiply by cost in bp for $ cost
    return P, C, pd.DataFrame(logs)
