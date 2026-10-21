"""Risk metrics on daily $ P&L series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ann_vol(x: pd.Series) -> float:
    return float(x.std(ddof=1) * np.sqrt(252))


def var(x: pd.Series, q: float = 0.99) -> float:
    """Historical one-day VaR at confidence q, reported as a positive loss."""
    return float(-np.quantile(x.dropna(), 1 - q))


def es(x: pd.Series, q: float = 0.975) -> float:
    """Historical expected shortfall at confidence q, positive loss."""
    x = x.dropna()
    cut = np.quantile(x, 1 - q)
    return float(-x[x <= cut].mean())


def worst_window(x: pd.Series, n: int) -> float:
    return float(-x.rolling(n).sum().min())


def variance_reduction(res: pd.Series, base: pd.Series) -> float:
    return float(1 - res.var() / base.var())


def block_bootstrap_ratio(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 7):
    """95% CI of vol(a)/vol(b) resampling whole calendar months (keeps volatility clustering)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    groups = [g for _, g in df.groupby([df.index.year, df.index.month])]
    sa = np.array([(g["a"] ** 2).sum() for g in groups])
    sb = np.array([(g["b"] ** 2).sum() for g in groups])
    ma = np.array([g["a"].sum() for g in groups])
    mb = np.array([g["b"].sum() for g in groups])
    nn = np.array([len(g) for g in groups])
    out = []
    for _ in range(n_boot):
        k = rng.integers(0, len(groups), len(groups))
        N = nn[k].sum()
        va = sa[k].sum() / N - (ma[k].sum() / N) ** 2
        vb = sb[k].sum() / N - (mb[k].sum() / N) ** 2
        out.append(np.sqrt(va / vb))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def summary(P: pd.DataFrame, base: str = "unhedged") -> pd.DataFrame:
    rows = []
    for c in P.columns:
        x = P[c].dropna()
        rows.append({
            "rule": c,
            "ann_vol_usd": ann_vol(x),
            "var99_1d_usd": var(x, 0.99),
            "es975_1d_usd": es(x, 0.975),
            "worst_day_usd": float(-x.min()),
            "worst_10d_usd": worst_window(x, 10),
            "variance_reduction": variance_reduction(x, P[base].loc[x.index]),
        })
    return pd.DataFrame(rows).set_index("rule")
