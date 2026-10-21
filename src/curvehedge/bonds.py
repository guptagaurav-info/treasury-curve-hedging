"""Constant-maturity par-bond arithmetic.

Each curve point is treated as a par bond bought at the previous close (coupon = that day's
par yield, semiannual) and repriced at today's par yield for the same tenor. This is the
standard way to turn a constant-maturity yield series into a daily total return; roll-down
over one day is ignored (it is a fraction of a basis point). Every leg is financed at the
3-month bill rate, so returns are excess returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price(y: np.ndarray, coupon: np.ndarray, maturity: float, dt: np.ndarray) -> np.ndarray:
    """Dirty price per 100 face of a bond with `maturity` years left at issue, `dt` years later."""
    n = int(round(maturity * 2))
    k = np.arange(1, n + 1) / 2.0                       # cash-flow times from issue (years)
    tau = k[None, :] - dt[:, None]                      # time remaining at the new date
    cf = np.repeat((coupon / 2 * 100)[:, None], n, axis=1)
    cf[:, -1] += 100.0
    disc = (1 + y[:, None] / 2) ** (-2 * tau)
    return (cf * disc).sum(axis=1)


def dv01_par(y: np.ndarray, maturity: float) -> np.ndarray:
    """Dollar value of one basis point per $1 of market value of a par bond at yield y."""
    y = np.asarray(y, dtype=float)
    zero = np.zeros_like(y)
    up = price(y + 1e-4, y, maturity, zero)
    dn = price(y - 1e-4, y, maturity, zero)
    return (dn - up) / 2 / 100.0


def excess_returns(curve: pd.DataFrame, tenors, financing_tenor: float = 0.25) -> pd.DataFrame:
    """Daily excess return per $1 for each tenor; NaN where either day's yield is missing."""
    dates = curve.index
    dt = np.r_[np.nan, np.diff(dates.values).astype("timedelta64[D]").astype(float) / 365.0]
    bill_prev = curve[financing_tenor].shift(1).to_numpy()
    out = {}
    for m in tenors:
        y_prev = curve[m].shift(1).to_numpy()
        y_now = curve[m].to_numpy()
        ok = ~(np.isnan(y_prev) | np.isnan(y_now) | np.isnan(dt))
        r = np.full(len(curve), np.nan)
        p = price(y_now[ok], y_prev[ok], float(m), dt[ok])
        r[ok] = p / 100.0 - 1.0 - bill_prev[ok] * dt[ok]
        out[m] = r
    return pd.DataFrame(out, index=dates)
