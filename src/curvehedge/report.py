"""Charts: plain-English titles, units on axes, integer year axes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .hedges import STRATEGIES

NAVY, GOLD, RED, GREY, GREEN, TEAL = "#1F3A5F", "#9A7F2E", "#B03A2E", "#8A8A85", "#2E7D4F", "#3A6A9A"
COL = {"unhedged": GREY, "dv01_10y": RED, "beta_10y": GOLD, "pca2": TEAL, "pca3": "#7B5EA7", "minvar": NAVY}
SHORT = {"unhedged": "Unhedged", "dv01_10y": "Duration-matched (10y)", "beta_10y": "Yield-beta (10y)",
         "pca2": "2-factor (2y+10y)", "pca3": "3-factor (2y+5y+10y)", "minvar": "Min-variance (2y+5y+10y)"}

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False, "text.usetex": False})
plt.rcParams["axes.formatter.use_mathtext"] = False


def _years(ax, step=4):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _musd(ax, axis="y"):
    f = FuncFormatter(lambda v, _: ("\u2212" if v < 0 else "") + f"\\${abs(v)/1e6:,.0f}M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def figures(curve, P, yearly, W, dslope, beta_by_year, stress, S, fig_dir, cfg):
    # 1. The curve
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for t, c, lab in [(2, TEAL, "2-year"), (10, NAVY, "10-year"), (30, GOLD, "30-year")]:
        ax[0].plot(curve.index, curve[t] * 100, color=c, lw=1.1, label=lab)
    ax[0].set_ylabel("Par yield (%)")
    ax[0].set_title("US Treasury par yields, 1990 to October 2026")
    ax[0].legend(loc="upper right", ncol=3)
    sl = (curve[30] - curve[10]) * 1e4
    ax[1].plot(curve.index, sl, color=RED, lw=1)
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set_ylabel("30y minus 10y (bp)")
    ax[1].set_title("The 10s30s curve: what a 10-year hedge on a 30-year book leaves open", fontsize=11)
    _years(ax[1])
    fig.text(0.01, -0.01, "Source: US Treasury daily par yield curve. 30-year series suspended Feb 2002 to Feb 2006.",
             fontsize=8, color=GREY)
    fig.savefig(fig_dir / "curve_history.png"); plt.close(fig)

    # 2. Residual risk by hedge (out of sample)
    order = ["unhedged", "dv01_10y", "pca3", "pca2", "beta_10y", "minvar"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    vals = [S.loc[r, "ann_vol_usd"] for r in order]
    bars = ax.barh([SHORT[r] for r in order], vals, color=[COL[r] for r in order])
    for bar, v, r in zip(bars, vals, order):
        vr = S.loc[r, "variance_reduction"]
        lab = f"\\${v/1e6:,.0f}M" + ("" if r == "unhedged" else f"  ({vr:.0%} of variance removed)")
        ax.text(v + 2e6, bar.get_y() + bar.get_height() / 2, lab, va="center", fontsize=9.5)
    ax.invert_yaxis()
    _musd(ax, "x")
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_xlabel("Annualised volatility of daily P&L, \\$ (per \\$1bn 30-year book)")
    ax.set_title("Risk left after hedging, walk-forward test 1994 to 2026")
    fig.savefig(fig_dir / "residual_risk_by_hedge.png"); plt.close(fig)

    # 3. Residual vol by year
    fig, ax = plt.subplots(figsize=(10, 4.4))
    yearly = yearly.reindex(range(int(yearly.index.min()), int(yearly.index.max()) + 1))
    yearly.loc[2002:2006] = np.nan
    yrs = yearly.index.astype(int)
    for r in ["dv01_10y", "beta_10y", "minvar", "pca3"]:
        ax.plot(yrs, yearly[r], marker="o", ms=3, lw=1.3, color=COL[r], label=SHORT[r])
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    _musd(ax)
    ax.set_ylabel("Residual volatility, \\$ per year")
    ax.set_xlabel("Calendar year")
    ax.set_title("Hedged risk by year: the regression hedge is lower in every year tested")
    ax.axvspan(2001.5, 2006.5, color=GREY, alpha=0.12)
    ax.text(2004, ax.get_ylim()[1] * 0.92, "no 30-year\nseries", ha="center", fontsize=8, color=GREY)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(fig_dir / "residual_vol_by_year.png"); plt.close(fig)

    # 4. Residual of duration hedge vs 10s30s move
    fig, ax = plt.subplots(figsize=(7.2, 5))
    x = dslope.loc[P.index]
    y = P["dv01_10y"]
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=4, alpha=0.25, color=RED)
    bfit = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(-25, 25, 10)
    ax.plot(xs, np.polyval(bfit, xs), color=NAVY, lw=1.6, label=f"≈ \\${-bfit[0]/1e6:,.2f}M lost per 1bp of 10s30s steepening")
    ax.set_xlim(-25, 25)
    _musd(ax)
    ax.set_xlabel("Daily change in 30-year minus 10-year yield (bp)")
    ax.set_ylabel("Daily P&L after a duration-matched hedge, \\$")
    ax.set_title("A duration-matched hedge is a curve position")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(fig_dir / "dv01_residual_vs_curve.png"); plt.close(fig)

    # 5. Yield beta by year
    fig, ax = plt.subplots(figsize=(10, 3.8))
    yb = beta_by_year
    ax.bar(yb.index.astype(int), yb.values, color=GOLD)
    ax.axhline(1.0, color=RED, lw=1.2, ls="--", label="Duration hedge assumes 1.00")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_ylabel("bp move in 30y per 1bp in 10y")
    ax.set_xlabel("Calendar year")
    ax.set_ylim(0, 1.2)
    ax.set_title("Long yields move less than 10-year yields, so a duration hedge over-hedges")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_ylim(0, 1.35)
    fig.savefig(fig_dir / "yield_beta_by_year.png"); plt.close(fig)

    # 6. Hedge weights through time
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for r, ls in [("beta_10y", "-"), ("minvar", "-"), ("pca3", ":")]:
        d = W[W["rule"] == r].set_index("date")
        net = d[[c for c in d.columns if c.startswith("w")]].sum(axis=1)
        brk = net.index.to_series().diff().dt.days > 45          # break the line across the 30y gap
        net = pd.concat([net, pd.Series(np.nan, index=net.index[brk] - pd.Timedelta(days=1))]).sort_index()
        ax.plot(net.index, -net, ls=ls, color=COL[r], lw=1.4, label=f"{SHORT[r]}: net hedge")
    ax.axhline(1.0, color=RED, lw=1.2, ls="--", label="Duration-matched (1.00)")
    ax.set_ylabel("Hedge DV01 as a multiple of book DV01")
    ax.set_title("How much rate risk each rule actually hedges, refit monthly")
    _years(ax)
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    fig.savefig(fig_dir / "hedge_ratio_history.png"); plt.close(fig)

    # 7. Stress windows
    fig, ax = plt.subplots(figsize=(10, 4.4))
    rules = ["dv01_10y", "beta_10y", "minvar"]
    xi = np.arange(len(stress))
    wdt = 0.26
    for j, r in enumerate(rules):
        ax.bar(xi + (j - 1) * wdt, stress[f"{r}_worst_day_usd"], wdt, color=COL[r], label=SHORT[r])
    ax.set_xticks(xi)
    ax.set_xticklabels([w.replace(" (", "\n(") for w in stress["window"]], fontsize=9)
    _musd(ax)
    ax.set_ylabel("Worst single-day loss, \\$")
    ax.set_title("Worst hedged day in each stress window, per \\$1bn 30-year book")
    ax.legend(fontsize=9)
    fig.savefig(fig_dir / "stress_worst_day.png"); plt.close(fig)
