"""End to end: pinned data -> par-bond excess returns -> walk-forward hedges -> metrics -> figures.

Run with `make all` (PYTHONPATH=src python -m curvehedge.pipeline).
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from . import bonds, config, hedges, ingest, metrics, report


def main(cfg_path=None) -> dict:
    cfg = config.load(cfg_path)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out = cfg["out_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=True)

    curve = ingest.load_curve(cfg)
    tenors = sorted(set(cfg["pca_tenors"]) | set(cfg["hedge_tenors"]) | {cfg["book"]["tenor"]})
    xr = bonds.excess_returns(curve, tenors, cfg["financing_tenor"])
    dv01 = pd.DataFrame({m: bonds.dv01_par(curve[m].to_numpy(), m) for m in tenors}, index=curve.index)

    # Data quality, recorded not hidden
    dq = pd.DataFrame({
        "first_date": [curve[c].first_valid_index() for c in curve.columns],
        "last_date": [curve[c].last_valid_index() for c in curve.columns],
        "missing_share": [float(curve[c].isna().mean()) for c in curve.columns],
    }, index=[f"{c}y" for c in curve.columns])
    dq.to_csv(out / "data_quality.csv")
    gap = curve[30].isna()
    gap_days = curve.index[gap & (curve.index > pd.Timestamp("2000-01-01")) & (curve.index < pd.Timestamp("2007-01-01"))]

    P, C, W = hedges.walk_forward(curve, xr, dv01, cfg)
    Pl, _, _ = hedges.walk_forward(curve, xr, dv01, cfg, lookahead=True)
    W.to_csv(out / "hedge_weights_log.csv", index=False)

    base = "unhedged"
    S = metrics.summary(P, base)
    Sl = metrics.summary(Pl.loc[P.index], base)
    split = pd.Timestamp(cfg["walk_forward"]["split_date"])
    S1 = metrics.summary(P[P.index < split], base)
    S2 = metrics.summary(P[P.index >= split], base)

    years = P.index.year
    yearly = P.groupby(years).std() * np.sqrt(252)

    # Costs (illustrative): $DV01 traded x cost in bp
    yrs = (P.index[-1] - P.index[0]).days / 365.25
    cost_bp = cfg["costs"]["bp_of_dv01_traded"]
    ann_cost = (C.sum() * cost_bp / yrs).to_dict()
    gross = W.assign(g=W[[c for c in W.columns if c.startswith("w")]].abs().sum(axis=1)).groupby("rule")["g"].mean()
    turnover = W.sort_values("date").groupby("rule")[[c for c in W.columns if c.startswith("w")]].apply(
        lambda d: d.diff().abs().sum(axis=1).mean())

    # What the duration hedge's residual actually is: a 10s30s curve position
    dslope = (curve[30] - curve[10]).diff().loc[P.index] * 1e4
    dlevel = curve[10].diff().loc[P.index] * 1e4
    res = P["dv01_10y"]
    ok = dslope.notna() & res.notna()
    X = np.c_[np.ones(ok.sum()), dslope[ok]]
    b = np.linalg.lstsq(X, res[ok].to_numpy(), rcond=None)[0]
    fit = X @ b
    r2_slope = 1 - ((res[ok] - fit) ** 2).sum() / ((res[ok] - res[ok].mean()) ** 2).sum()
    corr_level = float(np.corrcoef(res[ok], dlevel[ok])[0, 1])

    # Yield beta of 30s to 10s by year (why DV01-matching over-hedges)
    dy = curve[[10, 30]].diff().dropna()
    beta_by_year = dy.groupby(dy.index.year).apply(lambda d: np.cov(d[30], d[10])[0, 1] / np.var(d[10], ddof=1))

    # Stress windows
    stress = []
    for name, (a, z) in cfg["stress_windows"].items():
        m = (P.index >= pd.Timestamp(a)) & (P.index <= pd.Timestamp(z))
        row = {"window": name, "days": int(m.sum())}
        for r in ["unhedged", "dv01_10y", "beta_10y", "minvar", "pca3"]:
            row[f"{r}_cum_pnl_usd"] = float(P.loc[m, r].sum())
            row[f"{r}_worst_day_usd"] = float(-P.loc[m, r].min())
            row[f"{r}_vol_usd"] = float(P.loc[m, r].std() * np.sqrt(252))
        stress.append(row)
    stress = pd.DataFrame(stress)
    stress.to_csv(out / "stress_windows.csv", index=False)

    # Statistical confidence on the headline comparisons
    ci = {f"{a}_vs_dv01_10y": metrics.block_bootstrap_ratio(P[a], P["dv01_10y"], seed=cfg["seed"])
          for a in ["beta_10y", "pca2", "pca3", "minvar"]}

    # Business translation (illustrative economics, labelled)
    mv = cfg["book"]["market_value_usd"]
    k = cfg["economics"]["capital_multiplier"]
    coc = cfg["economics"]["cost_of_capital"]
    def capital(rule):
        return k * np.sqrt(10) * S.loc[rule, "var99_1d_usd"]
    biz = {r: {"capital_usd": float(capital(r)), "annual_capital_cost_usd": float(capital(r) * coc)}
           for r in S.index}
    latest = W[W["date"] == W["date"].max()].set_index("rule")

    summary_tbl = S.join(Sl[["variance_reduction"]].rename(columns={"variance_reduction": "vr_lookahead"}))
    summary_tbl["gross_hedge_dv01_x_book"] = gross
    summary_tbl["ann_cost_usd_illustrative"] = pd.Series(ann_cost)
    summary_tbl.to_csv(out / "hedge_summary.csv")
    yearly.to_csv(out / "residual_vol_by_year.csv")

    m = {
        "data": {
            "source": cfg["source"]["url"], "sha256": cfg["source"]["sha256"],
            "revision": cfg["source"]["revision"], "downloaded": cfg["source"]["downloaded"],
            "first_date": str(curve.index.min().date()), "last_date": str(curve.index.max().date()),
            "n_curve_days": int(len(curve)), "n_test_days": int(len(P)),
            "test_start": str(P.index.min().date()), "test_end": str(P.index.max().date()),
            "thirty_year_gap_days": int(len(gap_days)),
            "thirty_year_gap": [str(gap_days.min().date()), str(gap_days.max().date())] if len(gap_days) else None,
            "latest_curve_pct": {f"{c}y": round(float(curve[c].iloc[-1]) * 100, 2) for c in [0.25, 2, 5, 10, 30]},
            "book_dv01_usd_latest": float(dv01[30].iloc[-1] * mv),
        },
        "summary": json.loads(summary_tbl.to_json(orient="index")),
        "first_half": json.loads(S1.to_json(orient="index")),
        "second_half": json.loads(S2.to_json(orient="index")),
        "bootstrap_vol_ratio_ci95": ci,
        "residual_of_dv01_hedge": {"r2_on_10s30s_change": float(r2_slope),
                                    "usd_per_bp_10s30s": float(b[1]),
                                    "corr_with_10y_level_change": corr_level},
        "yield_beta_30_on_10": {"full": float(np.cov(dy[30], dy[10])[0, 1] / np.var(dy[10], ddof=1)),
                                "min_year": float(beta_by_year.min()), "max_year": float(beta_by_year.max()),
                                "min_year_at": int(beta_by_year.idxmin()), "max_year_at": int(beta_by_year.idxmax())},
        "turnover_weight_change_per_month": json.loads(turnover.to_json()),
        "business_illustrative": biz,
        "latest_weights": json.loads(latest[[c for c in latest.columns if c.startswith("w")]].to_json(orient="index")),
        "latest_weights_date": str(W["date"].max().date()),
        "stress": json.loads(stress.to_json(orient="records")),
    }
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    report.figures(curve, P, yearly, W, dslope, beta_by_year, stress, S, out / "figures", cfg)
    print(summary_tbl.round(3).to_string())
    return m


if __name__ == "__main__":
    main()
