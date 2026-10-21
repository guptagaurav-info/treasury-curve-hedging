"""Unit tests: bond arithmetic, no look-ahead in hedge fitting, factor neutrality, metrics.

Tests that need the real curve skip if data/raw has not been downloaded (`make ingest`).
"""
import numpy as np
import pandas as pd
import pytest

from curvehedge import bonds, config, hedges, ingest, metrics

CFG = config.load()
HAVE_DATA = (CFG["raw_dir"] / CFG["source"]["file"]).exists()


def test_par_bond_prices_at_par():
    y = np.array([0.05, 0.02])
    p = bonds.price(y, y, 10.0, np.zeros(2))
    assert np.allclose(p, 100.0, atol=1e-9)


def test_dv01_matches_closed_form_modified_duration():
    y = 0.05
    closed = (1 / y) * (1 - (1 + y / 2) ** (-20)) * 1e-4   # par bond modified duration x 1bp
    assert bonds.dv01_par(np.array([y]), 10.0)[0] == pytest.approx(closed, rel=1e-4)


def test_unchanged_yield_earns_carry_over_bills():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cv = pd.DataFrame({0.25: [0.04, 0.04], 10: [0.05, 0.05]}, index=idx)
    r = bonds.excess_returns(cv, [10])[10].iloc[1]
    # semiannual compounding of the coupon vs simple bill accrual: within 10%
    assert r == pytest.approx((0.05 - 0.04) / 365, rel=0.10)


def test_metrics_var_es():
    x = pd.Series(np.r_[-np.arange(1, 21), np.zeros(980)].astype(float))   # 20 losses of 1..20 in 1,000 days
    v = metrics.var(x, 0.99)
    assert 10 <= v <= 12                       # 1% tail = the 10th-11th largest loss
    assert metrics.es(x, 0.995) == pytest.approx(18.0, abs=0.6)   # mean of the 5 largest losses (16..20)
    assert metrics.es(x, 0.99) >= v


@pytest.fixture(scope="module")
def real():
    if not HAVE_DATA:
        pytest.skip("real curve not downloaded")
    cv = ingest.load_curve(CFG)
    ten = [2, 5, 10, 30]
    xr = bonds.excess_returns(cv, ten)
    dv = pd.DataFrame({m: bonds.dv01_par(cv[m].to_numpy(), m) for m in ten}, index=cv.index)
    return cv, xr, dv


def test_pca_hedge_neutralises_factors(real):
    cv, xr, dv = real
    win = cv.loc["2015":"2017", [2, 5, 10, 30]].diff().dropna()
    w = hedges.fit_weights("pca3", win, None, None, 30, [2, 5, 10])
    L = hedges.pca_loadings(win, 3)
    expo = L.loc[30].to_numpy() + sum(w[t] * L.loc[t].to_numpy() for t in [2, 5, 10])
    assert np.allclose(expo, 0, atol=1e-10)


def test_no_lookahead_future_shock_does_not_change_weights(real):
    cv, xr, dv = real
    cfg = dict(CFG)
    cfg["walk_forward"] = dict(CFG["walk_forward"], test_start="2015-01-01")
    cut = pd.Timestamp("2016-06-01")
    cv2 = cv.copy()
    cv2.loc[cv2.index >= cut, [2, 5, 10, 30]] += 0.03      # a large shock that only exists after the cut
    xr2 = bonds.excess_returns(cv2, [2, 5, 10, 30])
    dv2 = pd.DataFrame({m: bonds.dv01_par(cv2[m].to_numpy(), m) for m in [2, 5, 10, 30]}, index=cv2.index)
    rules = ["beta_10y", "pca2", "minvar"]
    _, _, W1 = hedges.walk_forward(cv.loc[:"2017"], xr.loc[:"2017"], dv.loc[:"2017"], cfg, rules)
    _, _, W2 = hedges.walk_forward(cv2.loc[:"2017"], xr2.loc[:"2017"], dv2.loc[:"2017"], cfg, rules)
    a = W1[W1["date"] <= cut].reset_index(drop=True)
    b = W2[W2["date"] <= cut].reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)
    assert (W1["train_end"] < W1["date"]).all()


def test_duration_hedge_removes_most_variance(real):
    cv, xr, dv = real
    cfg = dict(CFG)
    cfg["walk_forward"] = dict(CFG["walk_forward"], test_start="2010-01-01")
    P, _, _ = hedges.walk_forward(cv.loc[:"2014"], xr.loc[:"2014"], dv.loc[:"2014"], cfg, ["unhedged", "dv01_10y"])
    vr = metrics.variance_reduction(P["dv01_10y"], P["unhedged"])
    assert 0.6 < vr < 0.99
