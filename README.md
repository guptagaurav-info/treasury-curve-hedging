# Is Your Rate Hedge a Curve Bet? Testing Long-Bond Hedges on 32 Years of Treasury Data

> Companion repos: [cmbs-distress-early-warning](https://github.com/guptagaurav-info/cmbs-distress-early-warning) · [risk-budgeting-multi-asset](https://github.com/guptagaurav-info/risk-budgeting-multi-asset) · [smb-credit-pd-lgd](https://github.com/guptagaurav-info/smb-credit-pd-lgd). Author: Gaurav Gupta.

A walk-forward test of how to hedge a long-duration Treasury position, built on the US Treasury's daily par yield curve from 1990 to 20 October 2026. It answers the question a treasurer, ALM committee, insurer or rates PM asks when long yields move: **if we hedge a 30-year exposure with the liquid part of the curve, how much risk is actually left, and what kind of risk is it?**

```bash
pip install -r requirements.txt
make all          # download the pinned curve (sha256-verified) -> par-bond returns -> walk-forward hedges -> reports
make test         # unit tests: bond maths, no look-ahead, factor neutrality, metrics
```

**How it works:** flowcharts of the pipeline, the monthly hedge decision and a proposed production design are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data

- **What:** US Treasury Daily Par Yield Curve Rates (1-month to 30-year), the official series published by the US Department of the Treasury ([home.treasury.gov](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve)).
- **Copy used:** the Hugging Face mirror `defeatbeta/yahoo-finance-data`, file `data/US/daily_treasury_yield.parquet` (licence ODC-BY), pinned to dataset revision `f9120ae6774148dfee69efb536dc1be94f1f7298`.
- **Checksum:** sha256 `13004d0befd8fa0c7cf7f2674b5129cdba875e33d3e3101aa15d25a931810ef0`, downloaded 2026-10-21. `ingest.py` stops if the bytes change.
- **Coverage:** 9,207 business days, 2 January 1990 to 20 October 2026. The 30-year series is missing from 19 February 2002 to 8 February 2006 (Treasury suspended 30-year issuance); no returns are computed across that gap. Data-quality counts are in `reports/data_quality.csv`.

## The question as a position

- **Book:** $1bn market value of a constant-maturity 30-year Treasury (DV01 about $1.47M per basis point at today's 5.42% yield). Illustrative size; every result scales linearly.
- **Hedge instruments:** the 2-, 5- and 10-year points, the liquid part of the curve where futures and on-the-run cash bonds trade.
- **Returns:** each curve point is a par bond bought at the prior close and repriced at today's par yield; every leg is financed at the 3-month bill rate, so all P&L is excess of funding. This is the standard way to turn a constant-maturity yield series into daily returns; one-day roll-down is ignored.

## Hedge rules compared

| Rule | What it does | Fitted on |
|---|---|---|
| Unhedged | Nothing | — |
| **Duration-matched, 10-year (baseline)** | Short 10-year DV01 equal to the book's DV01 | Nothing to fit |
| Yield-beta, 10-year | Short 10-year DV01 scaled by the regression of 30-year on 10-year yield changes | Trailing 3 years |
| Two-factor curve hedge | 2-year + 10-year sized so the book has zero exposure to the first two principal components (level, slope) | Trailing 3 years |
| Three-factor curve hedge | 2-, 5- and 10-year neutralising level, slope and curvature | Trailing 3 years |
| **Minimum-variance regression** | 2-, 5- and 10-year notionals from an OLS of the book's excess return on the hedges' excess returns | Trailing 3 years |

Every rule is refit on the first trading day of each month using only data **before** that day (asserted in code and in a unit test that shocks future data and checks that past weights do not move). Notionals are re-sized daily from the previous close's DV01. Test: January 1994 to October 2026, 6,691 trading days and 322 monthly refits; the 30-year gap pauses the test from February 2002 to March 2008 while fresh history accumulates.

## Results (walk-forward, out of sample)

| Rule | Residual volatility, $/yr | Variance removed | 1-day 99% VaR | Worst day | Fitted on the whole sample (look-ahead) |
|---|---|---|---|---|---|
| Unhedged | $150.4M | — | $23.3M | $72.0M | — |
| Duration-matched, 10y (baseline) | $60.4M | 83.9% | $10.5M | $45.2M | 83.9% |
| Yield-beta, 10y | $56.9M | 85.7% | $9.5M | $32.6M | 85.5% |
| Two-factor (2y+10y) | $57.6M | 85.3% | $9.5M | $48.3M | 87.0% |
| Three-factor (2y+5y+10y) | $82.6M | 69.8% | $13.7M | $59.9M | 79.1% |
| **Minimum-variance (2y+5y+10y)** | **$49.3M** | **89.3%** | **$8.3M** | $40.0M | 89.0% |

Per $1bn of 30-year exposure. Residual volatility ratio versus the duration hedge, 95% block-bootstrap CI (resampling whole months): minimum-variance 0.79-0.84; yield-beta 0.92-0.97; three-factor 1.25-1.51.

### Findings I would raise with an ALM or risk committee

**1. The duration-matched hedge is not a hedge of rates. It is a 10s30s curve position.**
- After the 10-year DV01 hedge, 97% of the remaining daily P&L is explained by the change in the 30-year minus 10-year spread: about $1.70M lost per basis point of steepening.
- That position made $68M in 2022 and lost $31M in March 2023 on a book that was nominally hedged. The minimum-variance hedge was -$13M and -$3M in the same windows.

**2. Long yields move less than 10-year yields, so DV01-matching over-hedges.**
- The 30-year moved 0.85bp per 1bp in the 10-year over the full sample, with yearly readings from 0.66 (1992) to 1.12 (2020).
- The yield-beta and minimum-variance hedges carry on average 0.87 and 0.83 times the book's net DV01, not 1.0.

**3. More factors made it worse.**
- The three-factor curve hedge left 37% more risk than the simple duration hedge out of sample, ran 4.5 times the book's DV01 in gross hedges and cost about 80 times as much to rebalance.
- Fitted on the whole sample it looks respectable (79% of variance removed against 70% out of sample): the curvature factor is estimated noise, and only a walk-forward test shows it.

**4. The improvement holds in both halves and in every stress window tested.**
- Test days before 2010: minimum-variance $50.6M vs duration $61.2M. 2010-2026: $48.4M vs $60.0M.
- Worst hedged day in March 2023 (SVB): $7.2M against $27.1M for the duration hedge; 2022: $13.9M against $20.1M.

### Illustrative economics (assumptions labelled)

- Risk capital at an assumed 3 x 10-day 99% VaR: $99.9M for the duration hedge, $79.0M for minimum variance, **$20.9M less per $1bn of 30-year exposure**. At an assumed 10% cost of capital, about $2.1M a year.
- Rebalancing cost at an assumed 0.25bp per $ of DV01 traded: about $166k a year for minimum variance, $10k for the duration hedge.

## What this does not show

- Constant-maturity par bonds, not specific CUSIPs or futures: no cheapest-to-deliver, futures basis, repo specialness or bid-ask on the bonds themselves.
- Treasury curve only; a corporate, mortgage or liability book adds spread and convexity risk that this test does not cover.
- Historical VaR on a single position; no portfolio diversification, no liquidity horizon beyond 10 days.
- Independent research on public data. Not deployed, not investment advice.

## Repo layout

```
configs/config.yaml       all choices, fixed before results were examined; data URL, revision, sha256
src/curvehedge/ingest.py  download + sha256 check; tidy the curve
src/curvehedge/bonds.py   par-bond pricing, DV01, daily excess returns
src/curvehedge/hedges.py  hedge rules and the walk-forward engine (no look-ahead, asserted)
src/curvehedge/metrics.py VaR, ES, block bootstrap
src/curvehedge/report.py  figures
src/curvehedge/pipeline.py end to end -> reports/
tests/                    bond maths, look-ahead, factor neutrality, metrics
reports/metrics.json      every number quoted in the article
```

MIT licence.
