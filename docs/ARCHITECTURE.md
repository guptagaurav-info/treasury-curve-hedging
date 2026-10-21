# How this project works

Three diagrams:
1. **Pipeline**: every step, its source file, and the logic inside it.
2. **Monthly hedge decision**: how one rebalance date becomes hedge notionals and a daily residual.
3. **Production design**: how the same logic would run as a daily hedge-and-control process. This is a proposed design and is not implemented in this repo.

Colour key: **blue** = external data source · **lavender** = code step (file named in the box) · **yellow** = model · **red** = check or gate · **grey** = output

## 1. Pipeline (what `make all` runs)

```mermaid
flowchart TD
    SRC["SOURCE: US Treasury Daily Par Yield Curve<br/>1990-01-02 to 2026-10-20<br/>Mirror: huggingface.co/datasets/defeatbeta/yahoo-finance-data"]

    subgraph S0["Stage 0 · Ingest (ingest.py)"]
      I1{"sha256 + dataset revision<br/>match configs/config.yaml?"}
      I1 -- "no" --> IX["Stop: source changed"]
      I1 -- "yes" --> I2["Tidy curve: one column per tenor<br/>reports/data_quality.csv (30y gap 2002-2006)"]
    end

    subgraph S1["Stage 1 · Returns (bonds.py)"]
      B1["Par bond per tenor bought at prior close<br/>repriced at today's par yield"]
      B2["Excess of 3-month bill financing<br/>DV01 per $ from the prior close"]
      B1 --> B2
    end

    subgraph S2["Stage 2 · Walk-forward (hedges.py)"]
      W1{"Rebalance on first trading day of each month<br/>train window ends BEFORE that day (asserted)"}
      W1 --> M0["Duration-matched 10y (baseline)"]
      W1 --> M1["Yield-beta 10y"]
      W1 --> M2["2- and 3-factor curve hedges (PCA)"]
      W1 --> M3["Minimum-variance regression 2y/5y/10y"]
      M0 & M1 & M2 & M3 --> H1["Hold weights for the month;<br/>re-size notionals daily from prior-close DV01"]
    end

    subgraph S3["Stage 3 · Validation (metrics.py, pipeline.py)"]
      V1["Residual vol, 99% VaR, 97.5% ES, worst day / 10 days"]
      V2["Checks: whole-sample fit (look-ahead) vs walk-forward,<br/>month-block bootstrap CI, split halves, stress windows,<br/>residual vs 10s30s"]
    end

    OUT["reports/metrics.json · hedge_summary.csv · stress_windows.csv<br/>hedge_weights_log.csv · figures/*.png"]

    SRC --> S0 --> S1 --> S2 --> S3 --> OUT
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e1,stroke:#b03a2e,color:#4a0f0a
    classDef out fill:#eeeeee,stroke:#777,color:#222
    class SRC src
    class M0,M1,M2,M3 model
    class I1,W1,IX gate
    class OUT out
```

## 2. Monthly hedge decision

```mermaid
flowchart TD
    A["Rebalance date d<br/>(first trading day of month)"] --> B["Trailing 3 years of curve changes<br/>and par-bond returns, all dated < d"]
    B --> C["Fit rule:<br/>regress 30y excess return on<br/>2y, 5y, 10y excess returns"]
    C --> D["Hedge weights as DV01 multiples<br/>e.g. Oct 2026: 10y -1.46x, 5y +0.46x, 2y +0.12x<br/>net 0.88x of book DV01"]
    D --> E["Each day t in the month:<br/>notional = weight x book DV01(t-1) / hedge DV01(t-1)"]
    E --> F["Residual P&L(t) = book + hedge legs<br/>all excess of bills"]
    F --> G{"Controls"}
    G --> G1["Residual VaR vs limit"]
    G --> G2["Gross hedge DV01 vs cap"]
    G --> G3["Residual vs 10s30s: is it a curve bet?"]
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e1,stroke:#b03a2e,color:#4a0f0a
    class C model
    class G gate
```

## 3. Production design (proposed, not implemented)

```mermaid
flowchart TD
    P1["Daily curve and futures prices<br/>(official close)"] --> P2{"Data checks:<br/>staleness, jumps > 4 sd, missing tenors"}
    P2 --> P3["Hedge engine: champion = min-variance<br/>challenger = yield-beta; baseline = duration"]
    P3 --> P4["Map curve-point weights to futures<br/>(TU / FV / TY) via CTD DV01"]
    P4 --> P5{"Limits: residual VaR, gross DV01,<br/>10s30s exposure"}
    P5 -- "breach" --> P6["Escalate to ALM / risk committee"]
    P5 -- "within" --> P7["Execution and hedge accounting file"]
    P7 --> P8["Monthly review: champion vs challenger<br/>on realised residual; fall back if beaten twice"]
    classDef gate fill:#fde2e1,stroke:#b03a2e,color:#4a0f0a
    class P2,P5 gate
```
