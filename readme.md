[![DOI](https://zenodo.org/badge/732363363.svg)](https://doi.org/10.5281/zenodo.15480922)

# MASTER — US SPX Reproduction

Independent reproduction of the AAAI 2024 paper [**MASTER: Market-Guided Stock Transformer for Stock Price Forecasting**](https://ojs.aaai.org/index.php/AAAI/article/view/27767) ([arXiv](https://arxiv.org/abs/2312.15235)), adapted to **US equities (S&P 500 universe)** with daily Alpha158 factors and walk-forward training.

This repo is based on the authors’ public release ([Zenodo](https://doi.org/10.5281/zenodo.15480922)).

---

## Overview

MASTER is a stock transformer that:

1. Uses **market-guided gating** to re-weight stock features from macro/market signals
2. Applies **intra-stock** (temporal) and **inter-stock** (cross-sectional) attention
3. Aggregates the time dimension to produce a cross-sectional alpha score

**This implementation**

| Item | Setting |
|------|---------|
| Universe | S&P 500 constituents (CRSP daily, WRDS export) |
| Features | Alpha158 (158) + turnover + market gate (63) |
| Label | 5-day forward return: `close[t+5]/close[t] - 1` |
| Lookback | `STEP_LEN = 8` trading days |
| Training | 20-year rolling window, refit at **2000 / 2010 / 2020** |
| OOS horizon | 10 years per fold (combined OOS: **2000–2025**) |
| Portfolio | Long top 10% / short bottom 10%, rebalance every **5** trading days |

End-to-end workflow lives in [`train.ipynb`](train.ipynb): data prep → walk-forward training → IC / portfolio / CAPM analysis.

---

## Key Results

Statistics below are from the saved walk-forward OOS predictions (`outputs/walkforward/predictions.parquet`), evaluated in `train.ipynb` (Section 5). Portfolio metrics use **no slippage** (`SLIPPAGE = 0`); annualization uses `252 / 5 = 50.4` periods per year.

### Prediction quality (daily cross-sectional)

| Metric | Value |
|--------|------:|
| **Mean IC** | 0.0260 |
| **ICIR** | 0.4195 |
| **Mean Rank IC** | 0.0244 |
| **Rank ICIR** | 0.4980 |

**Per-fold OOS IC** (from `outputs/walkforward/metrics_summary.csv`):

| Refit year | OOS start | IC | ICIR | Rank IC | Rank ICIR |
|:----------:|:---------:|---:|-----:|--------:|----------:|
| 2000 | 2000 | 0.0497 | 0.418 | 0.0556 | 0.484 |
| 2010 | 2010 | 0.0126 | 0.095 | 0.0189 | 0.149 |
| 2020 | 2020 | 0.0164 | 0.106 | 0.0197 | 0.125 |

### Portfolio performance (long–short, top/bottom 10%)

| Strategy | Annual return | Sharpe |
|----------|--------------:|-------:|
| **Long–Short** | 28.40% | 2.86 |
| Long only | 27.69% | 3.96 |
| Short only | 0.70% | 0.11 |

### CAPM vs SPX (5-day holding periods)

Regression: `beta = corr(port, SPX) × std(port) / std(SPX)`,  
`alpha_period = mean(port) − beta × mean(SPX)`,  
`alpha_annual = alpha_period × 50.4`.

| Strategy | Beta | Alpha (annual) |
|----------|-----:|---------------:|
| **Long–Short** | −0.027 | 28.60% |
| Long only | −0.009 | 27.76% |
| Short only | −0.018 | 0.84% |

> **Note:** Backtest returns are **gross** (no transaction costs unless `SLIPPAGE` is set in the notebook). Results are in-sample to the walk-forward OOS design and depend on the CRSP panel and factor pipeline.

---

## Demo Figures

All figures are exported to [`demo/`](demo/) (7 images).

### Model architecture

![MASTER framework](demo/framework.png)

Market-guided gating → feature layer → temporal attention (intra-stock) → cross-sectional attention (inter-stock) → temporal pooling → return prediction.

### Information Coefficient (IC)

![Monthly IC and cumulative IC](demo/IC.png)

Daily cross-sectional Pearson correlation between model predictions and 5-day forward labels, aggregated to monthly bars with cumulative sum.

### Rank Information Coefficient (Rank IC)

![Monthly Rank IC and cumulative Rank IC](demo/RankIC.png)

Daily cross-sectional Spearman correlation between predicted and realized ranks.

### Portfolio PnL

![Long–short portfolio PnL](demo/portfolioPNL.png)

Period PnL, return distribution, and cumulative PnL for the long–short book.

### Portfolio PnL decomposition (no slippage)

![Cumulative PnL decomposition](demo/PNL_decomp.png)

Cumulative simple returns for long–short, long-only, and short-only legs (top/bottom 10%, no slippage).

### CAPM regression vs SPX

![Long–short CAPM scatter](demo/CAPM.png)

Scatter of long–short 5-day portfolio returns vs SPX 5-day returns with OLS fit (slope ≈ beta, intercept ≈ alpha per period).

### Portfolio turnover

![Rebalance turnover rate](demo/Turnover.png)

One-sided turnover at each rebalance date (average of long-leg and short-leg turnover), with a 12-period rolling mean.

---

## Quick Start

### 1. Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, pandas, pyqlib, pyarrow, matplotlib, jupyter.

### 2. Data

Place WRDS CRSP export at `data/spx_stock_data.csv` (gitignored), then either:

- Set `RUN_DATA_PREP = True` in `train.ipynb`, or
- Run scripts manually:

```bash
python scripts/download_market_data.py
python scripts/build_market_features.py
python scripts/build_master_panel.py
```

Panel output: `data/processed/master_panel_long.parquet`.

### 3. Train & analyze

Open `train.ipynb`:

| Flag | Purpose |
|------|---------|
| `SKIP_TRAINING = True` | Load saved predictions and run analysis only |
| `SKIP_TRAINING = False` | Run walk-forward training (hours on GPU) |
| `RETRAIN = True` | Delete old outputs and retrain from scratch |

Artifacts:

- `outputs/walkforward/predictions.parquet`
- `outputs/walkforward/metrics_summary.csv`
- `outputs/walkforward/models/us_{year}__0.pkl`

---

## Repository layout

```
MASTER_SPX/
├── config.py              # Hyperparameters & paths
├── master.py              # MASTER model (PyTorch)
├── base_model.py          # Training loop base class
├── train.ipynb            # Main notebook (train + analysis + plots)
├── scripts/
│   ├── build_master_panel.py
│   ├── run_walkforward.py
│   ├── eval_utils.py
│   └── ...
├── demo/                  # Exported figures for README / reports
├── data/                  # Raw & processed data (mostly gitignored)
└── outputs/walkforward/   # Predictions, metrics, checkpoints
```

---

## Citation

If you use this code or follow the MASTER method, please cite the original paper:

```bibtex
@inproceedings{master2024,
  title={MASTER: Market-Guided Stock Transformer for Stock Price Forecasting},
  booktitle={AAAI},
  year={2024}
}
```

Original code: [Zenodo 10.5281/zenodo.15480922](https://doi.org/10.5281/zenodo.15480922).
