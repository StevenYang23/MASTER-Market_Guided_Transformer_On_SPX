"""Merge 153 factors, 63 market features, and label into a long panel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import (
    FACTOR_PARQUET, LABEL_COL, MARKET_FEATURES_CSV,
    MASTER_PANEL_PARQUET, META_COLS, N_FACTORS, PROCESSED_DIR,
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_factor_cols(columns) -> list:
    return list(columns[-N_FACTORS:])


def main():
    df = pd.read_parquet(FACTOR_PARQUET)
    factor_cols = get_factor_cols(df.columns)
    assert len(factor_cols) == N_FACTORS, f"Expected {N_FACTORS} factors, got {len(factor_cols)}"

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ym"] = df["date"].dt.to_period("M")

    market = pd.read_csv(MARKET_FEATURES_CSV, index_col=0, parse_dates=True)
    market.index = pd.to_datetime(market.index).normalize()
    market = market.reset_index().rename(columns={"datetime": "date", "index": "date"})
    if "date" not in market.columns:
        market = market.rename(columns={market.columns[0]: "date"})
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market["ym"] = market["date"].dt.to_period("M")
    market_cols = [c for c in market.columns if c not in ("date", "ym")]

    panel = df[META_COLS + factor_cols].copy()
    panel["ym"] = panel["date"].dt.to_period("M")
    panel = panel.merge(market[["ym"] + market_cols], on="ym", how="inner")
    panel = panel.drop(columns=["ym"])
    panel = panel.rename(columns={"date": "datetime", "permno": "instrument"})
    panel["instrument"] = panel["instrument"].astype(str)
    panel = panel.set_index(["datetime", "instrument"]).sort_index()

    feature_cols = factor_cols + market_cols
    ordered = feature_cols + [LABEL_COL]
    panel = panel[ordered].astype(np.float64)
    panel.to_parquet(MASTER_PANEL_PARQUET)
    print(f"Panel shape: {panel.shape}")
    print(f"Features: {len(feature_cols)} + label = {len(ordered)}")
    print(f"Date range: {panel.index.get_level_values(0).min()} -> {panel.index.get_level_values(0).max()}")
    print(f"Saved to {MASTER_PANEL_PARQUET}")


if __name__ == "__main__":
    main()
