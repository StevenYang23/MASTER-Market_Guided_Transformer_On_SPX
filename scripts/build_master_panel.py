"""Build daily Alpha158 + market gate + label panel from WRDS CRSP export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    LABEL_COL,
    MARKET_FEATURES_CSV,
    MASTER_PANEL_PARQUET,
    N_FACTORS,
    PANEL_END_DATE,
    PANEL_START_DATE,
    PROCESSED_DIR,
    SPX_STOCK_CSV,
)
from scripts.alpha158 import ALPHA158_NAMES
from scripts.qlib_alpha158 import compute_alpha158_panel

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

USECOLS = [
    "PERMNO", "DlyCalDt", "MbrStartDt", "MbrEndDt",
    "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol",
    "DlyCumFacPr", "DlyCumFacShr", "ShrOut",
    "ShareType", "SecurityType", "TradingStatusFlg",
]


def load_clean_crsp(csv_path: Path) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        parse_dates=["DlyCalDt", "MbrStartDt", "MbrEndDt"],
        low_memory=False,
    )
    df = df.rename(columns={
        "PERMNO": "permno",
        "DlyCalDt": "date",
        "MbrStartDt": "mbr_start",
        "MbrEndDt": "mbr_end",
        "DlyOpen": "open",
        "DlyHigh": "high",
        "DlyLow": "low",
        "DlyClose": "close",
        "DlyVol": "volume",
        "DlyCumFacPr": "cumfacpr",
        "DlyCumFacShr": "cumfacshr",
        "ShrOut": "shrout",
        "ShareType": "sharetype",
        "SecurityType": "securitytype",
        "TradingStatusFlg": "tradingstatus",
    })

    df = df[
        (df["securitytype"] == "EQTY")
        & (df["sharetype"] == "NS")
        & (df["tradingstatus"] == "A")
    ]
    df = df[(df["date"] >= df["mbr_start"]) & (df["date"] <= df["mbr_end"])]
    df = df[(df["date"] >= PANEL_START_DATE) & (df["date"] <= PANEL_END_DATE)]

    for col in ["open", "high", "low", "close", "volume", "cumfacpr", "cumfacshr", "shrout"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cumfacpr"] = df["cumfacpr"].replace(0, np.nan)
    df["cumfacshr"] = df["cumfacshr"].replace(0, np.nan)
    df["shrout"] = df["shrout"].replace(0, np.nan)

    # Calculate turnover rate (volume / (shrout * 1000))
    df["turnover"] = (df["volume"] / (df["shrout"] * 1000.0)).fillna(0.0)
    df["turnover"] = df["turnover"].replace([np.inf, -np.inf], 0.0)

    df["close"] = df["close"] / df["cumfacpr"]
    df["open"] = df["open"].fillna(df["close"]) / df["cumfacpr"]
    df["high"] = df["high"].fillna(df["close"]) / df["cumfacpr"]
    df["low"] = df["low"].fillna(df["close"]) / df["cumfacpr"]
    df["volume"] = df["volume"] * df["cumfacshr"]

    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    df = df.sort_values(["permno", "date"])
    print(
        f"Clean rows: {len(df):,} | permnos: {df['permno'].nunique():,} | "
        f"dates: {df['date'].min().date()} -> {df['date'].max().date()}"
    )
    return df


def build_alpha158_panel(df: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    del workers  # kept for CLI compatibility
    panel = compute_alpha158_panel(df)
    assert len(ALPHA158_NAMES) == 158
    return panel


def merge_market(panel: pd.DataFrame) -> pd.DataFrame:
    market = pd.read_csv(MARKET_FEATURES_CSV, index_col=0, parse_dates=True)
    market.index = pd.to_datetime(market.index).normalize()
    market = market.reset_index().rename(columns={"datetime": "date", "index": "date"})
    if "date" not in market.columns:
        market = market.rename(columns={market.columns[0]: "date"})
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    return panel.merge(market, on="date", how="inner")


def to_long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    factor_cols = ALPHA158_NAMES + ["turnover"]
    market_cols = [c for c in panel.columns if c not in factor_cols + [LABEL_COL, "date", "permno"]]
    feature_cols = factor_cols + market_cols

    out = panel.rename(columns={"date": "datetime", "permno": "instrument"})
    out["instrument"] = out["instrument"].astype(str)
    out = out.set_index(["datetime", "instrument"]).sort_index()
    ordered = feature_cols + [LABEL_COL]
    return out[ordered].astype(np.float32)


def main():
    if not SPX_STOCK_CSV.exists():
        raise FileNotFoundError(f"Missing {SPX_STOCK_CSV}")
    if not Path(MARKET_FEATURES_CSV).exists():
        raise FileNotFoundError(
            f"Missing {MARKET_FEATURES_CSV}. Run download_market_data.py and build_market_features.py first."
        )

    crsp = load_clean_crsp(SPX_STOCK_CSV)
    alpha = build_alpha158_panel(crsp)

    # Merge turnover from crsp into alpha
    turnover_df = crsp[["date", "permno", "turnover"]].copy()
    alpha = alpha.merge(turnover_df, on=["date", "permno"], how="inner")

    merged = merge_market(alpha)
    panel = to_long_panel(merged)
    panel.to_parquet(MASTER_PANEL_PARQUET)

    print(f"Panel shape: {panel.shape}")
    print(f"Features: {N_FACTORS} Alpha158 (Qlib) + {panel.shape[1] - N_FACTORS - 1} market + label")
    print(f"Date range: {panel.index.get_level_values(0).min()} -> {panel.index.get_level_values(0).max()}")
    print(f"Saved to {MASTER_PANEL_PARQUET}")


if __name__ == "__main__":
    main()
