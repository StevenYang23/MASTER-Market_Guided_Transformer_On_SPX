"""Build daily Alpha158 + market gate + label panel from WRDS CRSP export."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    LABEL_COL, LABEL_FORWARD_DAYS, MARKET_FEATURES_CSV,
    MASTER_PANEL_PARQUET, N_FACTORS, PANEL_END_DATE, PANEL_START_DATE,
    PROCESSED_DIR, SPX_STOCK_CSV,
)
from scripts.alpha158 import ALPHA158_NAMES, compute_alpha158_frame, compute_label

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

USECOLS = [
    "PERMNO", "DlyCalDt", "MbrStartDt", "MbrEndDt",
    "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol",
    "DlyCumFacPr", "DlyCumFacShr",
    "ShareType", "SecurityType", "TradingStatusFlg",
]


def _alpha158_one(group: pd.DataFrame) -> pd.DataFrame:
    permno = group["permno"].iloc[0]
    g = group.set_index("date")
    feats = compute_alpha158_frame(g)
    label = compute_label(g["close"], LABEL_FORWARD_DAYS).rename(LABEL_COL)
    out = feats.join(label)
    # Avoid large whole-table replace() later (can trigger OOM).
    cols = ALPHA158_NAMES + [LABEL_COL]
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    out["permno"] = permno
    return out.reset_index()


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

    for col in ["open", "high", "low", "close", "volume", "cumfacpr", "cumfacshr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cumfacpr"] = df["cumfacpr"].replace(0, np.nan)
    df["cumfacshr"] = df["cumfacshr"].replace(0, np.nan)
    df["close"] = df["close"] / df["cumfacpr"]
    df["open"] = df["open"].fillna(df["close"]) / df["cumfacpr"]
    df["high"] = df["high"].fillna(df["close"]) / df["cumfacpr"]
    df["low"] = df["low"].fillna(df["close"]) / df["cumfacpr"]
    df["volume"] = df["volume"] * df["cumfacshr"]
    df["vwap"] = df["close"]

    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    df = df.sort_values(["permno", "date"])
    print(f"Clean rows: {len(df):,} | permnos: {df['permno'].nunique():,} | "
          f"dates: {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def build_alpha158_panel(df: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    groups = [g for _, g in df.groupby("permno", sort=False)]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in tqdm(pool.map(_alpha158_one, groups), total=len(groups), desc="Alpha158 per stock"):
            parts.append(out)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.dropna(subset=[LABEL_COL])
    factor_cols = ALPHA158_NAMES
    panel = panel.dropna(subset=factor_cols, how="all")
    # Reduce memory footprint early.
    panel[factor_cols + [LABEL_COL]] = panel[factor_cols + [LABEL_COL]].astype(np.float32, copy=False)
    panel[factor_cols] = panel[factor_cols].fillna(0)
    assert len(factor_cols) == N_FACTORS
    return panel


def merge_market(panel: pd.DataFrame) -> pd.DataFrame:
    market = pd.read_csv(MARKET_FEATURES_CSV, index_col=0, parse_dates=True)
    market.index = pd.to_datetime(market.index).normalize()
    market = market.reset_index().rename(columns={"datetime": "date", "index": "date"})
    if "date" not in market.columns:
        market = market.rename(columns={market.columns[0]: "date"})
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market_cols = [c for c in market.columns if c != "date"]
    return panel.merge(market, on="date", how="inner")


def to_long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    factor_cols = ALPHA158_NAMES
    market_cols = [c for c in panel.columns if c not in factor_cols + [LABEL_COL, "date", "permno"]]
    feature_cols = factor_cols + market_cols

    out = panel.rename(columns={"date": "datetime", "permno": "instrument"})
    out["instrument"] = out["instrument"].astype(str)
    out = out.set_index(["datetime", "instrument"]).sort_index()
    ordered = feature_cols + [LABEL_COL]
    # float32 is enough for training and saves memory.
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
    merged = merge_market(alpha)
    panel = to_long_panel(merged)
    panel.to_parquet(MASTER_PANEL_PARQUET)

    print(f"Panel shape: {panel.shape}")
    print(f"Features: {N_FACTORS} Alpha158 + {panel.shape[1] - N_FACTORS - 1} market + label")
    print(f"Date range: {panel.index.get_level_values(0).min()} -> {panel.index.get_level_values(0).max()}")
    print(f"Saved to {MASTER_PANEL_PARQUET}")


if __name__ == "__main__":
    main()
