"""Build 63 market-gate features from SPX/NDX/DJI daily data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import INDICES, MARKET_FEATURES_CSV, MARKET_RAW_PARQUET, ROLLING_WINDOWS, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)


def build_index_features(close: pd.Series, dollar_vol: pd.Series, prefix: str) -> pd.DataFrame:
    ret = close.pct_change()
    feats = {f"{prefix}_ret_1d": ret}
    for d in ROLLING_WINDOWS:
        feats[f"{prefix}_ret_mean_{d}d"] = ret.rolling(d, min_periods=1).mean()
        feats[f"{prefix}_ret_std_{d}d"] = ret.rolling(d, min_periods=1).std().fillna(0)
        roll_amt = dollar_vol.rolling(d, min_periods=1).mean()
        feats[f"{prefix}_amt_mean_ratio_{d}d"] = roll_amt / dollar_vol.replace(0, np.nan)
        roll_std = dollar_vol.rolling(d, min_periods=1).std().fillna(0)
        feats[f"{prefix}_amt_std_ratio_{d}d"] = roll_std / dollar_vol.replace(0, np.nan)
    return pd.DataFrame(feats)


def main():
    raw = pd.read_parquet(MARKET_RAW_PARQUET)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    all_feats = []
    for name in INDICES:
        sub = raw[raw["index_name"] == name].sort_values("date").set_index("date")
        close = sub[f"{name}_close"]
        dollar_vol = sub[f"{name}_dollar_volume"]
        all_feats.append(build_index_features(close, dollar_vol, name))

    market = pd.concat(all_feats, axis=1)
    market.index.name = "datetime"
    market = market.replace([np.inf, -np.inf], np.nan)
    market.to_csv(MARKET_FEATURES_CSV)
    print(f"Saved {market.shape[1]} market features x {len(market)} days to {MARKET_FEATURES_CSV}")


if __name__ == "__main__":
    main()
