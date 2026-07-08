"""Download SPX, NDX, DJI monthly OHLCV via yfinance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
from config import INDICES, MARKET_RAW_PARQUET, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)


def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    monthly = df.resample("ME").last()
    monthly.index = monthly.index.normalize()
    return monthly


def download_index(ticker: str, start: str = "1979-01-01", end: str = "2026-01-01") -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = to_month_end(data)
    data["dollar_volume"] = data["Close"] * data["Volume"]
    data["ret"] = data["Close"].pct_change()
    return data


def main():
    frames = []
    for name, ticker in INDICES.items():
        print(f"Downloading {name} ({ticker})...")
        df = download_index(ticker)
        df = df.rename(columns={
            "Open": f"{name}_open", "High": f"{name}_high", "Low": f"{name}_low",
            "Close": f"{name}_close", "Volume": f"{name}_volume",
            "dollar_volume": f"{name}_dollar_volume", "ret": f"{name}_ret",
        })
        df["index_name"] = name
        frames.append(df.reset_index().rename(columns={"index": "date", "Date": "date"}))

    long_df = pd.concat(frames, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.normalize()
    long_df.to_parquet(MARKET_RAW_PARQUET, index=False)
    print(f"Saved {len(long_df)} rows to {MARKET_RAW_PARQUET}")


if __name__ == "__main__":
    main()
