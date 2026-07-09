"""Alpha158 technical factors (Qlib-compatible) from daily OHLCV."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]

KBAR_NAMES = [
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
]
PRICE_NAMES = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
ROLLING_NAMES = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD",
    "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
]
ALPHA158_NAMES = KBAR_NAMES + PRICE_NAMES + [
    f"{name}{w}" for w in WINDOWS for name in ROLLING_NAMES
]


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    def slope(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        x = np.arange(len(y), dtype=np.float64)
        if np.std(x) == 0:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    return s.rolling(window, min_periods=window).apply(slope, raw=True)


def _rolling_rsquare(s: pd.Series, window: int) -> pd.Series:
    def rsq(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        x = np.arange(len(y), dtype=np.float64)
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        corr = np.corrcoef(x, y)[0, 1]
        return corr * corr

    return s.rolling(window, min_periods=window).apply(rsq, raw=True)


def _rolling_resi(s: pd.Series, window: int) -> pd.Series:
    def resi(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        x = np.arange(len(y), dtype=np.float64)
        if len(x) < 2:
            return np.nan
        coef = np.polyfit(x, y, 1)
        pred = coef[0] * x[-1] + coef[1]
        return y[-1] - pred

    return s.rolling(window, min_periods=window).apply(resi, raw=True)


def _rolling_idxmax(s: pd.Series, window: int) -> pd.Series:
    def idxmax(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        return (len(y) - 1 - int(np.argmax(y))) / window

    return s.rolling(window, min_periods=window).apply(idxmax, raw=True)


def _rolling_idxmin(s: pd.Series, window: int) -> pd.Series:
    def idxmin(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        return (len(y) - 1 - int(np.argmin(y))) / window

    return s.rolling(window, min_periods=window).apply(idxmin, raw=True)


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    def rank_last(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        return pd.Series(y).rank(pct=True).iloc[-1]

    return s.rolling(window, min_periods=window).apply(rank_last, raw=True)


def compute_alpha158_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Alpha158 for one stock. df must have open/high/low/close/volume sorted by date."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    vwap = df["vwap"] if "vwap" in df.columns else c
    eps = 1e-12

    feats = {}
    feats["KMID"] = (c - o) / (o + eps)
    feats["KLEN"] = (h - l) / (o + eps)
    hl = h - l
    feats["KMID2"] = (c - o) / (hl + eps)
    feats["KUP"] = (h - np.maximum(o, c)) / (o + eps)
    feats["KUP2"] = (h - np.maximum(o, c)) / (hl + eps)
    feats["KLOW"] = (np.minimum(o, c) - l) / (o + eps)
    feats["KLOW2"] = (np.minimum(o, c) - l) / (hl + eps)
    feats["KSFT"] = (2 * c - h - l) / (o + eps)
    feats["KSFT2"] = (2 * c - h - l) / (hl + eps)
    feats["OPEN0"] = o / (c + eps)
    feats["HIGH0"] = h / (c + eps)
    feats["LOW0"] = l / (c + eps)
    feats["VWAP0"] = vwap / (c + eps)

    ret = c.pct_change()
    vol_chg = v.pct_change()
    up = (c > c.shift(1)).astype(float)
    down = (c < c.shift(1)).astype(float)
    abs_ret = (c - c.shift(1)).abs()
    vol_up = (v > v.shift(1)).astype(float)
    vol_down = (v < v.shift(1)).astype(float)
    abs_vol = (v - v.shift(1)).abs()
    wvol = (ret.abs() * v).replace([np.inf, -np.inf], np.nan)

    for d in WINDOWS:
        feats[f"ROC{d}"] = c.shift(d) / (c + eps)
        feats[f"MA{d}"] = c.rolling(d, min_periods=d).mean() / (c + eps)
        feats[f"STD{d}"] = c.rolling(d, min_periods=d).std() / (c + eps)
        feats[f"BETA{d}"] = _rolling_slope(c, d) / (c + eps)
        feats[f"RSQR{d}"] = _rolling_rsquare(c, d)
        feats[f"RESI{d}"] = _rolling_resi(c, d) / (c + eps)
        feats[f"MAX{d}"] = h.rolling(d, min_periods=d).max() / (c + eps)
        feats[f"MIN{d}"] = l.rolling(d, min_periods=d).min() / (c + eps)
        feats[f"QTLU{d}"] = c.rolling(d, min_periods=d).quantile(0.8) / (c + eps)
        feats[f"QTLD{d}"] = c.rolling(d, min_periods=d).quantile(0.2) / (c + eps)
        feats[f"RANK{d}"] = _rolling_rank(c, d)
        lo = l.rolling(d, min_periods=d).min()
        hi = h.rolling(d, min_periods=d).max()
        feats[f"RSV{d}"] = (c - lo) / (hi - lo + eps)
        feats[f"IMAX{d}"] = _rolling_idxmax(h, d)
        feats[f"IMIN{d}"] = _rolling_idxmin(l, d)
        feats[f"IMXD{d}"] = feats[f"IMAX{d}"] - feats[f"IMIN{d}"]
        feats[f"CORR{d}"] = c.rolling(d, min_periods=d).corr(np.log(v + 1))
        feats[f"CORD{d}"] = ret.rolling(d, min_periods=d).corr(np.log(vol_chg + 1))
        feats[f"CNTP{d}"] = up.rolling(d, min_periods=d).mean()
        feats[f"CNTN{d}"] = down.rolling(d, min_periods=d).mean()
        feats[f"CNTD{d}"] = feats[f"CNTP{d}"] - feats[f"CNTN{d}"]
        abs_sum = abs_ret.rolling(d, min_periods=d).sum() + eps
        feats[f"SUMP{d}"] = (c - c.shift(1)).clip(lower=0).rolling(d, min_periods=d).sum() / abs_sum
        feats[f"SUMN{d}"] = (c.shift(1) - c).clip(lower=0).rolling(d, min_periods=d).sum() / abs_sum
        feats[f"SUMD{d}"] = feats[f"SUMP{d}"] - feats[f"SUMN{d}"]
        feats[f"VMA{d}"] = v.rolling(d, min_periods=d).mean() / (v + eps)
        feats[f"VSTD{d}"] = v.rolling(d, min_periods=d).std() / (v + eps)
        wv_mean = wvol.rolling(d, min_periods=d).mean() + eps
        feats[f"WVMA{d}"] = wvol.rolling(d, min_periods=d).std() / wv_mean
        vol_abs_sum = abs_vol.rolling(d, min_periods=d).sum() + eps
        feats[f"VSUMP{d}"] = (v - v.shift(1)).clip(lower=0).rolling(d, min_periods=d).sum() / vol_abs_sum
        feats[f"VSUMN{d}"] = (v.shift(1) - v).clip(lower=0).rolling(d, min_periods=d).sum() / vol_abs_sum
        feats[f"VSUMD{d}"] = feats[f"VSUMP{d}"] - feats[f"VSUMN{d}"]

    out = pd.DataFrame(feats, index=df.index)
    return out[ALPHA158_NAMES]


def compute_label(close: pd.Series, forward_days: int = 5) -> pd.Series:
    """Paper label: Ref(close,-5)/Ref(close,-1) - 1."""
    return close.shift(-forward_days) / close.shift(-1) - 1
