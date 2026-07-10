"""Evaluation helpers: IC series and eval frame from saved predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_COL


def build_eval_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Build analysis frame from predictions saved at train time (pred + label)."""
    if LABEL_COL not in pred_df.columns:
        raise ValueError(
            f"Missing {LABEL_COL} in predictions. "
            "Re-run training so OOS labels are saved with predictions."
        )

    cols = ["pred", LABEL_COL]
    if "refit_year" in pred_df.columns:
        cols.append("refit_year")

    df = pred_df[cols].copy()
    df = df.dropna(subset=["pred", LABEL_COL])
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("predictions index must be MultiIndex (datetime, instrument)")
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.sort_index()


def calc_ic_series(df: pd.DataFrame) -> pd.DataFrame:
    def _per_date(g: pd.DataFrame) -> pd.Series:
        if len(g) < 5:
            return pd.Series({"IC": np.nan, "RankIC": np.nan, "n": len(g)})
        return pd.Series(
            {
                "IC": g["pred"].corr(g[LABEL_COL]),
                "RankIC": g["pred"].corr(g[LABEL_COL], method="spearman"),
                "n": len(g),
            }
        )

    ic = df.groupby(level="datetime", group_keys=False).apply(_per_date, include_groups=False)
    ic = ic.dropna(subset=["IC"])
    ic["cum_IC"] = ic["IC"].cumsum()
    ic["cum_RankIC"] = ic["RankIC"].cumsum()
    return ic


def verify_saved_ic(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Check that recomputed IC matches metrics saved during training."""
    rows = []
    for year, g in pred_df.groupby("refit_year"):
        eval_df = build_eval_frame(g)
        ic = calc_ic_series(eval_df)["IC"]
        saved = metrics_df.loc[metrics_df["refit_year"] == year, "IC"]
        saved_ic = float(saved.iloc[0]) if len(saved) else np.nan
        recomputed = float(ic.mean())
        rows.append(
            {
                "refit_year": year,
                "saved_IC": saved_ic,
                "recomputed_IC": recomputed,
                "gap": recomputed - saved_ic,
            }
        )
    return pd.DataFrame(rows)
