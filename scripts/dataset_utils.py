"""Dataset utilities: normalization and time-series sampling for MASTER."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config import LABEL_COL, STEP_LEN, TRAIN_YEARS, OOS_YEARS, VAL_RATIO


def robust_zscore_norm(df: pd.DataFrame, fit_df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    out = df.astype(np.float64, copy=True)
    feature_cols = list(out.columns[:-1])
    med = fit_df[feature_cols].astype(np.float64).median()
    mad = (fit_df[feature_cols].astype(np.float64) - med).abs().median().replace(0, 1.0)
    normed = (out[feature_cols] - med) / mad
    out[feature_cols] = normed.clip(-clip, clip).fillna(0)
    return out


def get_fold_dates(refit_year: int, train_years: int = TRAIN_YEARS, oos_years: int = OOS_YEARS, val_ratio: float = VAL_RATIO):
    train_end = pd.Timestamp(f"{refit_year}-01-01")
    train_start = train_end - pd.DateOffset(years=train_years)
    oos_end = train_end + pd.DateOffset(years=oos_years) - pd.DateOffset(days=1)

    train_months = pd.period_range(train_start.to_period("M"), (train_end - pd.DateOffset(days=1)).to_period("M"), freq="M")
    n_val = max(1, int(len(train_months) * val_ratio))
    val_months = set(train_months[-n_val:])
    fit_months = set(train_months[:-n_val])

    return {
        "train_start": train_start,
        "train_end": train_end - pd.DateOffset(days=1),
        "oos_start": train_end,
        "oos_end": oos_end,
        "fit_months": fit_months,
        "val_months": val_months,
        "train_months": set(train_months),
        "oos_months": set(pd.period_range(train_end.to_period("M"), oos_end.to_period("M"), freq="M")),
        "lookback_start": train_start - pd.DateOffset(months=STEP_LEN),
    }


class MasterTSDataset(Dataset):
    def __init__(
        self,
        panel: pd.DataFrame,
        target_months: set,
        feature_cols: list,
        label_col: str = LABEL_COL,
        step_len: int = STEP_LEN,
        show_progress: bool = False,
    ):
        self.step_len = step_len
        self.all_cols = feature_cols + [label_col]
        self.samples = []
        self.index = []

        panel = panel.copy()
        panel["ym"] = panel.index.get_level_values("datetime").to_period("M")

        if not set(panel.loc[panel["ym"].isin(target_months)].index.get_level_values(0)):
            raise ValueError("No samples for provided months")

        instruments = panel.groupby(level="instrument")
        for inst, grp in tqdm(instruments, desc="build sequences", leave=False, disable=not show_progress):
            grp = grp.sort_index(level="datetime")
            dt_index = grp.index.get_level_values("datetime")
            ym_index = grp["ym"].to_numpy()
            values = grp[self.all_cols].to_numpy(dtype=np.float32)

            for i in range(step_len - 1, len(grp)):
                if ym_index[i] not in target_months:
                    continue
                window = values[i - step_len + 1 : i + 1].copy()
                window[:-1, -1] = np.nan
                self.samples.append(window)
                self.index.append((dt_index[i], str(inst)))

        if not self.samples:
            raise ValueError("No sequence samples built; check date coverage / lookback")

        self.data_index = pd.MultiIndex.from_tuples(self.index, names=["datetime", "instrument"])

    def get_index(self):
        return self.data_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            idx = np.asarray(idx).astype(int)
            return torch.stack([torch.from_numpy(self.samples[i]) for i in idx])
        return torch.from_numpy(self.samples[int(idx)])


def slice_panel_by_dates(panel: pd.DataFrame, date_start, date_end) -> pd.DataFrame:
    dt = panel.index.get_level_values("datetime")
    mask = (dt >= pd.Timestamp(date_start)) & (dt <= pd.Timestamp(date_end))
    return panel.loc[mask]


def prepare_fold_data(panel: pd.DataFrame, refit_year: int, val_ratio: float = VAL_RATIO, show_progress: bool = True):
    fold = get_fold_dates(refit_year, val_ratio=val_ratio)
    feature_cols = [c for c in panel.columns if c != LABEL_COL]

    steps = ["slice panel", "normalize", "train ds", "valid ds", "oos ds"]
    bar = tqdm(steps, desc=f"prepare {refit_year}", leave=False, disable=not show_progress)
    for step in bar:
        bar.set_postfix_str(step)
        if step == "slice panel":
            hist_panel = slice_panel_by_dates(panel, fold["lookback_start"], fold["oos_end"])
            fit_panel = hist_panel.loc[hist_panel.index.get_level_values("datetime").to_period("M").isin(fold["fit_months"])]
        elif step == "normalize":
            normed_hist = robust_zscore_norm(hist_panel, fit_panel)
        elif step == "train ds":
            train_ds = MasterTSDataset(normed_hist, fold["fit_months"], feature_cols, show_progress=show_progress)
        elif step == "valid ds":
            val_ds = MasterTSDataset(normed_hist, fold["val_months"], feature_cols, show_progress=show_progress)
        elif step == "oos ds":
            oos_ds = MasterTSDataset(normed_hist, fold["oos_months"], feature_cols, show_progress=show_progress)

    return fold, train_ds, val_ds, oos_ds
