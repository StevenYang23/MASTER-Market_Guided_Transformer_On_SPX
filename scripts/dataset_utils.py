"""Dataset utilities: normalization and time-series sampling for MASTER (daily)."""
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config import LABEL_COL, STEP_LEN, TRAIN_YEARS, OOS_YEARS, VAL_RATIO

# Qlib RobustZScoreNorm scales MAD by 1.4826 to approximate std for normal data.
MAD_SCALE = np.float32(1.4826)


def compute_norm_stats(fit_df: pd.DataFrame, feature_cols: list) -> tuple[pd.Series, pd.Series]:
    med = fit_df[feature_cols].median().astype(np.float32)
    mad = pd.Series(np.float32(1.0), index=feature_cols, dtype=np.float32)
    for col in feature_cols:
        fit_arr = fit_df[col].to_numpy(dtype=np.float32, copy=False)
        col_mad = np.median(np.abs(fit_arr - med[col]))
        mad[col] = np.float32(1.0 if col_mad == 0 else col_mad * MAD_SCALE)
    return med, mad


def apply_robust_zscore_norm(
    df: pd.DataFrame,
    med: pd.Series,
    mad: pd.Series,
    feature_cols: list | None = None,
    clip: float = 3.0,
) -> pd.DataFrame:
    feature_cols = feature_cols or list(df.columns[:-1])
    for col in feature_cols:
        arr = df[col].to_numpy(dtype=np.float32, copy=True)
        arr -= med[col]
        arr /= mad[col]
        np.clip(arr, -clip, clip, out=arr)
        arr = np.nan_to_num(arr, nan=0.0, copy=False)
        df[col] = arr
    return df


def robust_zscore_norm(df: pd.DataFrame, fit_df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Robust normalization using median/MAD (column-wise, in-place)."""
    feature_cols = list(df.columns[:-1])
    med, mad = compute_norm_stats(fit_df, feature_cols)
    return apply_robust_zscore_norm(df, med, mad, feature_cols=feature_cols, clip=clip)


class PanelInstrumentCache:
    """Per-instrument float32 arrays built once and shared across datasets."""

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_cols: list,
        label_col: str = LABEL_COL,
        show_progress: bool = False,
    ):
        self.all_cols = feature_cols + [label_col]
        self._values_by_instrument = {}
        self._dt_by_instrument = {}

        instruments = panel.index.get_level_values("instrument").unique()
        for inst in tqdm(
            instruments,
            desc="cache instruments",
            leave=False,
            disable=not show_progress,
        ):
            inst_key = str(inst)
            grp = panel.xs(inst, level="instrument", drop_level=False)
            self._dt_by_instrument[inst_key] = grp.index.get_level_values("datetime")
            self._values_by_instrument[inst_key] = grp[self.all_cols].to_numpy(
                dtype=np.float32, copy=True
            )
            del grp

    def get(self, inst_key: str):
        return self._dt_by_instrument[inst_key], self._values_by_instrument[inst_key]


def _trading_dates_between(panel: pd.DataFrame, start, end) -> list:
    dt = panel.index.get_level_values("datetime")
    mask = (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))
    return sorted(dt[mask].unique())


def get_fold_dates(
    panel: pd.DataFrame,
    refit_year: int,
    train_years: int = TRAIN_YEARS,
    oos_years: int = OOS_YEARS,
    val_ratio: float = VAL_RATIO,
    smoke: bool = False,
    smoke_max_fit_dates: int = 60,
    smoke_max_val_dates: int = 20,
    smoke_max_oos_dates: int = 20,
):
    train_end = pd.Timestamp(f"{refit_year}-01-01")
    train_start = train_end - pd.DateOffset(years=train_years)
    oos_end = train_end + pd.DateOffset(years=oos_years) - pd.DateOffset(days=1)

    train_dates = _trading_dates_between(panel, train_start, train_end - pd.DateOffset(days=1))
    oos_dates = _trading_dates_between(panel, train_end, oos_end)

    n_val = max(1, int(len(train_dates) * val_ratio))
    val_dates = set(train_dates[-n_val:])
    fit_dates = set(train_dates[:-n_val])

    lookback_start = train_start - pd.DateOffset(days=90)

    if smoke:
        # Smoke mode keeps only a small recent slice of each split.
        # This reduces normalization memory + sequence count in MasterTSDataset.
        train_dates_sorted = sorted(train_dates)
        fit_sorted = train_dates_sorted[:-n_val]
        val_sorted = train_dates_sorted[-n_val:]
        oos_sorted = sorted(oos_dates)

        fit_dates = set(fit_sorted[-smoke_max_fit_dates:]) if fit_sorted else set()
        val_dates = set(val_sorted[-smoke_max_val_dates:]) if val_sorted else set()
        oos_dates = set(oos_sorted[-smoke_max_oos_dates:]) if oos_sorted else set()

    return {
        "train_start": train_start,
        "train_end": train_end - pd.DateOffset(days=1),
        "oos_start": train_end,
        "oos_end": oos_end,
        "fit_dates": fit_dates,
        "val_dates": val_dates,
        "train_dates": set(train_dates),
        "oos_dates": set(oos_dates),
        "lookback_start": lookback_start,
    }


class MasterTSDataset(Dataset):
    def __init__(
        self,
        panel_or_cache,
        target_dates: set,
        feature_cols: list | None = None,
        label_col: str = LABEL_COL,
        step_len: int = STEP_LEN,
        show_progress: bool = False,
    ):
        self.step_len = step_len
        if isinstance(panel_or_cache, PanelInstrumentCache):
            self._cache = panel_or_cache
            self.all_cols = self._cache.all_cols
        else:
            panel = panel_or_cache
            cols = feature_cols or [c for c in panel.columns if c != label_col]
            self.all_cols = cols + [label_col]
            self._cache = PanelInstrumentCache(
                panel, cols, label_col=label_col, show_progress=show_progress
            )

        self._seq_locs = []
        self.index = []

        for inst_key, dt_index in tqdm(
            self._cache._dt_by_instrument.items(),
            desc="index sequences",
            leave=False,
            disable=not show_progress,
        ):
            values = self._cache._values_by_instrument[inst_key]
            for i in range(step_len - 1, len(dt_index)):
                if dt_index[i] not in target_dates:
                    continue
                self._seq_locs.append((inst_key, i))
                self.index.append((dt_index[i], inst_key))

        if not self._seq_locs:
            raise ValueError("No sequence samples built; check date coverage / lookback")

        self.data_index = pd.MultiIndex.from_tuples(self.index, names=["datetime", "instrument"])

    def get_index(self):
        return self.data_index

    def __len__(self):
        return len(self._seq_locs)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            idx = np.asarray(idx).astype(int)
            windows = []
            for i in idx:
                inst_key, end_pos = self._seq_locs[i]
                values = self._cache._values_by_instrument[inst_key]
                window = values[end_pos - self.step_len + 1 : end_pos + 1]
                windows.append(torch.from_numpy(window))
            return torch.stack(windows)

        inst_key, end_pos = self._seq_locs[int(idx)]
        values = self._cache._values_by_instrument[inst_key]
        window = values[end_pos - self.step_len + 1 : end_pos + 1]
        return torch.from_numpy(window)


def _parquet_slice_to_indexed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pyarrow parquet slices to a (datetime, instrument) MultiIndex frame."""
    if isinstance(df.index, pd.MultiIndex) and set(df.index.names) >= {"datetime", "instrument"}:
        dt = df.index.get_level_values("datetime")
        inst = df.index.get_level_values("instrument").astype(str)
        df.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
        return df.sort_index()
    if "datetime" in df.columns and "instrument" in df.columns:
        df = df.copy()
        df["instrument"] = df["instrument"].astype(str)
        return df.set_index(["datetime", "instrument"]).sort_index()
    raise KeyError("Could not reconstruct datetime/instrument index for parquet slice.")


def load_labels_for_range(parquet_path, date_start, date_end) -> pd.DataFrame:
    """Load label column for a datetime range without reading the full panel."""
    table = pq.read_table(
        str(parquet_path),
        columns=[LABEL_COL, "datetime", "instrument"],
        filters=[
            ("datetime", ">=", pd.Timestamp(date_start)),
            ("datetime", "<=", pd.Timestamp(date_end)),
        ],
    )
    df = _parquet_slice_to_indexed_df(table.to_pandas())
    return df[[LABEL_COL]].sort_index()


def slice_panel_by_dates(panel: pd.DataFrame, date_start, date_end) -> pd.DataFrame:
    dt = panel.index.get_level_values("datetime")
    mask = (dt >= pd.Timestamp(date_start)) & (dt <= pd.Timestamp(date_end))
    return panel.loc[mask]


def _build_datasets_from_panel(
    panel,
    fold,
    feature_cols,
    show_progress: bool,
    norm_stats=None,
    target_splits=("fit", "val", "oos"),
):
    fit_panel = panel.loc[panel.index.get_level_values("datetime").isin(fold["fit_dates"])]
    if norm_stats is None:
        med, mad = compute_norm_stats(fit_panel, feature_cols)
        norm_stats = (med, mad)
    else:
        med, mad = norm_stats
    apply_robust_zscore_norm(panel, med, mad, feature_cols=feature_cols)
    del fit_panel

    cache = PanelInstrumentCache(panel, feature_cols, show_progress=show_progress)
    del panel
    gc.collect()

    datasets = {}
    if "fit" in target_splits:
        datasets["train"] = MasterTSDataset(cache, fold["fit_dates"], show_progress=show_progress)
    if "val" in target_splits:
        datasets["val"] = MasterTSDataset(cache, fold["val_dates"], show_progress=show_progress)
    if "oos" in target_splits:
        datasets["oos"] = MasterTSDataset(cache, fold["oos_dates"], show_progress=show_progress)
    return norm_stats, datasets


def prepare_fold_data(
    panel: pd.DataFrame,
    refit_year: int,
    val_ratio: float = VAL_RATIO,
    show_progress: bool = True,
    smoke: bool = False,
):
    fold = get_fold_dates(panel, refit_year, val_ratio=val_ratio, smoke=smoke)
    feature_cols = [c for c in panel.columns if c != LABEL_COL]

    steps = ["slice panel", "build datasets"]
    bar = tqdm(steps, desc=f"prepare {refit_year}", leave=False, disable=not show_progress)
    for step in bar:
        bar.set_postfix_str(step)
        if step == "slice panel":
            all_target_dates = fold["fit_dates"] | fold["val_dates"] | fold["oos_dates"]
            min_dt = min(all_target_dates) if all_target_dates else fold["lookback_start"]
            max_dt = max(all_target_dates) if all_target_dates else fold["oos_end"]
            hist_panel = slice_panel_by_dates(panel, min_dt - pd.DateOffset(days=60), max_dt)
        elif step == "build datasets":
            _, datasets = _build_datasets_from_panel(
                hist_panel, fold, feature_cols, show_progress=show_progress
            )
            train_ds = datasets["train"]
            val_ds = datasets["val"]
            oos_ds = datasets["oos"]

    return fold, train_ds, val_ds, oos_ds


def prepare_fold_data_from_fold(
    panel: pd.DataFrame,
    fold: dict,
    feature_cols: list,
    show_progress: bool = True,
    norm_stats=None,
    target_splits=("fit", "val", "oos"),
):
    """
    Build normalized datasets directly from a precomputed fold dict.
    `panel` must already include all dates needed for the requested splits.
    """
    steps = ["build datasets"]
    bar = tqdm(steps, desc="prepare from fold", leave=False, disable=not show_progress)
    for step in bar:
        bar.set_postfix_str(step)
        if step == "build datasets":
            norm_stats, datasets = _build_datasets_from_panel(
                panel,
                fold,
                feature_cols,
                show_progress=show_progress,
                norm_stats=norm_stats,
                target_splits=target_splits,
            )

    train_ds = datasets.get("train")
    val_ds = datasets.get("val")
    oos_ds = datasets.get("oos")
    return fold, train_ds, val_ds, oos_ds, norm_stats
