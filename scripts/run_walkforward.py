"""Walk-forward training for daily Alpha158 MASTER."""
import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm.auto import tqdm
import numpy as np
import pyarrow.parquet as pq

from config import (
    BETA, D_FEAT, D_MODEL, DROPOUT, EARLY_STOP_METRIC, EARLY_STOP_PATIENCE,
    GATE_INPUT_END, GATE_INPUT_START,
    LR, MASTER_PANEL_PARQUET, N_EPOCH, OUTPUT_DIR, REFIT_YEARS,
    OOS_YEARS, TRAIN_YEARS, VAL_RATIO, LABEL_COL,
    S_NHEAD, T_NHEAD, TRAIN_STOP_LOSS_THRED,
)
from master import MASTERModel
from scripts.dataset_utils import prepare_fold_data_from_fold


def load_panel_slice(parquet_path, date_start, date_end):
    # Load only the needed datetime-range slice from parquet to avoid OOM.
    table = pq.read_table(
        str(parquet_path),
        filters=[
            ("datetime", ">=", pd.Timestamp(date_start)),
            ("datetime", "<=", pd.Timestamp(date_end)),
        ],
    )
    df = table.to_pandas()

    # pyarrow may reconstruct datetime/instrument as MultiIndex levels.
    if isinstance(df.index, pd.MultiIndex) and set(df.index.names) >= {"datetime", "instrument"}:
        dt = df.index.get_level_values("datetime")
        inst = df.index.get_level_values("instrument").astype(str)
        df.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
        panel = df.sort_index()
    elif "instrument" in df.columns:
        df["instrument"] = df["instrument"].astype(str)
        panel = df.set_index(["datetime", "instrument"]).sort_index()
    else:
        raise KeyError("Could not find instrument/datetime as column or MultiIndex level.")

    # Reduce memory: features/label columns are numeric.
    panel = panel.astype(np.float32, copy=False)
    return panel


def load_unique_trading_dates(parquet_path, date_start, date_end):
    # Read only datetime column, then unique() to get trading dates.
    table = pq.read_table(
        str(parquet_path),
        columns=["datetime"],
        filters=[
            ("datetime", ">=", pd.Timestamp(date_start)),
            ("datetime", "<=", pd.Timestamp(date_end)),
        ],
    )
    dt = table.column("datetime").to_pandas()
    # dt is repeated for each instrument; unique() compresses to trading calendar.
    dt_unique = pd.Series(dt.unique()).sort_values().tolist()
    # Convert to pandas Timestamps for downstream comparisons.
    return [pd.Timestamp(x) for x in dt_unique]


def compute_fold_from_trading_dates(trading_dates, refit_year, smoke: bool = False):
    train_end = pd.Timestamp(f"{refit_year}-01-01")
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    oos_end = train_end + pd.DateOffset(years=OOS_YEARS) - pd.DateOffset(days=1)

    train_dates = [d for d in trading_dates if train_start <= d <= (train_end - pd.DateOffset(days=1))]
    oos_dates = [d for d in trading_dates if train_end <= d <= oos_end]

    n_val = max(1, int(len(train_dates) * VAL_RATIO))
    val_sorted = train_dates[-n_val:]
    fit_sorted = train_dates[:-n_val]

    fit_dates = set(fit_sorted)
    val_dates = set(val_sorted)
    oos_dates_set = set(oos_dates)

    if smoke:
        smoke_max_fit_dates = 60
        smoke_max_val_dates = 20
        smoke_max_oos_dates = 20
        fit_dates = set(fit_sorted[-smoke_max_fit_dates:]) if fit_sorted else set()
        val_dates = set(val_sorted[-smoke_max_val_dates:]) if val_sorted else set()
        oos_dates_set = set(sorted(oos_dates)[-smoke_max_oos_dates:]) if oos_dates else set()

    lookback_start = train_start - pd.DateOffset(days=90)
    return {
        "train_start": train_start,
        "train_end": train_end - pd.DateOffset(days=1),
        "oos_start": train_end,
        "oos_end": oos_end,
        "fit_dates": fit_dates,
        "val_dates": val_dates,
        "oos_dates": oos_dates_set,
        "lookback_start": lookback_start,
    }


def run_fold(parquet_path, refit_year, seed=0, n_epoch=N_EPOCH, gpu=0, smoke: bool = False):
    train_end = pd.Timestamp(f"{refit_year}-01-01")
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    oos_end = train_end + pd.DateOffset(years=OOS_YEARS) - pd.DateOffset(days=1)

    tqdm.write(f"\n[fold {refit_year}] Computing fold dates (datetime only) ...")
    trading_dates = load_unique_trading_dates(parquet_path, train_start, oos_end)
    fold = compute_fold_from_trading_dates(trading_dates, refit_year, smoke=smoke)

    all_target_dates = fold["fit_dates"] | fold["val_dates"] | fold["oos_dates"]
    train_target_dates = fold["fit_dates"] | fold["val_dates"]
    train_min_dt = min(train_target_dates) if train_target_dates else fold["train_start"]
    train_max_dt = max(train_target_dates) if train_target_dates else fold["train_end"]
    oos_min_dt = min(fold["oos_dates"]) if fold["oos_dates"] else fold["oos_start"]
    oos_max_dt = max(fold["oos_dates"]) if fold["oos_dates"] else fold["oos_end"]

    train_hist_start = train_min_dt - pd.DateOffset(days=60)
    train_hist_end = train_max_dt
    tqdm.write(
        f"[fold {refit_year}] Loading train/val slice: "
        f"{train_hist_start.date()} -> {train_hist_end.date()} ..."
    )
    panel_train = load_panel_slice(parquet_path, train_hist_start, train_hist_end)

    feature_cols = [c for c in panel_train.columns if c != LABEL_COL]
    fold, dl_train, dl_valid, _, norm_stats = prepare_fold_data_from_fold(
        panel_train,
        fold,
        feature_cols,
        show_progress=True,
        target_splits=("fit", "val"),
    )
    del panel_train
    gc.collect()

    oos_hist_start = oos_min_dt - pd.DateOffset(days=60)
    oos_hist_end = oos_max_dt
    tqdm.write(
        f"[fold {refit_year}] Loading OOS slice: "
        f"{oos_hist_start.date()} -> {oos_hist_end.date()} ..."
    )
    panel_oos = load_panel_slice(parquet_path, oos_hist_start, oos_hist_end)
    _, _, _, dl_test, _ = prepare_fold_data_from_fold(
        panel_oos,
        fold,
        feature_cols,
        show_progress=True,
        norm_stats=norm_stats,
        target_splits=("oos",),
    )
    del panel_oos
    gc.collect()

    model = MASTERModel(
        d_feat=D_FEAT,
        d_model=D_MODEL,
        t_nhead=T_NHEAD,
        s_nhead=S_NHEAD,
        T_dropout_rate=DROPOUT,
        S_dropout_rate=DROPOUT,
        beta=BETA,
        gate_input_start_index=GATE_INPUT_START,
        gate_input_end_index=GATE_INPUT_END,
        n_epochs=n_epoch,
        lr=LR,
        GPU=gpu,
        seed=seed,
        train_stop_loss_thred=TRAIN_STOP_LOSS_THRED,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_metric=EARLY_STOP_METRIC,
        save_path=str(OUTPUT_DIR / "models"),
        save_prefix=f"us_{refit_year}_",
    )

    tqdm.write(
        f"\n=== Refit {refit_year}: train={len(dl_train)} val={len(dl_valid)} oos={len(dl_test)} ==="
    )
    model.fit(dl_train, dl_valid, show_progress=True)
    predictions, metrics = model.predict(dl_test, show_progress=True)
    tqdm.write(f"OOS metrics: {metrics}")
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-years", nargs="*", type=int, default=REFIT_YEARS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)

    n_epoch = 2 if args.smoke else N_EPOCH

    if torch.cuda.is_available():
        tqdm.write(f"Using GPU: {torch.cuda.get_device_name(args.gpu)} (cuda:{args.gpu})")
    else:
        tqdm.write("WARNING: CUDA not available, training on CPU.")

    tqdm.write(
        f"Folds: {args.refit_years} | max_epochs={n_epoch} | "
        f"early_stop: {EARLY_STOP_METRIC} patience={EARLY_STOP_PATIENCE}"
    )

    all_preds, metrics_rows = [], []
    refit_years = args.refit_years[:1] if args.smoke else args.refit_years
    for year in tqdm(refit_years, desc="walk-forward folds"):
        preds, metrics = run_fold(
            MASTER_PANEL_PARQUET,
            year,
            seed=args.seed,
            n_epoch=n_epoch,
            gpu=args.gpu,
            smoke=args.smoke,
        )
        preds = preds.to_frame("pred")
        preds["refit_year"] = year
        all_preds.append(preds)
        metrics_rows.append({"refit_year": year, **metrics})

        pred_df = pd.concat(all_preds)
        pred_df.to_parquet(OUTPUT_DIR / "predictions.parquet")
        pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
        tqdm.write(f"  [fold {year} checkpoint saved]")

    metrics_df = pd.DataFrame(metrics_rows)
    tqdm.write("\n=== Per-fold OOS ===")
    tqdm.write(str(metrics_df))
    tqdm.write("\n=== Mean OOS ===")
    tqdm.write(str(metrics_df.mean(numeric_only=True)))


if __name__ == "__main__":
    main()
