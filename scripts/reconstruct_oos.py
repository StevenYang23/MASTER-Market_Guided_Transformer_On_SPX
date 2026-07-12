"""Reconstruct Out-Of-Sample (OOS) predictions from saved model checkpoints."""
import gc
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import (
    BETA,
    D_FEAT,
    D_MODEL,
    DROPOUT,
    EARLY_STOP_METRIC,
    EARLY_STOP_PATIENCE,
    GATE_INPUT_END,
    GATE_INPUT_START,
    LABEL_COL,
    LR,
    MASTER_PANEL_PARQUET,
    OUTPUT_DIR,
    REFIT_YEARS,
    S_NHEAD,
    T_NHEAD,
    TRAIN_STOP_LOSS_THRED,
)
from master import MASTERModel
from scripts.dataset_utils import prepare_fold_data_from_fold
from scripts.run_walkforward import (
    compute_fold_from_trading_dates,
    load_panel_slice,
    load_unique_trading_dates,
)

SEED = 0
GPU_ID = 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_DIR / "models"

    gpu = GPU_ID if GPU_ID >= 0 and torch.cuda.is_available() else 0
    if GPU_ID >= 0 and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(gpu)} (cuda:{gpu})")
    else:
        print("Using CPU (with optimization: 8 threads)")

    all_preds = []
    metrics_rows = []

    for year in tqdm(REFIT_YEARS, desc="Reconstructing folds"):
        model_path = models_dir / f"us_{year}__{SEED}.pkl"
        if not model_path.exists():
            print(f"No checkpoint for fold {year} at {model_path}, skipping.")
            continue

        print(f"\n--- Reconstructing OOS for fold {year} ---")

        trading_dates = load_unique_trading_dates(
            MASTER_PANEL_PARQUET,
            pd.Timestamp(f"{year - 20}-01-01"),
            pd.Timestamp(f"{year + 10}-01-01"),
        )
        fold = compute_fold_from_trading_dates(trading_dates, year, smoke=False)

        train_target_dates = fold["fit_dates"] | fold["val_dates"]
        train_min_dt = min(train_target_dates) if train_target_dates else fold["train_start"]
        train_max_dt = max(train_target_dates) if train_target_dates else fold["train_end"]
        oos_min_dt = min(fold["oos_dates"]) if fold["oos_dates"] else fold["oos_start"]
        oos_max_dt = max(fold["oos_dates"]) if fold["oos_dates"] else fold["oos_end"]

        train_hist_start = train_min_dt - pd.DateOffset(days=60)
        panel_train = load_panel_slice(MASTER_PANEL_PARQUET, train_hist_start, train_max_dt)
        feature_cols = [c for c in panel_train.columns if c != LABEL_COL]
        fold, _, _, _, norm_stats = prepare_fold_data_from_fold(
            panel_train,
            fold,
            feature_cols,
            show_progress=True,
            target_splits=("fit", "val"),
        )
        del panel_train
        gc.collect()

        oos_hist_start = oos_min_dt - pd.DateOffset(days=60)
        panel_oos = load_panel_slice(MASTER_PANEL_PARQUET, oos_hist_start, oos_max_dt)
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
            n_epochs=1,
            lr=LR,
            GPU=gpu,
            seed=SEED,
            train_stop_loss_thred=TRAIN_STOP_LOSS_THRED,
            early_stop_patience=EARLY_STOP_PATIENCE,
            early_stop_metric=EARLY_STOP_METRIC,
            save_path=str(models_dir),
            save_prefix=f"us_{year}_",
        )
        model.load_param(str(model_path))
        model.fitted = 999

        predictions, metrics = model.predict(dl_test, show_progress=True, label_col=LABEL_COL)
        predictions["refit_year"] = year
        print(f"OOS metrics for fold {year}: {metrics}")

        all_preds.append(predictions)
        metrics_rows.append({"refit_year": year, **metrics})

    if not all_preds:
        print("No predictions were reconstructed.")
        return

    final_preds = pd.concat(all_preds)
    final_preds.to_parquet(OUTPUT_DIR / "predictions.parquet")
    final_metrics = pd.DataFrame(metrics_rows)
    final_metrics.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print(f"Saved predictions ({len(final_preds)} rows) -> {OUTPUT_DIR / 'predictions.parquet'}")
    print(f"Saved metrics -> {OUTPUT_DIR / 'metrics_summary.csv'}")
    print("\n=== Recompiled Metrics Summary ===")
    print(final_metrics)


if __name__ == "__main__":
    main()
