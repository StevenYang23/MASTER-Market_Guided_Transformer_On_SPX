"""Export CRSP clean frame to Qlib CSV + bin, compute official Alpha158."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LABEL_COL, LABEL_FORWARD_DAYS, PANEL_END_DATE, PANEL_START_DATE, QLIB_DATA_DIR
from scripts.alpha158 import ALPHA158_NAMES, compute_label
from scripts.qlib_dump_bin import DumpDataAll


def _qlib_csv_dir() -> Path:
    return QLIB_DATA_DIR.parent / "qlib_csv"


def prepare_crsp_for_qlib(df: pd.DataFrame) -> pd.DataFrame:
    """Add Qlib-required factor/vwap columns to cleaned CRSP frame."""
    out = df.copy()
    out["factor"] = (1.0 / out["cumfacpr"]).astype(np.float32)
    out["vwap"] = ((out["high"] + out["low"] + out["close"]) / 3.0).astype(np.float32)
    return out


def export_crsp_to_qlib_csv(df: pd.DataFrame, csv_dir: Path | None = None) -> Path:
    csv_dir = csv_dir or _qlib_csv_dir()
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    cols = ["date", "open", "high", "low", "close", "volume", "factor", "vwap"]
    for permno, g in tqdm(df.groupby("permno", sort=False), desc="export qlib csv"):
        out = g[cols].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(csv_dir / f"{int(permno)}.csv", index=False)
    return csv_dir


def dump_qlib_bin(csv_dir: Path, qlib_dir: Path | None = None, max_workers: int | None = None) -> Path:
    qlib_dir = qlib_dir or QLIB_DATA_DIR
    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)

    if max_workers is None:
        # Windows spawn + heavy imports: keep dump single-process by default.
        max_workers = 1 if os.name == "nt" else 8

    DumpDataAll(
        str(csv_dir),
        str(qlib_dir),
        include_fields="open,close,high,low,volume,factor,vwap",
        date_field_name="date",
        max_workers=max_workers,
    ).dump()
    return qlib_dir


def _init_qlib(qlib_dir: Path | None = None):
    import qlib

    qlib.init(provider_uri=str(qlib_dir or QLIB_DATA_DIR))


def compute_alpha158_panel(
    df: pd.DataFrame,
    qlib_dir: Path | None = None,
    batch_size: int = 200,
) -> pd.DataFrame:
    """
    Build Alpha158 with Qlib official expressions (Alpha158DL + QlibDataLoader).
 
    `df` must be the cleaned CRSP frame from load_clean_crsp (with cumfacpr).
    """
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D
    from qlib.data.dataset.loader import QlibDataLoader

    prepared = prepare_crsp_for_qlib(df)
    csv_dir = export_crsp_to_qlib_csv(prepared)
    qlib_dir = dump_qlib_bin(csv_dir, qlib_dir=qlib_dir)
    _init_qlib(qlib_dir)

    fields, names = Alpha158DL.get_feature_config()
    if list(names) != list(ALPHA158_NAMES):
        raise ValueError("Qlib Alpha158 factor names differ from project config.")

    loader = QlibDataLoader(config={"feature": (fields, names)})
    instruments = D.list_instruments(D.instruments("all"), as_list=True)

    parts = []
    inst_list = list(instruments)
    for i in tqdm(range(0, len(inst_list), batch_size), desc="qlib Alpha158 batches"):
        batch = inst_list[i : i + batch_size]
        batch_df = loader.load(
            instruments=batch,
            start_time=PANEL_START_DATE,
            end_time=PANEL_END_DATE,
        )
        if batch_df is None or len(batch_df) == 0:
            continue
        batch_df.columns = batch_df.columns.droplevel(0)
        batch_df = batch_df.reset_index()
        batch_df = batch_df.rename(columns={"datetime": "date", "instrument": "permno"})
        batch_df["permno"] = pd.to_numeric(batch_df["permno"], errors="coerce")
        batch_df = batch_df.dropna(subset=["permno"])
        batch_df["permno"] = batch_df["permno"].astype(int)
        parts.append(batch_df)

    if not parts:
        raise RuntimeError("Qlib Alpha158 loader returned no rows.")

    alpha = pd.concat(parts, ignore_index=True)
    alpha["date"] = pd.to_datetime(alpha["date"])

    label_parts = []
    for permno, g in prepared.groupby("permno", sort=False):
        s = g.set_index("date").sort_index()
        label_parts.append(
            compute_label(s["close"], LABEL_FORWARD_DAYS)
            .rename(LABEL_COL)
            .to_frame()
            .assign(permno=int(permno))
            .reset_index()
        )
    labels = pd.concat(label_parts, ignore_index=True)

    panel = alpha.merge(labels, on=["date", "permno"], how="inner")
    panel = panel.dropna(subset=[LABEL_COL])
    panel = panel.dropna(subset=ALPHA158_NAMES, how="all")
    panel[ALPHA158_NAMES + [LABEL_COL]] = panel[ALPHA158_NAMES + [LABEL_COL]].astype(np.float32)
    panel[ALPHA158_NAMES] = panel[ALPHA158_NAMES].fillna(0)
    return panel
