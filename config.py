from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs" / "walkforward"

# Raw WRDS export (local only; gitignored)
SPX_STOCK_CSV = DATA_DIR / "spx_stock_data.csv"
CRSP_CLEAN_PARQUET = PROCESSED_DIR / "crsp_daily_clean.parquet"
QLIB_DATA_DIR = PROCESSED_DIR / "qlib_us"

MARKET_RAW_PARQUET = RAW_DIR / "us_indices_daily.parquet"
MARKET_FEATURES_CSV = RAW_DIR / "us_market_information.csv"
MASTER_PANEL_PARQUET = PROCESSED_DIR / "master_panel_long.parquet"

LABEL_COL = "label"
N_FACTORS = 159
N_MARKET = 63
STEP_LEN = 8
LABEL_FORWARD_DAYS = 5  # paper: Ref(close,-5)/Ref(close,-1)-1

D_FEAT = N_FACTORS
GATE_INPUT_START = N_FACTORS
GATE_INPUT_END = N_FACTORS + N_MARKET

INDICES = {"SPX": "^GSPC", "NDX": "^IXIC", "DJI": "^DJI"}
ROLLING_WINDOWS = [5, 10, 20, 30, 60]

# Panel build filters
PANEL_START_DATE = "1990-01-01"
PANEL_END_DATE = "2025-12-31"

REFIT_YEARS = [2000, 2010, 2020]
TRAIN_YEARS = 20
VAL_RATIO = 0.15
OOS_YEARS = 10

D_MODEL = 256
T_NHEAD = 4
S_NHEAD = 2
DROPOUT = 0.5
BETA = 5
LR = 1e-5
N_EPOCH = 40  # paper: at most 40 epochs with early stopping
TRAIN_STOP_LOSS_THRED = 0.95
EARLY_STOP_PATIENCE = 10
EARLY_STOP_METRIC = "IC"  # IC or RIC
