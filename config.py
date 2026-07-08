from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs" / "walkforward"

FACTOR_PARQUET = ROOT / "153factor.parquet"
MARKET_RAW_PARQUET = RAW_DIR / "us_indices_monthly.parquet"
MARKET_FEATURES_CSV = RAW_DIR / "us_market_information.csv"
MASTER_PANEL_PARQUET = PROCESSED_DIR / "master_panel_long.parquet"

META_COLS = [
    "obs_main", "exch_main", "common", "primary_sec", "permno", "date",
    "permco", "excntry", "eom", "me", "ret_exc_lead1m", "ret_exc", "crsp_exchcd",
]
LABEL_COL = "ret_exc_lead1m"
N_FACTORS = 153
N_MARKET = 63
STEP_LEN = 8

D_FEAT = N_FACTORS
GATE_INPUT_START = N_FACTORS
GATE_INPUT_END = N_FACTORS + N_MARKET

INDICES = {"SPX": "^GSPC", "NDX": "^IXIC", "DJI": "^DJI"}
ROLLING_WINDOWS = [5, 10, 20, 30, 60]

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
N_EPOCH = 100
TRAIN_STOP_LOSS_THRED = 0.95
EARLY_STOP_PATIENCE = 10
EARLY_STOP_METRIC = "IC"  # IC or RIC
