"""Alpha158 factor names (Qlib official) and MASTER project label."""
from __future__ import annotations

import pandas as pd

try:
    from qlib.contrib.data.loader import Alpha158DL
except ImportError as exc:  # pragma: no cover - build-time dependency
    raise ImportError(
        "Microsoft Qlib is required for Alpha158. "
        "Install in your training env, e.g. `pip install pyqlib` (conda env2 already has it)."
    ) from exc

_, ALPHA158_NAMES = Alpha158DL.get_feature_config()
ALPHA158_NAMES = list(ALPHA158_NAMES)


def compute_label(close: pd.Series, forward_days: int = 5) -> pd.Series:
    """MASTER paper label modified for US market (same-day execution): close.shift(-5)/close - 1."""
    return close.shift(-forward_days) / close - 1
