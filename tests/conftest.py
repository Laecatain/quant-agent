from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "auto_quant_project"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@pytest.fixture
def synthetic_market_data() -> pd.DataFrame:
    """Small OHLCV panel with deliberately non-contiguous, shuffled indexes."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    codes = ["AAA", "BBB", "CCC", "DDD"]

    base_prices = {
        "AAA": 10.0,
        "BBB": 20.0,
        "CCC": 30.0,
        "DDD": 40.0,
    }
    daily_steps = {
        "AAA": 0.20,
        "BBB": -0.10,
        "CCC": 0.30,
        "DDD": -0.20,
    }

    rows: list[dict[str, object]] = []
    for day_number, date in enumerate(dates):
        for code_number, code in enumerate(codes):
            close = base_prices[code] + day_number * daily_steps[code]
            open_price = close * (1.0 - 0.002 * (code_number + 1))
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "volume": 1_000_000 + day_number * 10_000 + code_number * 1_000,
                    "amount": close * (1_000_000 + day_number * 10_000 + code_number * 1_000),
                }
            )

    data = pd.DataFrame(rows)
    shuffled = data.sample(frac=1.0, random_state=42)
    shuffled.index = np.arange(100, 100 + len(shuffled) * 3, 3)
    return shuffled
