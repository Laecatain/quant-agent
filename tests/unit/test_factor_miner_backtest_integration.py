from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import agents.factor_miner as factor_miner_module
from agents.factor_miner import FactorCandidate, FactorMiner


class DummyLLMClient:
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        raise AssertionError("evaluate_candidate must not call the LLM client")


def _six_day_four_stock_market_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    codes = ["000001", "000002", "000003", "000004"]
    base_prices = {
        "000001": 10.0,
        "000002": 20.0,
        "000003": 30.0,
        "000004": 40.0,
    }
    daily_steps = {
        "000001": 0.30,
        "000002": -0.20,
        "000003": 0.15,
        "000004": -0.10,
    }

    rows: list[dict[str, Any]] = []
    for day_number, date in enumerate(dates):
        for code_number, code in enumerate(codes):
            close = base_prices[code] + day_number * daily_steps[code] + code_number * 0.01
            open_price = close * (1.0 - 0.001 * (code_number + 1))
            volume = 1_000_000 + day_number * 20_000 + code_number * 2_500
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "volume": volume,
                    "amount": close * volume,
                }
            )

    return pd.DataFrame(rows)


def _miner(tmp_path: Path) -> FactorMiner:
    return FactorMiner(
        llm_client=DummyLLMClient(),
        data=_six_day_four_stock_market_data(),
        factors_pool_dir=tmp_path,
    )


def _valid_cross_sectional_candidate() -> FactorCandidate:
    return FactorCandidate(
        name="volume_rank_factor",
        hypothesis="Higher same-day volume rank may predict next-day cross-sectional returns.",
        code=(
            "df = data.copy()\n"
            "factor = df.groupby('date')['volume'].rank(pct=True).reindex(data.index)"
        ),
        lookback_days=1,
        expected_direction="positive",
    )


def test_successful_candidate_records_backtest_summary_without_changing_score(tmp_path: Path) -> None:
    miner = _miner(tmp_path)
    candidate = _valid_cross_sectional_candidate()

    trial = miner.evaluate_candidate(candidate=candidate, generation=1)

    assert trial.sandbox_success is True
    assert trial.metrics is not None
    assert "backtest" in trial.metrics
    assert set(trial.metrics["backtest"]) == {"train", "valid", "test"}

    for split_name in ("train", "valid", "test"):
        split_backtest = trial.metrics["backtest"][split_name]
        assert isinstance(split_backtest["metrics"], dict)
        assert isinstance(split_backtest["daily_return_count"], int)
        assert "final_equity" in split_backtest
        assert "average_turnover" in split_backtest

    assert "score_breakdown" in trial.metrics
    assert trial.score == pytest.approx(trial.metrics["score_breakdown"]["final_score"])


def test_backtest_failure_is_recorded_without_marking_sandbox_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_backtest_error(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic backtest failure")

    monkeypatch.setattr(
        factor_miner_module,
        "backtest_factor_strategy",
        raise_backtest_error,
        raising=False,
    )
    miner = _miner(tmp_path)
    candidate = _valid_cross_sectional_candidate()

    trial = miner.evaluate_candidate(candidate=candidate, generation=1)

    assert trial.sandbox_success is True
    assert trial.metrics is not None
    assert "backtest_error" in trial.metrics
    assert "synthetic backtest failure" in trial.metrics["backtest_error"]
    assert "score_breakdown" in trial.metrics
