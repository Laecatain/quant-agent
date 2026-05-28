from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import agents.factor_miner as factor_miner_module
from agents.factor_miner import FactorCandidate, FactorMiner
from core.backtester import StrategySpec


class DummyLLMClient:
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        raise AssertionError("evaluate_candidate must not call the LLM client")


class InvalidStrategyLLMClient:
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return json.dumps(
            {
                "name": "bad_strategy",
                "hypothesis": "bad strategy payload",
                "code": "factor = data['close']",
                "lookback_days": 1,
                "expected_direction": "positive",
                "strategy": {
                    "top_quantile": 0.35,
                    "rebalance_days": 2,
                    "cost_bps": 0,
                    "side": "short_only",
                },
            }
        )


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


def _valid_cross_sectional_candidate(strategy: StrategySpec | None = None) -> FactorCandidate:
    return FactorCandidate(
        name="volume_rank_factor",
        hypothesis="Higher same-day volume rank may predict next-day cross-sectional returns.",
        code=(
            "df = data.copy()\n"
            "factor = df.groupby('date')['volume'].rank(pct=True).reindex(data.index)"
        ),
        lookback_days=1,
        expected_direction="positive",
        strategy=strategy or StrategySpec(),
    )


def _valid_payload() -> dict[str, Any]:
    return {
        "name": "volume_rank_factor",
        "hypothesis": "Higher same-day volume rank may predict next-day cross-sectional returns.",
        "code": "factor = data['volume'].rank(pct=True)",
        "lookback_days": 1,
        "expected_direction": "positive",
    }


def test_candidate_from_payload_defaults_to_default_strategy() -> None:
    candidate = FactorMiner._candidate_from_payload(_valid_payload())

    assert candidate.strategy == StrategySpec()


def test_candidate_from_payload_parses_strategy_spec() -> None:
    payload = _valid_payload() | {
        "strategy": {
            "top_quantile": 0.25,
            "rebalance_days": 5,
            "cost_bps": 8.0,
            "side": "long_only",
        }
    }

    candidate = FactorMiner._candidate_from_payload(payload)

    assert candidate.strategy == StrategySpec(
        top_quantile=0.25,
        rebalance_days=5,
        cost_bps=8.0,
        side="long_only",
    )


@pytest.mark.parametrize(
    "strategy_payload",
    [
        [],
        {"top_quantile": 0.0, "rebalance_days": 5, "cost_bps": 0.0, "side": "long_short"},
        {"top_quantile": 0.35, "rebalance_days": 5, "cost_bps": 0.0, "side": "long_short"},
        {"top_quantile": 0.2, "rebalance_days": 2, "cost_bps": 0.0, "side": "long_short"},
        {"top_quantile": 0.2, "rebalance_days": 3.9, "cost_bps": 0.0, "side": "long_short"},
        {"top_quantile": 0.2, "rebalance_days": True, "cost_bps": 0.0, "side": "long_short"},
        {"top_quantile": 0.2, "rebalance_days": 5, "cost_bps": -1.0, "side": "long_short"},
        {"top_quantile": 0.2, "rebalance_days": 5, "cost_bps": 0.0, "side": "short_only"},
    ],
)
def test_candidate_from_payload_rejects_invalid_strategy(strategy_payload: object) -> None:
    payload = _valid_payload() | {"strategy": strategy_payload}

    with pytest.raises(ValueError, match="strategy|策略|top_quantile|rebalance_days|cost_bps|side"):
        FactorMiner._candidate_from_payload(payload)


def test_invalid_generated_strategy_records_failed_trial_and_continues(tmp_path: Path) -> None:
    miner = FactorMiner(
        llm_client=InvalidStrategyLLMClient(),
        data=_six_day_four_stock_market_data(),
        factors_pool_dir=tmp_path,
    )

    trials = miner.run(generations=1)

    assert len(trials) == 1
    assert trials[0].sandbox_success is False
    assert trials[0].candidate.name == "generation_error"
    assert "strategy" in (trials[0].error or "")


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
        assert split_backtest["strategy"] == {
            "top_quantile": 0.2,
            "rebalance_days": 1,
            "cost_bps": 0.0,
            "side": "long_short",
        }
        assert isinstance(split_backtest["metrics"], dict)
        assert isinstance(split_backtest["daily_return_count"], int)
        assert "final_equity" in split_backtest
        assert "average_turnover" in split_backtest

    assert "score_breakdown" in trial.metrics
    assert trial.score == pytest.approx(trial.metrics["score_breakdown"]["final_score"])


def test_candidate_strategy_is_passed_to_each_backtest_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_specs: list[StrategySpec] = []

    def capture_backtest(*args: object, **kwargs: object) -> dict[str, object]:
        spec = kwargs.get("spec")
        assert isinstance(spec, StrategySpec)
        captured_specs.append(spec)
        index = pd.Index([pd.Timestamp("2024-01-02")], name="date")
        return {
            "daily_returns": pd.Series([0.01], index=index, name="daily_return"),
            "equity_curve": pd.Series([1.01], index=index, name="equity"),
            "positions": pd.DataFrame(),
            "turnover": pd.Series([0.4], index=index, name="turnover"),
            "metrics": {
                "total_return": 0.01,
                "annualized_return": 0.12,
                "sharpe": 1.5,
                "max_drawdown": -0.02,
                "average_turnover": 0.4,
            },
        }

    monkeypatch.setattr(factor_miner_module, "backtest_factor_strategy", capture_backtest)
    strategy = StrategySpec(top_quantile=0.25, rebalance_days=5, cost_bps=8.0, side="long_only")
    miner = _miner(tmp_path)
    candidate = _valid_cross_sectional_candidate(strategy=strategy)

    trial = miner.evaluate_candidate(candidate=candidate, generation=1)

    assert trial.sandbox_success is True
    assert captured_specs == [strategy, strategy, strategy]


def test_score_metrics_receives_backtest_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def score_after_backtest(metrics_by_split: dict[str, object], **kwargs: object) -> dict[str, object]:
        assert "backtest" in metrics_by_split
        assert kwargs["backtest_weight"] == pytest.approx(1.0)
        return {
            "raw_score": 42.0,
            "overfit_penalty": 0.0,
            "quality_penalty": 0.0,
            "final_score": 42.0,
            "details": {},
        }

    monkeypatch.setattr(factor_miner_module, "score_metrics", score_after_backtest)
    miner = _miner(tmp_path)
    candidate = _valid_cross_sectional_candidate()

    trial = miner.evaluate_candidate(candidate=candidate, generation=1)

    assert trial.sandbox_success is True
    assert trial.score == pytest.approx(42.0)


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
