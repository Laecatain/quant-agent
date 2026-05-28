from __future__ import annotations

import copy

import pytest

from core.scoring import score_metrics


def _base_metrics() -> dict[str, object]:
    return {
        "train": {
            "Rank_IC": 0.04,
            "ICIR": 1.2,
            "Sharpe_Ratio": 1.5,
            "Sample_Size": 500,
        },
        "valid": {
            "Rank_IC": 0.035,
            "ICIR": 1.1,
            "Sharpe_Ratio": 1.3,
            "Sample_Size": 400,
        },
        "test": {
            "Rank_IC": 0.03,
            "ICIR": 1.0,
            "Sharpe_Ratio": 1.1,
            "Sample_Size": 400,
        },
    }


def _with_backtest(metrics: dict[str, object], *, sharpe: float, max_drawdown: float, turnover: float) -> dict[str, object]:
    enriched = copy.deepcopy(metrics)
    enriched["backtest"] = {
        "train": {
            "metrics": {
                "total_return": 0.12,
                "annualized_return": 0.18,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "average_turnover": turnover,
            },
            "final_equity": 1.12,
            "average_turnover": turnover,
        },
        "valid": {
            "metrics": {
                "total_return": 0.10,
                "annualized_return": 0.16,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "average_turnover": turnover,
            },
            "final_equity": 1.10,
            "average_turnover": turnover,
        },
        "test": {
            "metrics": {
                "total_return": 0.08,
                "annualized_return": 0.14,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "average_turnover": turnover,
            },
            "final_equity": 1.08,
            "average_turnover": turnover,
        },
    }
    return enriched


def test_score_metrics_rewards_better_backtest_when_enabled() -> None:
    base = _base_metrics()
    weak = _with_backtest(base, sharpe=-0.5, max_drawdown=-0.35, turnover=2.5)
    strong = _with_backtest(base, sharpe=2.0, max_drawdown=-0.05, turnover=0.4)

    weak_score = score_metrics(weak, backtest_weight=1.0)
    strong_score = score_metrics(strong, backtest_weight=1.0)

    assert strong_score["final_score"] > weak_score["final_score"]
    assert strong_score["details"]["backtest_score"] > weak_score["details"]["backtest_score"]
    assert "backtest_split_scores" in strong_score["details"]


def test_score_metrics_keeps_factor_only_score_when_backtest_weight_is_default() -> None:
    base = _base_metrics()
    enriched = _with_backtest(base, sharpe=2.0, max_drawdown=-0.05, turnover=0.4)

    factor_only = score_metrics(base)
    with_backtest = score_metrics(enriched)

    assert with_backtest["final_score"] == pytest.approx(factor_only["final_score"])
    assert with_backtest["details"]["backtest_score"] == pytest.approx(0.0)


def test_score_metrics_tolerates_missing_or_malformed_backtest_when_enabled() -> None:
    malformed = _base_metrics() | {"backtest": {"valid": [], "test": {"metrics": None}}}

    result = score_metrics(malformed, backtest_weight=1.0)

    assert isinstance(result["final_score"], float)
    assert result["details"]["backtest_score"] == pytest.approx(0.0)
