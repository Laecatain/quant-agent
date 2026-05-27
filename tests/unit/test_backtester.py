from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtester import BacktestEngine, StrategySpec, backtest_factor_strategy


REQUIRED_RESULT_KEYS = {"daily_returns", "equity_curve", "positions", "turnover", "metrics"}


def _factor_for(data: pd.DataFrame, *, day_scores: list[dict[str, float]] | None = None) -> pd.Series:
    """Build an index-aligned factor Series without assuming data is sorted."""
    default_scores = {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0}
    sorted_dates = sorted(pd.to_datetime(data["date"]).unique())
    values: dict[int, float] = {}

    for day_number, date in enumerate(sorted_dates):
        scores = day_scores[day_number] if day_scores is not None else default_scores
        same_day = data[pd.to_datetime(data["date"]) == date]
        for index, row in same_day.iterrows():
            values[index] = scores[str(row["code"])]

    return pd.Series(values, name="factor").reindex(data.index)


def _sorted_unique_dates(data: pd.DataFrame) -> list[pd.Timestamp]:
    return list(pd.Index(pd.to_datetime(data["date"]).unique()).sort_values())


def test_backtest_factor_strategy_returns_expected_result_shape(synthetic_market_data: pd.DataFrame) -> None:
    spec = StrategySpec()
    factor = _factor_for(synthetic_market_data)

    result = backtest_factor_strategy(synthetic_market_data, factor, spec)

    assert REQUIRED_RESULT_KEYS.issubset(result.keys())
    assert isinstance(result["daily_returns"], pd.Series)
    assert isinstance(result["equity_curve"], pd.Series)
    assert isinstance(result["positions"], pd.DataFrame)
    assert isinstance(result["turnover"], pd.Series)
    assert isinstance(result["metrics"], dict)
    assert not result["daily_returns"].empty
    assert result["equity_curve"].index.equals(result["daily_returns"].index)
    assert result["turnover"].index.equals(result["daily_returns"].index)
    assert set(result["positions"].index.names) == {"date", "code"}


def test_backtest_engine_class_delegates_to_same_result_contract(synthetic_market_data: pd.DataFrame) -> None:
    spec = StrategySpec(top_quantile=0.25)
    factor = _factor_for(synthetic_market_data)
    engine = BacktestEngine(spec)

    result = engine.run(synthetic_market_data, factor)

    assert REQUIRED_RESULT_KEYS.issubset(result.keys())
    assert isinstance(result["daily_returns"], pd.Series)
    assert isinstance(result["positions"], pd.DataFrame)


def test_signal_on_day_t_uses_only_t_to_t_plus_one_return_and_drops_final_day(
    synthetic_market_data: pd.DataFrame,
) -> None:
    spec = StrategySpec(top_quantile=0.25, side="long_only")
    factor = _factor_for(synthetic_market_data)
    result = backtest_factor_strategy(synthetic_market_data, factor, spec)

    daily_returns = result["daily_returns"]
    sorted_data = synthetic_market_data.sort_values(["date", "code"])
    dates = _sorted_unique_dates(synthetic_market_data)
    first_day = dates[0]
    second_day = dates[1]
    final_day = dates[-1]

    first_day_scores = factor.loc[sorted_data[sorted_data["date"] == first_day].index]
    selected_index = first_day_scores.idxmax()
    selected_code = sorted_data.loc[selected_index, "code"]
    first_close = sorted_data[(sorted_data["date"] == first_day) & (sorted_data["code"] == selected_code)]["close"].iloc[0]
    second_close = sorted_data[(sorted_data["date"] == second_day) & (sorted_data["code"] == selected_code)]["close"].iloc[0]
    expected_first_return = second_close / first_close - 1.0

    assert list(daily_returns.index) == dates[:-1]
    assert final_day not in daily_returns.index
    assert daily_returns.loc[first_day] == pytest.approx(expected_first_return)


def test_long_short_positions_are_separately_normalized_and_market_neutral(
    synthetic_market_data: pd.DataFrame,
) -> None:
    spec = StrategySpec(top_quantile=0.25, side="long_short")
    factor = _factor_for(synthetic_market_data)

    result = backtest_factor_strategy(synthetic_market_data, factor, spec)
    positions = result["positions"]["weight"].unstack("code")

    for _, weights in positions.iterrows():
        long_weight = weights[weights > 0].sum()
        short_abs_weight = weights[weights < 0].abs().sum()
        net_exposure = weights.sum()

        assert long_weight == pytest.approx(1.0)
        assert short_abs_weight == pytest.approx(1.0)
        assert net_exposure == pytest.approx(0.0, abs=1e-12)


def test_rebalance_days_two_carries_positions_on_non_rebalance_days(
    synthetic_market_data: pd.DataFrame,
) -> None:
    dates = _sorted_unique_dates(synthetic_market_data)
    day_scores = [
        {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
        {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0},
        {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0},
        {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
        {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
    ]
    spec = StrategySpec(top_quantile=0.25, rebalance_days=2, side="long_short")
    factor = _factor_for(synthetic_market_data, day_scores=day_scores)

    result = backtest_factor_strategy(synthetic_market_data, factor, spec)
    positions = result["positions"]["weight"].unstack("code")

    pd.testing.assert_series_equal(
        positions.loc[dates[1]],
        positions.loc[dates[0]],
        check_names=False,
    )
    assert not np.allclose(positions.loc[dates[2]].to_numpy(), positions.loc[dates[0]].to_numpy())


def test_cost_bps_reduces_net_returns_when_turnover_occurs(synthetic_market_data: pd.DataFrame) -> None:
    day_scores = [
        {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
        {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0},
        {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
        {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0},
        {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0},
    ]
    factor = _factor_for(synthetic_market_data, day_scores=day_scores)

    no_cost_result = backtest_factor_strategy(
        synthetic_market_data,
        factor,
        StrategySpec(top_quantile=0.25, cost_bps=0.0),
    )
    cost_result = backtest_factor_strategy(
        synthetic_market_data,
        factor,
        StrategySpec(top_quantile=0.25, cost_bps=25.0),
    )

    assert no_cost_result["turnover"].sum() > 0.0
    assert cost_result["daily_returns"].sum() < no_cost_result["daily_returns"].sum()
    assert cost_result["equity_curve"].iloc[-1] < no_cost_result["equity_curve"].iloc[-1]


def test_annualized_return_is_nan_when_final_equity_is_non_positive() -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    prices = {
        "AAA": [100.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        "BBB": [100.0, 150.0, 150.0, 150.0, 150.0, 150.0],
    }
    data = pd.DataFrame(
        [
            {"date": date, "code": code, "close": prices[code][day_number]}
            for day_number, date in enumerate(dates)
            for code in ["AAA", "BBB"]
        ]
    )
    factor = pd.Series(
        [1.0 if code == "AAA" else 0.0 for code in data["code"]],
        index=data.index,
        name="factor",
    )

    result = backtest_factor_strategy(data, factor, StrategySpec(top_quantile=0.5))

    assert result["equity_curve"].iloc[-1] == pytest.approx(0.0)
    assert np.isnan(result["metrics"]["annualized_return"])


def test_inputs_are_not_modified_in_place(synthetic_market_data: pd.DataFrame) -> None:
    spec = StrategySpec(top_quantile=0.25, cost_bps=10.0)
    data_before = synthetic_market_data.copy(deep=True)
    factor = _factor_for(synthetic_market_data)
    factor_before = factor.copy(deep=True)

    backtest_factor_strategy(synthetic_market_data, factor, spec)

    pd.testing.assert_frame_equal(synthetic_market_data, data_before)
    pd.testing.assert_series_equal(factor, factor_before)
