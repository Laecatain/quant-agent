from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
VALID_SIDES = {"long_short", "long_only"}


@dataclass(frozen=True)
class StrategySpec:
    top_quantile: float = 0.2
    rebalance_days: int = 1
    cost_bps: float = 0.0
    side: str = "long_short"

    def __post_init__(self) -> None:
        if not 0 < self.top_quantile <= 0.5:
            raise ValueError("top_quantile must satisfy 0 < top_quantile <= 0.5")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be >= 0")
        if self.side not in VALID_SIDES:
            raise ValueError(f"side must be one of {sorted(VALID_SIDES)}")


class BacktestEngine:
    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec

    def run(self, data: pd.DataFrame, factor: pd.Series) -> dict[str, object]:
        return backtest_factor_strategy(data=data, factor=factor, spec=self.spec)


def backtest_factor_strategy(
    data: pd.DataFrame,
    factor: pd.Series,
    spec: StrategySpec | None = None,
) -> dict[str, object]:
    strategy = spec or StrategySpec()
    frame = _build_backtest_frame(data=data, factor=factor)
    if frame.empty:
        return _empty_result()

    weights_by_date = _build_daily_weights(frame=frame, spec=strategy)
    turnover = _calculate_turnover(weights_by_date)
    daily_returns = _calculate_daily_returns(
        frame=frame,
        weights_by_date=weights_by_date,
        turnover=turnover,
        spec=strategy,
    )
    equity_curve = (1.0 + daily_returns).cumprod().rename("equity")
    positions = _format_positions(weights_by_date=weights_by_date)
    metrics = _calculate_metrics(daily_returns=daily_returns, turnover=turnover)

    return {
        "daily_returns": daily_returns,
        "equity_curve": equity_curve,
        "positions": positions,
        "turnover": turnover.reindex(daily_returns.index),
        "metrics": metrics,
    }


def _build_backtest_frame(data: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    required_cols = {"date", "code", "close"}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"data missing required columns: {sorted(missing_cols)}")
    if not isinstance(factor, pd.Series):
        raise TypeError(f"factor must be pandas.Series, got {type(factor).__name__}")

    frame = data[["date", "code", "close"]].copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["factor"] = pd.to_numeric(factor.reindex(data.index), errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["date", "code", "close"])
    frame = frame.sort_values(["code", "date"], kind="mergesort")
    frame["forward_return"] = frame.groupby("code", sort=False)["close"].shift(-1) / frame["close"] - 1.0
    return frame[["date", "code", "close", "factor", "forward_return"]]


def _build_daily_weights(frame: pd.DataFrame, spec: StrategySpec) -> dict[pd.Timestamp, pd.Series]:
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype="float64")
    dates = list(pd.Index(frame["date"].drop_duplicates()).sort_values())

    for date_index, date in enumerate(dates):
        day = frame.loc[frame["date"] == date, ["code", "factor"]]
        if date_index % spec.rebalance_days == 0 or current_weights.empty:
            current_weights = _weights_for_rebalance_day(day=day, spec=spec)
        weights_by_date[pd.Timestamp(date)] = current_weights.copy(deep=True)

    return weights_by_date


def _weights_for_rebalance_day(day: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    ranked = day.dropna(subset=["factor"]).sort_values("factor", kind="mergesort")
    if ranked.empty:
        return pd.Series(dtype="float64")

    count = max(1, int(np.floor(len(ranked) * spec.top_quantile)))
    count = min(count, len(ranked))
    long_codes = ranked.tail(count)["code"].to_list()

    if spec.side == "long_only":
        return pd.Series(
            1.0 / len(long_codes),
            index=pd.Index(long_codes, name="code"),
            dtype="float64",
        )

    short_codes = ranked.head(count)["code"].to_list()
    weights = pd.concat(
        [
            pd.Series(1.0 / len(long_codes), index=pd.Index(long_codes, name="code"), dtype="float64"),
            pd.Series(-1.0 / len(short_codes), index=pd.Index(short_codes, name="code"), dtype="float64"),
        ]
    )
    return weights.groupby(level=0, sort=True).sum()


def _calculate_turnover(weights_by_date: dict[pd.Timestamp, pd.Series]) -> pd.Series:
    turnover_values: dict[pd.Timestamp, float] = {}
    previous = pd.Series(dtype="float64")

    for date, weights in weights_by_date.items():
        aligned_index = previous.index.union(weights.index)
        previous_weights = previous.reindex(aligned_index, fill_value=0.0)
        current_weights = weights.reindex(aligned_index, fill_value=0.0)
        turnover_values[date] = float((current_weights - previous_weights).abs().sum())
        previous = weights.copy(deep=True)

    return pd.Series(turnover_values, dtype="float64", name="turnover")


def _calculate_daily_returns(
    frame: pd.DataFrame,
    weights_by_date: dict[pd.Timestamp, pd.Series],
    turnover: pd.Series,
    spec: StrategySpec,
) -> pd.Series:
    return_values: dict[pd.Timestamp, float] = {}

    for date, weights in weights_by_date.items():
        if weights.empty:
            continue

        day = frame.loc[frame["date"] == date, ["code", "forward_return"]].dropna(subset=["forward_return"])
        if day.empty:
            continue

        returns = day.set_index("code")["forward_return"]
        aligned_weights = weights.reindex(returns.index, fill_value=0.0)
        gross_return = float((aligned_weights * returns).sum())
        cost = float(turnover.loc[date] * spec.cost_bps / 10_000.0)
        return_values[pd.Timestamp(date)] = gross_return - cost

    return pd.Series(return_values, dtype="float64", name="daily_return")


def _format_positions(weights_by_date: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    records: list[tuple[pd.Timestamp, object, float]] = []
    for date, weights in weights_by_date.items():
        for code, weight in weights.items():
            if weight != 0:
                records.append((date, code, float(weight)))

    index = pd.MultiIndex.from_tuples([], names=["date", "code"])
    if not records:
        return pd.DataFrame({"weight": pd.Series(dtype="float64")}, index=index)

    positions = pd.DataFrame(records, columns=["date", "code", "weight"])
    positions = positions.sort_values("date", kind="mergesort")
    return positions.set_index(["date", "code"])[["weight"]]


def _calculate_metrics(daily_returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    clean_returns = pd.to_numeric(daily_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean_returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": 0.0,
            "average_turnover": _safe_mean(turnover),
        }

    equity = (1.0 + clean_returns).cumprod()
    final_equity = equity.iloc[-1]
    total_return = float(final_equity - 1.0)
    annualized_return = (
        float(final_equity ** (TRADING_DAYS_PER_YEAR / len(clean_returns)) - 1.0)
        if np.isfinite(final_equity) and final_equity > 0.0
        else float("nan")
    )
    std = clean_returns.std(ddof=1)
    sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * clean_returns.mean() / std) if std and not np.isnan(std) else float("nan")
    drawdown = equity / equity.cummax() - 1.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "average_turnover": _safe_mean(turnover),
    }


def _safe_mean(values: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean_values.empty:
        return 0.0
    return float(clean_values.mean())


def _empty_result() -> dict[str, object]:
    empty_returns = pd.Series(dtype="float64", name="daily_return")
    empty_turnover = pd.Series(dtype="float64", name="turnover")
    empty_positions = pd.DataFrame(
        {"weight": pd.Series(dtype="float64")},
        index=pd.MultiIndex.from_tuples([], names=["date", "code"]),
    )
    return {
        "daily_returns": empty_returns,
        "equity_curve": pd.Series(dtype="float64", name="equity"),
        "positions": empty_positions,
        "turnover": empty_turnover,
        "metrics": _calculate_metrics(empty_returns, empty_turnover),
    }
