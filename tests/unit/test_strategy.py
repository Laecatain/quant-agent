from __future__ import annotations

import pytest

from agents.factor_miner import FactorCandidate
from core.backtester import StrategySpec


class TestStrategySpecDefaults:
    def test_defaults_describe_minimal_long_short_strategy(self) -> None:
        spec = StrategySpec()

        assert spec.top_quantile == pytest.approx(0.2)
        assert spec.rebalance_days == 1
        assert spec.cost_bps == pytest.approx(0.0)
        assert spec.side == "long_short"

    @pytest.mark.parametrize("top_quantile", [0.01, 0.2, 0.5])
    def test_accepts_valid_top_quantile(self, top_quantile: float) -> None:
        spec = StrategySpec(top_quantile=top_quantile)

        assert spec.top_quantile == pytest.approx(top_quantile)

    @pytest.mark.parametrize("top_quantile", [0.0, -0.1, 0.5001, 1.0])
    def test_rejects_top_quantile_outside_open_zero_closed_half(self, top_quantile: float) -> None:
        with pytest.raises(ValueError, match="top_quantile"):
            StrategySpec(top_quantile=top_quantile)

    @pytest.mark.parametrize("rebalance_days", [1, 2, 5])
    def test_accepts_positive_rebalance_days(self, rebalance_days: int) -> None:
        spec = StrategySpec(rebalance_days=rebalance_days)

        assert spec.rebalance_days == rebalance_days

    @pytest.mark.parametrize("rebalance_days", [0, -1])
    def test_rejects_rebalance_days_less_than_one(self, rebalance_days: int) -> None:
        with pytest.raises(ValueError, match="rebalance_days"):
            StrategySpec(rebalance_days=rebalance_days)

    @pytest.mark.parametrize("cost_bps", [0.0, 1.5, 25.0])
    def test_accepts_non_negative_cost_bps(self, cost_bps: float) -> None:
        spec = StrategySpec(cost_bps=cost_bps)

        assert spec.cost_bps == pytest.approx(cost_bps)

    def test_rejects_negative_cost_bps(self) -> None:
        with pytest.raises(ValueError, match="cost_bps"):
            StrategySpec(cost_bps=-0.01)

    @pytest.mark.parametrize("side", ["long_short", "long_only"])
    def test_accepts_supported_strategy_sides(self, side: str) -> None:
        spec = StrategySpec(side=side)

        assert spec.side == side

    @pytest.mark.parametrize("side", ["short_only", "market_neutral", "", "LONG_ONLY"])
    def test_rejects_unsupported_strategy_sides(self, side: str) -> None:
        with pytest.raises(ValueError, match="side"):
            StrategySpec(side=side)


class TestFactorCandidateStrategy:
    def test_defaults_to_default_strategy(self) -> None:
        candidate = FactorCandidate(
            name="factor",
            hypothesis="hypothesis",
            code="factor = data['close']",
            lookback_days=1,
            expected_direction="positive",
        )

        assert candidate.strategy == StrategySpec()

    def test_accepts_explicit_strategy(self) -> None:
        strategy = StrategySpec(
            top_quantile=0.25,
            rebalance_days=3,
            cost_bps=12.5,
            side="long_only",
        )

        candidate = FactorCandidate(
            name="factor",
            hypothesis="hypothesis",
            code="factor = data['close']",
            lookback_days=1,
            expected_direction="positive",
            strategy=strategy,
        )

        assert candidate.strategy == strategy
