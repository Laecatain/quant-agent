"""
因子评估器 MVP。

当前阶段实现两个核心指标：
1. Rank_IC：因子值与下期收益率的横截面 Spearman 秩相关；
2. Sharpe_Ratio：基于因子分组多空组合收益的年化夏普比率。

防未来函数原则：
在 t 日形成因子 f_t，只能预测 t -> t+1 的未来收益。
因此收益率必须使用 close.shift(-1) / close - 1，不能用 t 日收益或未对齐收益。
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _groupby_apply_without_group_keys(grouped: pd.core.groupby.DataFrameGroupBy, func):
    """
    兼容 pandas 2.x 与 3.x 的 groupby.apply。

    pandas 3.x 支持 include_groups=False，可避免分组键列参与 apply；
    旧版本没有该参数。为了让干净环境更稳，这里做一次函数签名检测。
    """
    if "include_groups" in inspect.signature(grouped.apply).parameters:
        return grouped.apply(func, include_groups=False)
    return grouped.apply(func)


def _safe_spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """
    不依赖 scipy 的 Spearman 秩相关。

    Spearman 相关本质是对两个变量分别取秩后再做 Pearson 相关：
        rho = corr(rank(x), rank(y))
    这样既保留“只关心排序能力”的 Rank IC 含义，又避免 pandas 在
    method='spearman' 时隐式导入 scipy，降低干净环境依赖复杂度。
    """
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return float("nan")

    x_rank = pair["x"].rank(method="average")
    y_rank = pair["y"].rank(method="average")
    if x_rank.nunique() < 2 or y_rank.nunique() < 2:
        return float("nan")

    return float(x_rank.corr(y_rank))


def _build_eval_frame(factor: pd.Series, price_data: pd.DataFrame) -> pd.DataFrame:
    """
    构建评估用面板数据：date、code、factor、forward_return。

    计算数学解释：
    对每只股票 i，若 t 日收盘价为 P_{t,i}，则下一期收益为：
        r_{t+1,i} = P_{t+1,i} / P_{t,i} - 1
    pandas 中按股票分组后使用 shift(-1) 取 P_{t+1,i}，这正是防止未来函数的关键对齐步骤。
    """
    required_cols = {"date", "code", "close"}
    missing_cols = required_cols - set(price_data.columns)
    if missing_cols:
        raise ValueError(f"price_data 缺少必要字段：{sorted(missing_cols)}")

    if not isinstance(factor, pd.Series):
        raise TypeError(f"factor 必须是 pandas.Series，当前类型为：{type(factor).__name__}")

    df = price_data[["date", "code", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # 按股票和日期排序后计算下一交易日收益，严禁使用当日收益冒充未来收益。
    df = df.sort_values(["code", "date"]).reset_index(drop=False).rename(columns={"index": "original_index"})
    df["forward_return"] = df.groupby("code", sort=False)["close"].shift(-1) / df["close"] - 1.0

    # 因子通常由 sandbox 基于原始 data 逐行计算，索引应与 price_data 原索引一致。
    factor_aligned = factor.reindex(df["original_index"]).to_numpy()
    df["factor"] = pd.to_numeric(factor_aligned, errors="coerce")

    # 清理无法参与统计的样本：NaN、inf、最后一天无下一期收益等。
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["factor", "forward_return", "date", "code"])
    return df[["date", "code", "factor", "forward_return"]]


def calculate_rank_ic(eval_frame: pd.DataFrame) -> float:
    """
    计算平均 Rank IC。

    Rank IC 是每个交易日横截面上：
        corr_spearman(rank(factor_{t,*}), rank(r_{t+1,*}))
    的时间均值。它衡量因子排序能力，而非绝对数值尺度。
    """
    daily_ic = _groupby_apply_without_group_keys(
        eval_frame.groupby("date", sort=True),
        lambda x: _safe_spearman_corr(x["factor"], x["forward_return"]),
    )
    daily_ic = daily_ic.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_ic.empty:
        return float("nan")
    return float(daily_ic.mean())


def calculate_long_short_returns(eval_frame: pd.DataFrame, quantile: float = 0.2) -> pd.Series:
    """
    构建日频多空组合收益序列。

    逻辑：每天按因子从小到大排序，做多顶部 quantile，做空底部 quantile。
    为了适配因子方向未知的 MVP，这里默认“因子越大越看多”。
    若某因子 Rank_IC 为负，后续 Agent 可通过取负号完成方向变异。
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile 必须位于 (0, 0.5) 区间。")

    def one_day_return(day: pd.DataFrame) -> float:
        day = day.sort_values("factor")
        n = len(day)
        group_size = int(np.floor(n * quantile))
        if group_size < 1:
            return np.nan

        short_return = day.iloc[:group_size]["forward_return"].mean()
        long_return = day.iloc[-group_size:]["forward_return"].mean()
        return float(long_return - short_return)

    returns = _groupby_apply_without_group_keys(eval_frame.groupby("date", sort=True), one_day_return)
    return returns.replace([np.inf, -np.inf], np.nan).dropna().rename("long_short_return")


def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """
    计算年化夏普比率。

    数学形式：
        Sharpe = sqrt(252) * mean(R_t) / std(R_t)
    这里暂不扣无风险利率，因为日频多空组合的 MVP 评估重点是相对排序收益质量。
    """
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return float("nan")

    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")

    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / std)


def evaluate_factor(factor: pd.Series, price_data: pd.DataFrame) -> dict[str, float]:
    """
    评估因子质量，返回 Rank_IC 和 Sharpe_Ratio。

    Args:
        factor: 因子值 Series，通常来自 sandbox.run_factor_code 的输出。
        price_data: 标准行情 DataFrame，至少包含 date、code、close。

    Returns:
        dict[str, float]:
            - Rank_IC: 平均横截面秩相关；
            - Sharpe_Ratio: 因子顶部/底部 20% 多空组合年化夏普；
            - ICIR: Rank IC 的均值/标准差年化，便于调试观察；
            - Sample_Size: 实际参与评估的样本行数。

    注：虽然 Phase 1 只强制要求 Rank_IC 和 Sharpe_Ratio，额外返回 ICIR 与样本数
    有助于排查“指标看似很高但样本极少”的统计陷阱。
    """
    eval_frame = _build_eval_frame(factor=factor, price_data=price_data)
    if eval_frame.empty:
        return {
            "Rank_IC": float("nan"),
            "Sharpe_Ratio": float("nan"),
            "ICIR": float("nan"),
            "Sample_Size": 0.0,
        }

    daily_ic = _groupby_apply_without_group_keys(
        eval_frame.groupby("date", sort=True),
        lambda x: _safe_spearman_corr(x["factor"], x["forward_return"]),
    )
    daily_ic = daily_ic.replace([np.inf, -np.inf], np.nan).dropna()

    rank_ic = float(daily_ic.mean()) if not daily_ic.empty else float("nan")
    ic_std = daily_ic.std(ddof=1) if len(daily_ic) >= 2 else np.nan
    icir = float(np.sqrt(TRADING_DAYS_PER_YEAR) * rank_ic / ic_std) if ic_std and not np.isnan(ic_std) else float("nan")

    long_short_returns = calculate_long_short_returns(eval_frame)
    sharpe_ratio = calculate_sharpe_ratio(long_short_returns)

    return {
        "Rank_IC": rank_ic,
        "Sharpe_Ratio": sharpe_ratio,
        "ICIR": icir,
        "Sample_Size": float(len(eval_frame)),
    }
