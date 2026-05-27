"""因子指标综合评分工具。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


DEFAULT_WEIGHTS: dict[str, float] = {
    "Rank_IC": 8.0,
    "ICIR": 0.7,
    "Sharpe_Ratio": 1.0,
}


def _safe_float(value: Any, default: float = math.nan) -> float:
    """将任意指标值安全转换为有限 float；失败、缺失、NaN、inf 返回 default。"""
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _get_split_metrics(metrics_by_split: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    """安全获取某个 split 的指标字典。"""
    value = metrics_by_split.get(split, {})
    if isinstance(value, Mapping):
        return value
    return {}


def _metric(metrics: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    """安全获取单个指标。"""
    return _safe_float(metrics.get(key), default=default)


def _bounded_metric(value: float, scale: float) -> float:
    """用 tanh 将极端指标压缩，避免单一异常值支配评分。"""
    if not math.isfinite(value):
        return 0.0
    if scale <= 0:
        raise ValueError("scale 必须大于 0")
    return float(math.tanh(value / scale))


def _split_raw_score(metrics: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    """计算单个样本段的稳健原始分。"""
    rank_ic = _metric(metrics, "Rank_IC", default=math.nan)
    icir = _metric(metrics, "ICIR", default=math.nan)
    sharpe = _metric(metrics, "Sharpe_Ratio", default=math.nan)

    score = 0.0
    score += float(weights.get("Rank_IC", 0.0)) * _bounded_metric(rank_ic, scale=0.08)
    score += float(weights.get("ICIR", 0.0)) * _bounded_metric(icir, scale=2.0)
    score += float(weights.get("Sharpe_Ratio", 0.0)) * _bounded_metric(sharpe, scale=3.0)
    return float(score)


def _nan_missing_penalty(metrics_by_split: Mapping[str, Any]) -> tuple[float, list[str]]:
    """对 valid/test 核心指标缺失或 NaN 施加惩罚。"""
    reasons: list[str] = []
    penalty = 0.0
    core_metrics = ("Rank_IC", "ICIR", "Sharpe_Ratio")

    for split in ("valid", "test"):
        split_metrics = _get_split_metrics(metrics_by_split, split)
        if not split_metrics:
            penalty += 3.0
            reasons.append(f"{split} 指标缺失")
            continue
        for key in core_metrics:
            value = _safe_float(split_metrics.get(key), default=math.nan)
            if not math.isfinite(value):
                penalty += 0.75
                reasons.append(f"{split}.{key} 缺失或 NaN")

    return float(penalty), reasons


def _sample_size_penalty(
    metrics_by_split: Mapping[str, Any],
    *,
    min_sample_size: int,
) -> tuple[float, list[str]]:
    """根据 valid/test 样本数不足情况施加渐进惩罚。"""
    reasons: list[str] = []
    penalty = 0.0

    if min_sample_size <= 0:
        return penalty, reasons

    for split in ("valid", "test"):
        split_metrics = _get_split_metrics(metrics_by_split, split)
        sample_size = _safe_float(split_metrics.get("Sample_Size"), default=math.nan)
        if not math.isfinite(sample_size):
            penalty += 0.5
            reasons.append(f"{split}.Sample_Size 缺失")
            continue
        if sample_size < min_sample_size:
            shortage_ratio = (min_sample_size - max(sample_size, 0.0)) / min_sample_size
            split_penalty = 2.0 * shortage_ratio
            penalty += split_penalty
            reasons.append(f"{split}.Sample_Size 过低：{sample_size:g} < {min_sample_size}")

    return float(penalty), reasons


def _complexity_penalty(complexity: float | None, max_complexity: float | None) -> tuple[float, list[str]]:
    """根据代码复杂度预留惩罚入口。"""
    reasons: list[str] = []
    if complexity is None or max_complexity is None or max_complexity <= 0:
        return 0.0, reasons

    value = _safe_float(complexity, default=math.nan)
    if not math.isfinite(value):
        return 0.0, reasons
    if value <= max_complexity:
        return 0.0, reasons

    penalty = min(3.0, (value - max_complexity) / max_complexity)
    reasons.append(f"复杂度过高：{value:g} > {max_complexity:g}")
    return float(penalty), reasons


def _directional_gap(train_value: float, test_value: float) -> float:
    """返回 train 明显好于 test 的非负差距，保留指标方向。"""
    if not (math.isfinite(train_value) and math.isfinite(test_value)):
        return 0.0
    return max(0.0, train_value - test_value)


def _overfit_penalty(
    metrics_by_split: Mapping[str, Any],
    *,
    threshold: float,
    penalty_weight: float,
) -> tuple[float, list[str]]:
    """train 指标明显优于 test 时扣分。"""
    reasons: list[str] = []
    train = _get_split_metrics(metrics_by_split, "train")
    test = _get_split_metrics(metrics_by_split, "test")
    if not train or not test:
        return 0.0, reasons

    rank_gap = _directional_gap(_metric(train, "Rank_IC"), _metric(test, "Rank_IC"))
    icir_gap = _directional_gap(_metric(train, "ICIR"), _metric(test, "ICIR"))
    sharpe_gap = _directional_gap(_metric(train, "Sharpe_Ratio"), _metric(test, "Sharpe_Ratio"))

    # Rank_IC 量级通常较小，先映射到与 Sharpe/ICIR 可比较的尺度。
    normalized_gap = (rank_gap / 0.05) + (icir_gap / 1.5) + (sharpe_gap / 2.0)
    if normalized_gap <= threshold:
        return 0.0, reasons

    penalty = penalty_weight * (normalized_gap - threshold)
    if rank_gap > 0:
        reasons.append(f"train Rank_IC 高于 test：gap={rank_gap:.4g}")
    if icir_gap > 0:
        reasons.append(f"train ICIR 高于 test：gap={icir_gap:.4g}")
    if sharpe_gap > 0:
        reasons.append(f"train Sharpe_Ratio 高于 test：gap={sharpe_gap:.4g}")
    return float(penalty), reasons


def score_metrics(
    metrics_by_split: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
    valid_weight: float = 0.65,
    test_weight: float = 0.35,
    overfit_threshold: float = 1.0,
    overfit_penalty_weight: float = 1.0,
    min_sample_size: int = 100,
    complexity: float | None = None,
    max_complexity: float | None = None,
) -> dict[str, Any]:
    """综合 train/valid/test 指标，返回稳健评分分解。

    评分重点：
    - 只用 valid/test 形成 ``raw_score``，避免把训练集表现直接计入收益；
    - Rank_IC、ICIR、Sharpe_Ratio 采用有界变换，降低极端值影响；
    - 若 train 明显优于 test，施加 ``overfit_penalty``；
    - 对 valid/test 样本数不足、核心指标 NaN/缺失、复杂度过高施加 ``quality_penalty``；
    - 输入指标缺失或 NaN 不会直接崩溃，会被视作 0 贡献并记录 penalty reason。

    Args:
        metrics_by_split: 形如 ``{"train": {...}, "valid": {...}, "test": {...}}`` 的指标字典。
        weights: 核心指标权重，默认见 ``DEFAULT_WEIGHTS``。
        valid_weight: valid 原始评分权重。
        test_weight: test 原始评分权重。
        overfit_threshold: 归一化 train-test 差距超过该阈值后开始扣分。
        overfit_penalty_weight: 过拟合惩罚强度。
        min_sample_size: valid/test 最低样本量建议值。
        complexity: 预留代码复杂度输入，例如 AST 节点数、表达式长度等。
        max_complexity: 复杂度阈值，超过后扣分。

    Returns:
        dict[str, Any]: 包含 raw_score、overfit_penalty、quality_penalty、final_score 及明细。
    """
    if not isinstance(metrics_by_split, Mapping):
        raise TypeError(f"metrics_by_split 必须是 dict-like，当前类型为：{type(metrics_by_split).__name__}")
    if valid_weight < 0 or test_weight < 0:
        raise ValueError("valid_weight 和 test_weight 必须非负")
    total_split_weight = valid_weight + test_weight
    if total_split_weight <= 0:
        raise ValueError("valid_weight 和 test_weight 不能同时为 0")

    metric_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    valid_metrics = _get_split_metrics(metrics_by_split, "valid")
    test_metrics = _get_split_metrics(metrics_by_split, "test")

    valid_score = _split_raw_score(valid_metrics, metric_weights)
    test_score = _split_raw_score(test_metrics, metric_weights)
    raw_score = (valid_weight * valid_score + test_weight * test_score) / total_split_weight

    overfit_penalty, overfit_reasons = _overfit_penalty(
        metrics_by_split,
        threshold=overfit_threshold,
        penalty_weight=overfit_penalty_weight,
    )

    nan_penalty, nan_reasons = _nan_missing_penalty(metrics_by_split)
    sample_penalty, sample_reasons = _sample_size_penalty(metrics_by_split, min_sample_size=min_sample_size)
    complexity_penalty_value, complexity_reasons = _complexity_penalty(complexity, max_complexity)
    quality_penalty = nan_penalty + sample_penalty + complexity_penalty_value

    final_score = raw_score - overfit_penalty - quality_penalty
    if not math.isfinite(final_score):
        final_score = float("-inf")

    return {
        "raw_score": float(raw_score),
        "overfit_penalty": float(overfit_penalty),
        "quality_penalty": float(quality_penalty),
        "final_score": float(final_score),
        "details": {
            "valid_score": float(valid_score),
            "test_score": float(test_score),
            "weights": metric_weights,
            "valid_weight": float(valid_weight),
            "test_weight": float(test_weight),
            "overfit_reasons": overfit_reasons,
            "quality_reasons": nan_reasons + sample_reasons + complexity_reasons,
        },
    }


__all__ = ["DEFAULT_WEIGHTS", "score_metrics"]
