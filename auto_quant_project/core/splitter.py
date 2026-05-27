"""按日期切分训练、验证、测试数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    """时间序列数据切分结果。

    Attributes:
        train: 训练集，日期最早的一段。
        valid: 验证集，日期居中的一段。
        test: 测试集，日期最后的一段。
        train_dates: 训练集包含的唯一日期。
        valid_dates: 验证集包含的唯一日期。
        test_dates: 测试集包含的唯一日期。
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    train_dates: tuple[pd.Timestamp, ...]
    valid_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """返回仅包含 train/valid/test DataFrame 的字典，方便旧代码集成。"""
        return {"train": self.train, "valid": self.valid, "test": self.test}


SplitResult = DateSplit


def _validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    """校验切分比例。"""
    ratios = (train_ratio, valid_ratio, test_ratio)
    names = ("train_ratio", "valid_ratio", "test_ratio")

    for name, ratio in zip(names, ratios, strict=True):
        if not isinstance(ratio, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} 必须是数值，当前类型为：{type(ratio).__name__}")
        if not np.isfinite(float(ratio)):
            raise ValueError(f"{name} 必须是有限数值，当前为：{ratio}")
        if float(ratio) <= 0:
            raise ValueError(f"{name} 必须大于 0，当前为：{ratio}")

    ratio_sum = float(sum(ratios))
    if not np.isclose(ratio_sum, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"切分比例之和必须为 1.0，当前为：{ratio_sum}")

    return tuple(float(ratio) for ratio in ratios)  # type: ignore[return-value]


def _allocate_date_counts(n_dates: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """按比例分配唯一日期数量，并保证 train/valid/test 均非空。"""
    if n_dates < 3:
        raise ValueError(f"日期数量过少：至少需要 3 个唯一日期才能切分 train/valid/test，当前为 {n_dates} 个")

    ideal = np.array(ratios, dtype=float) * n_dates
    counts = np.floor(ideal).astype(int)
    min_dates_per_split = 1

    # 小样本下 floor 可能把 valid/test 压成 0，先保证每段至少一个日期。
    counts = np.maximum(counts, min_dates_per_split)

    # 若保证非空后总数超出，从日期数最多的分段借出，仍保持每段至少 1 个日期。
    while int(counts.sum()) > n_dates:
        candidates = np.where(counts > min_dates_per_split)[0]
        if len(candidates) == 0:
            raise ValueError(f"日期数量过少：无法在 {n_dates} 个日期内完成非空 train/valid/test 切分")
        idx = int(candidates[np.argmax(counts[candidates] - ideal[candidates])])
        counts[idx] -= 1

    # 若仍有剩余日期，按最大余数法分配，尽量贴近目标比例。
    fractional = ideal - np.floor(ideal)
    while int(counts.sum()) < n_dates:
        deficits = ideal - counts
        max_deficit = float(np.max(deficits))
        if max_deficit > 0:
            idx = int(np.argmax(deficits))
        else:
            idx = int(np.argmax(fractional))
        counts[idx] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def split_by_date(
    data: pd.DataFrame,
    *,
    date_col: str = "date",
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2,
    reset_index: bool = False,
) -> DateSplit:
    """按日期顺序切分 DataFrame 为 train/valid/test。

    切分原则：
    1. 以唯一日期为单位切分，同一天的全部行一定落在同一集合；
    2. 按时间升序排序，严禁随机切分；
    3. 默认比例为 train=0.6、valid=0.2、test=0.2；
    4. 每个分段至少包含 1 个唯一日期，否则抛出明确异常。

    Args:
        data: 包含日期字段的 pandas.DataFrame。
        date_col: 日期列名，默认 ``date``。
        train_ratio: 训练集日期比例，默认 0.6。
        valid_ratio: 验证集日期比例，默认 0.2。
        test_ratio: 测试集日期比例，默认 0.2。
        reset_index: 是否重置返回 DataFrame 的索引。默认保留原索引，便于因子序列对齐。

    Returns:
        DateSplit: 包含 ``train``、``valid``、``test`` 三个 DataFrame 以及对应日期元数据。

    Raises:
        TypeError: 输入不是 pandas.DataFrame，或比例不是数值。
        ValueError: 空数据、缺少/无法解析日期、比例非法、唯一日期数量过少。
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data 必须是 pandas.DataFrame，当前类型为：{type(data).__name__}")
    if data.empty:
        raise ValueError("输入数据为空，无法按日期切分")
    if date_col not in data.columns:
        raise ValueError(f"输入数据缺少日期字段：{date_col!r}")

    ratios = _validate_ratios(train_ratio, valid_ratio, test_ratio)

    parsed_dates = pd.to_datetime(data[date_col], errors="coerce")
    invalid_count = int(parsed_dates.isna().sum())
    if invalid_count > 0:
        raise ValueError(f"日期字段 {date_col!r} 包含 {invalid_count} 个缺失或无法解析的值")

    work = data.copy()
    work[date_col] = parsed_dates
    work["__date_split_key__"] = parsed_dates
    work["__date_split_order__"] = np.arange(len(work))
    work = work.sort_values(["__date_split_key__", "__date_split_order__"], kind="mergesort")

    unique_dates = tuple(pd.Index(work["__date_split_key__"].drop_duplicates()).sort_values())
    train_count, valid_count, test_count = _allocate_date_counts(len(unique_dates), ratios)

    train_dates = unique_dates[:train_count]
    valid_dates = unique_dates[train_count : train_count + valid_count]
    test_dates = unique_dates[train_count + valid_count : train_count + valid_count + test_count]

    def build_part(dates: tuple[pd.Timestamp, ...]) -> pd.DataFrame:
        part = work.loc[work["__date_split_key__"].isin(dates)].drop(
            columns=["__date_split_key__", "__date_split_order__"]
        )
        if reset_index:
            part = part.reset_index(drop=True)
        return part

    train = build_part(train_dates)
    valid = build_part(valid_dates)
    test = build_part(test_dates)

    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "切分结果存在空集合，请检查日期数量和切分比例："
            f"train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    return DateSplit(
        train=train,
        valid=valid,
        test=test,
        train_dates=tuple(pd.Timestamp(date) for date in train_dates),
        valid_dates=tuple(pd.Timestamp(date) for date in valid_dates),
        test_dates=tuple(pd.Timestamp(date) for date in test_dates),
    )


def split_train_valid_test_by_date(data: pd.DataFrame, **kwargs: Any) -> DateSplit:
    """``split_by_date`` 的语义化别名，供 Agent/主流程集成时调用。"""
    return split_by_date(data, **kwargs)


__all__ = ["DateSplit", "SplitResult", "split_by_date", "split_train_valid_test_by_date"]
