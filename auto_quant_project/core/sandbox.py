"""
LLM 因子代码本地执行沙盒 MVP。

设计目标：
- 接收一段由 LLM 生成的 Python 字符串代码；
- 使用 exec() 在受控局部命名空间中执行；
- 成功时返回因子 Series，失败时返回完整 traceback 字符串，供 LLM 反思修正。

约定：
LLM 生成代码必须在最后显式赋值变量 `factor`，例如：

    factor = data.groupby('code')['close'].pct_change(5)

其中 data 是传入的行情 DataFrame，包含字段：
['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SandboxResult:
    """
    沙盒执行结果。

    Attributes:
        success: 是否执行成功。
        factor: 成功时返回的因子值，必须是 pandas.Series；失败时为 None。
        error: 失败时返回 traceback 字符串；成功时为 None。
    """

    success: bool
    factor: pd.Series | None = None
    error: str | None = None


def run_factor_code(code: str, data: pd.DataFrame) -> SandboxResult:
    """
    执行 LLM 生成的因子代码，并返回因子 Series 或错误信息。

    计算数学视角：因子 f_{t,i} 是定义在“日期 t、股票 i”上的标量场。
    因此沙盒要求输出为一维 pandas.Series，并尽量与 data 的行索引对齐，
    后续 evaluator 才能将 f_{t,i} 与下一期收益 r_{t+1,i} 做横截面相关分析。

    Args:
        code: LLM 生成的 Python 代码字符串。代码中必须产生变量 `factor`。
        data: 标准化行情数据 DataFrame。

    Returns:
        SandboxResult: success=True 时包含 factor；success=False 时包含 traceback。
    """
    # 传入副本，避免因子代码原地污染主数据。
    safe_data = data.copy(deep=True)

    # 只暴露量化因子常用对象，降低执行环境的不确定性。
    # 注意：这是 MVP 级沙盒，并非强安全隔离；生产环境应使用独立进程/容器和资源限制。
    global_namespace: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "bool": bool,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "range": range,
            "round": round,
            "sum": sum,
            "tuple": tuple,
        },
        "np": np,
        "pd": pd,
    }
    local_namespace: dict[str, Any] = {"data": safe_data}

    try:
        exec(code, global_namespace, local_namespace)

        if "factor" not in local_namespace:
            raise ValueError("因子代码未生成变量 `factor`。请在代码末尾赋值：factor = ...")

        factor = local_namespace["factor"]
        if not isinstance(factor, pd.Series):
            raise TypeError(f"变量 `factor` 必须是 pandas.Series，当前类型为：{type(factor).__name__}")

        # 统一转换为数值型。无法转成数值的异常输出会变成 NaN，由 evaluator 统一清理。
        factor = pd.to_numeric(factor, errors="coerce")

        # reindex 要求 factor.index 是唯一索引。若 LLM 代码通过 groupby/rolling/reset_index 等操作
        # 生成了重复索引，直接 reindex 会触发晦涩报错，甚至造成因子与行情错配风险。
        # 因此这里先显式检查，并给出可被下一轮 LLM 反思利用的清晰错误。
        if factor.index.has_duplicates:
            duplicated_examples = factor.index[factor.index.duplicated()].unique().tolist()[:5]
            raise ValueError(
                "因子 factor 的索引存在重复值，无法安全 reindex 到 data.index。"
                "请确保 factor 是与原始 data 行一一对应的 pandas.Series；"
                "若使用宽表/pivot/rolling，请在最后 stack 后按 ['date', 'code'] 映射回原始 data.index。"
                f"重复索引示例：{duplicated_examples}"
            )

        # 统一命名，便于落盘或拼接评估结果。
        # 若 LLM 代码经过 sort/groupby 后改变了索引顺序，reindex 会强制恢复到原始 data 行索引，
        # 避免 evaluator 将 f_{t,i} 错配到其他股票或日期。
        factor = factor.reindex(data.index).rename("factor")
        return SandboxResult(success=True, factor=factor, error=None)

    except Exception:  # noqa: BLE001 - 沙盒需要捕获所有异常并反馈给 LLM。
        return SandboxResult(success=False, factor=None, error=traceback.format_exc())
