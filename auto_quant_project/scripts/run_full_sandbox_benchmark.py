"""
全量数据沙盒基准测试脚本。

用途：
1. 读取 data/csi300_daily.parquet；
2. 在完整沪深 300 数据上运行 3 个基准因子；
3. 输出 sandbox/evaluator 成功状态、耗时、Rank IC、Sharpe、样本数、NaN 比例、索引对齐状态。

运行：
    .venv\\Scripts\\python.exe scripts\\run_full_sandbox_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from textwrap import dedent

import pandas as pd

# 允许从项目根目录直接运行脚本时导入 core 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluator import evaluate_factor  # noqa: E402
from core.sandbox import run_factor_code  # noqa: E402


# Windows 终端中文输出兼容。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DATA_PATH = PROJECT_ROOT / "data" / "csi300_daily.parquet"


BENCHMARK_FACTORS: dict[str, str] = {
    "5日反转": dedent(
        """
        df = data.sort_values(["code", "date"]).copy()
        ret_5d = df.groupby("code", sort=False)["close"].pct_change(5)

        # 短期反转假设：过去 5 日涨幅越高，下一期越可能均值回归；因此取负号。
        factor = (-ret_5d).reindex(data.index)
        """
    ).strip(),
    "20日量价背离": dedent(
        """
        df = data[["date", "code", "close", "volume"]].copy()
        df["date"] = pd.to_datetime(df["date"])

        # 使用 pivot 宽表，令每一列是一只股票，每一行是一个交易日。
        # 宽表 rolling corr 在时间轴上向后看历史窗口，不会制造未来函数，
        # 同时避免 groupby + rolling 后出现 MultiIndex/duplicate index 问题。
        close_wide = df.pivot(index="date", columns="code", values="close").sort_index()
        volume_wide = df.pivot(index="date", columns="code", values="volume").sort_index()

        ret_20d = close_wide.pct_change(20)
        volume_chg = volume_wide.pct_change()
        ret_1d = close_wide.pct_change()

        # 计算数学解释：rolling corr 衡量过去 20 日“价变动”和“量变动”的同步性。
        # 若 20 日收益为正但量价同步性下降，可能表示上涨缺乏成交量确认。
        # 这里构造一个反向量价背离因子：收益越强且量价相关越低，得分越低。
        pv_corr = ret_1d.rolling(20, min_periods=10).corr(volume_chg)
        factor_wide = -(ret_20d * (1.0 - pv_corr))

        # 从宽表还原成长表，再按 ['date', 'code'] 映射回原始 data.index，确保索引唯一且对齐。
        factor_long = factor_wide.stack(future_stack=True).rename("factor").reset_index()
        factor_map = factor_long.set_index(["date", "code"])["factor"]
        row_key = pd.MultiIndex.from_frame(df[["date", "code"]])
        factor = pd.Series(factor_map.reindex(row_key).to_numpy(), index=data.index)
        """
    ).strip(),
    "10日低波动率": dedent(
        """
        df = data.sort_values(["code", "date"]).copy()
        ret_1d = df.groupby("code", sort=False)["close"].pct_change()
        volatility_10d = ret_1d.groupby(df["code"], sort=False).rolling(10, min_periods=5).std()
        volatility_10d = volatility_10d.reset_index(level=0, drop=True)

        # 低波动假设：短期波动率越高，下一期风险补偿未必更好；取负号偏好低波动。
        factor = (-volatility_10d).reindex(data.index)
        """
    ).strip(),
}


def parse_args() -> argparse.Namespace:
    """解析 benchmark 参数。"""
    parser = argparse.ArgumentParser(description="运行全量沙盒基准测试")
    parser.add_argument("--limit-codes", type=int, default=0, help="调试用：只取前 N 只股票，0 表示全量")
    parser.add_argument("--factor", type=str, default="", help="调试用：只运行指定因子名称")
    return parser.parse_args()


def load_data(limit_codes: int = 0) -> pd.DataFrame:
    """读取全量 parquet 行情数据。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在：{DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    data["code"] = data["code"].astype(str).str.zfill(6)

    if limit_codes > 0:
        keep_codes = sorted(data["code"].unique())[:limit_codes]
        data = data[data["code"].isin(keep_codes)].reset_index(drop=True)

    return data


def run_one_factor(name: str, code: str, data: pd.DataFrame) -> dict[str, object]:
    """运行单个 benchmark 因子。"""
    started = time.perf_counter()
    sandbox_result = run_factor_code(code=code, data=data)
    elapsed = time.perf_counter() - started

    result: dict[str, object] = {
        "factor": name,
        "sandbox_success": sandbox_result.success,
        "elapsed_sec": round(elapsed, 4),
        "Rank_IC": None,
        "Sharpe_Ratio": None,
        "Sample_Size": None,
        "nan_ratio": None,
        "index_aligned": False,
        "error": None,
    }

    if not sandbox_result.success:
        result["error"] = sandbox_result.error
        return result

    assert sandbox_result.factor is not None
    factor = sandbox_result.factor
    result["nan_ratio"] = round(float(factor.isna().mean()), 6)
    result["index_aligned"] = bool(factor.index.equals(data.index))

    try:
        metrics = evaluate_factor(factor=factor, price_data=data)
        result["Rank_IC"] = metrics.get("Rank_IC")
        result["Sharpe_Ratio"] = metrics.get("Sharpe_Ratio")
        result["Sample_Size"] = metrics.get("Sample_Size")
    except Exception as exc:  # noqa: BLE001 - benchmark 需要把 evaluator 异常展示出来。
        result["sandbox_success"] = False
        result["error"] = repr(exc)

    return result


def print_result(result: dict[str, object]) -> None:
    """格式化输出单个因子结果。"""
    print("\n" + "=" * 80)
    print(f"因子：{result['factor']}")
    print(f"sandbox_success: {result['sandbox_success']}")
    print(f"elapsed_sec: {result['elapsed_sec']}")
    print(f"Rank_IC: {result['Rank_IC']}")
    print(f"Sharpe_Ratio: {result['Sharpe_Ratio']}")
    print(f"Sample_Size: {result['Sample_Size']}")
    print(f"NaN 比例: {result['nan_ratio']}")
    print(f"index 是否对齐: {result['index_aligned']}")
    if result.get("error"):
        print("error:")
        print(str(result["error"])[:3000])


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    data = load_data(limit_codes=args.limit_codes)
    print("全量数据加载完成")
    print(f"rows: {len(data):,}")
    print(f"codes: {data['code'].nunique()}")
    print(f"date_range: {data['date'].min().date()} ~ {data['date'].max().date()}")

    benchmark_factors = BENCHMARK_FACTORS
    if args.factor:
        if args.factor not in BENCHMARK_FACTORS:
            raise ValueError(f"未知因子：{args.factor}，可选：{list(BENCHMARK_FACTORS)}")
        benchmark_factors = {args.factor: BENCHMARK_FACTORS[args.factor]}

    results = []
    for name, code in benchmark_factors.items():
        result = run_one_factor(name=name, code=code, data=data)
        results.append(result)
        print_result(result)

    all_ok = all(result["sandbox_success"] and result["index_aligned"] for result in results)
    print("\n" + "=" * 80)
    print(f"ALL_BENCHMARKS_OK={all_ok}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
