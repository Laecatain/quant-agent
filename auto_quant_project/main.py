"""
Auto-Quant Researcher 主入口。

Phase 2：启动 LLM Agent 自进化因子挖掘循环。

运行前准备：
1. 确认 data/csi300_daily.parquet 已存在；
2. 设置 Gemini API Key：
   PowerShell:
       $env:GEMINI_API_KEY='你的密钥'
   CMD:
       set GEMINI_API_KEY=你的密钥

示例：
    python main.py --generations 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from agents.factor_miner import FactorMiner
from core.llm_client import GeminiClient, GeminiConfig


# Windows 终端中文输出兼容。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "csi300_daily.parquet"
DEFAULT_FACTORS_POOL = PROJECT_ROOT / "factors_pool"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Auto-Quant Researcher Alpha 因子自进化引擎")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="行情 parquet 数据路径，默认 data/csi300_daily.parquet",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="进化代数，每代生成并评估一个因子。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini 模型名称。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="LLM 采样温度，越高探索性越强。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="保存到 best_factors.json 的最佳因子数量。",
    )
    return parser.parse_args()


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not pd.notna(number):
        return "NA"
    return f"{number:.{digits}f}"


def _split_metrics_summary(metrics: dict[str, Any], split: str) -> str:
    split_metrics = metrics.get(split)
    if not isinstance(split_metrics, dict):
        return f"{split}: NA"
    return (
        f"{split}: Rank_IC={_format_number(split_metrics.get('Rank_IC'))}, "
        f"ICIR={_format_number(split_metrics.get('ICIR'))}, "
        f"Sharpe={_format_number(split_metrics.get('Sharpe_Ratio'))}, "
        f"Sample_Size={_format_number(split_metrics.get('Sample_Size'), digits=0)}"
    )


def _backtest_summary(metrics: dict[str, Any], split: str) -> str:
    backtest = metrics.get("backtest")
    if not isinstance(backtest, dict):
        return f"{split} backtest: NA"
    split_backtest = backtest.get(split)
    if not isinstance(split_backtest, dict):
        return f"{split} backtest: NA"
    backtest_metrics = split_backtest.get("metrics")
    if not isinstance(backtest_metrics, dict):
        backtest_metrics = {}
    return (
        f"{split} backtest: ann_return={_format_number(backtest_metrics.get('annualized_return'))}, "
        f"sharpe={_format_number(backtest_metrics.get('sharpe'))}, "
        f"max_drawdown={_format_number(backtest_metrics.get('max_drawdown'))}, "
        f"turnover={_format_number(split_backtest.get('average_turnover'))}, "
        f"final_equity={_format_number(split_backtest.get('final_equity'))}"
    )


def _print_best_strategy_summary(best_trial: Any) -> None:
    metrics = best_trial.metrics if isinstance(best_trial.metrics, dict) else {}
    score_breakdown = metrics.get("score_breakdown") if isinstance(metrics, dict) else {}
    details = score_breakdown.get("details", {}) if isinstance(score_breakdown, dict) else {}
    quality_reasons = details.get("quality_reasons", []) if isinstance(details, dict) else []
    overfit_reasons = details.get("overfit_reasons", []) if isinstance(details, dict) else []

    print(f"当前最佳策略：{best_trial.candidate.name}")
    print(f"假设：{best_trial.candidate.hypothesis}")
    print(f"策略参数：{best_trial.candidate.strategy}")
    print(_split_metrics_summary(metrics, "valid"))
    print(_split_metrics_summary(metrics, "test"))
    print(_backtest_summary(metrics, "valid"))
    print(_backtest_summary(metrics, "test"))
    print(f"综合得分：{best_trial.score:.4f}")
    if isinstance(score_breakdown, dict):
        print(
            "评分分解："
            f"raw={_format_number(score_breakdown.get('raw_score'))}, "
            f"overfit_penalty={_format_number(score_breakdown.get('overfit_penalty'))}, "
            f"quality_penalty={_format_number(score_breakdown.get('quality_penalty'))}"
        )
    if quality_reasons or overfit_reasons:
        print(f"惩罚原因：{'; '.join(str(reason) for reason in [*overfit_reasons, *quality_reasons])}")


def load_market_data(path: Path) -> pd.DataFrame:
    """读取本地 parquet 行情数据。"""
    if not path.exists():
        raise FileNotFoundError(f"行情数据不存在：{path}。请先运行 python download_data.py")

    data = pd.read_parquet(path)
    if data.empty:
        raise ValueError(f"行情数据为空：{path}")

    return data


def main() -> None:
    """主流程：加载数据 -> 初始化 LLM -> 运行因子挖掘 Agent。"""
    args = parse_args()

    data = load_market_data(args.data_path)
    print(
        "数据加载完成："
        f"{len(data):,} 行，{data['code'].nunique()} 只股票，"
        f"{pd.to_datetime(data['date']).min().date()} ~ {pd.to_datetime(data['date']).max().date()}"
    )

    config = GeminiConfig(model=args.model, temperature=args.temperature)
    llm_client = GeminiClient(config=config)

    miner = FactorMiner(
        llm_client=llm_client,
        data=data,
        factors_pool_dir=DEFAULT_FACTORS_POOL,
        top_k=args.top_k,
    )
    trials = miner.run(generations=args.generations)

    best = miner.best_trials(limit=1)
    print("\n========== 自动策略研究运行结束 ==========")
    print(f"总试验次数：{len(trials)}")
    if best:
        _print_best_strategy_summary(best[0])
    else:
        print("暂无成功策略，请查看 factors_pool/trial_*.json 中的报错反馈。")

    print(f"结果目录：{DEFAULT_FACTORS_POOL}")


if __name__ == "__main__":
    main()
