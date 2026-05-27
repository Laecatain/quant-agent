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
    print("\n========== Phase 2 运行结束 ==========")
    print(f"总试验次数：{len(trials)}")
    if best:
        best_trial = best[0]
        print(f"当前最佳因子：{best_trial.candidate.name}")
        print(f"指标：{best_trial.metrics}")
        print(f"得分：{best_trial.score:.4f}")
    else:
        print("暂无成功因子，请查看 factors_pool/trial_*.json 中的报错反馈。")

    print(f"结果目录：{DEFAULT_FACTORS_POOL}")


if __name__ == "__main__":
    main()
