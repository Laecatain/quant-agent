"""
沪深 300 日线后复权数据下载与补全脚本。

目标：
1. 使用 akshare 获取沪深 300 成分股在 2023-2026 区间的日线后复权行情；
2. 标准化字段为：['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']；
3. 保存至 data/csi300_daily.parquet；
4. 支持断点续传、失败重试、与已有 parquet 合并去重，便于多次补数。

运行方式：
    python download_data.py

常用参数：
    python download_data.py --max-retries 5 --sleep 0.8
    python download_data.py --only-missing
    python download_data.py --force-refresh
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from time import sleep
from typing import Iterable

# 当前机器存在系统代理指向 127.0.0.1:7890 但服务未启动的情况，会导致 akshare 请求失败。
# 在导入 requests/akshare 前关闭代理环境变量，强制走直连。
for proxy_key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import akshare as ak
import pandas as pd


# Windows 某些终端默认编码不是 UTF-8，直接打印中文会触发 UnicodeEncodeError。
# 这里显式切换 stdout/stderr 编码，保证日志输出不影响数据下载主流程。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# 项目根目录：当前脚本所在目录。
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "csi300_daily.parquet"
FAILED_PATH = DATA_DIR / "failed_codes.csv"

# 当前阶段指定区间：过去三年范围覆盖 2023-2026。
START_DATE = "20230101"
END_DATE = "20261231"

# 统一输出字段。后续所有因子公式都应基于这套稳定 schema 编写，降低 LLM 幻觉概率。
STANDARD_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="下载并补全沪深 300 日线后复权数据")
    parser.add_argument("--start-date", default=START_DATE, help="开始日期，格式 yyyymmdd")
    parser.add_argument("--end-date", default=END_DATE, help="结束日期，格式 yyyymmdd")
    parser.add_argument("--max-retries", type=int, default=5, help="单只股票最大重试次数")
    parser.add_argument("--sleep", type=float, default=0.8, help="每次请求后的基础等待秒数")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="失败重试的基础等待秒数")
    parser.add_argument("--only-missing", action="store_true", help="只下载当前 parquet 中不存在的股票")
    parser.add_argument("--force-refresh", action="store_true", help="忽略已有数据，强制重下全部股票")
    parser.add_argument("--limit", type=int, default=0, help="仅调试用：限制下载股票数量，0 表示不限")
    return parser.parse_args()


def normalize_stock_code(raw_code: str) -> str:
    """
    将 akshare 返回的股票代码规范为 6 位字符串。

    akshare 不同接口可能返回 int、float 或字符串。量化数据中代码是离散标识，
    不能被当作数值参与计算，因此这里统一转成零填充的字符串。
    """
    return str(raw_code).strip().zfill(6)


def get_csi300_codes() -> list[str]:
    """
    获取沪深 300 最新成分股代码列表。

    注意：这是当前成分股快照，不是历史成分股还原。
    当前目标是尽快补全 MVP 数据；历史成分股变更与幸存者偏差控制将在后续阶段增强。
    """
    cons = ak.index_stock_cons(symbol="000300")

    # akshare 字段名偶有变化，这里做兼容处理，优先寻找包含“品种代码/代码”的列。
    candidate_columns = ["品种代码", "成分券代码", "股票代码", "代码"]
    code_col = next((col for col in candidate_columns if col in cons.columns), None)
    if code_col is None:
        raise ValueError(f"无法在沪深 300 成分股结果中识别代码列，实际字段为：{list(cons.columns)}")

    codes = cons[code_col].map(normalize_stock_code).dropna().unique().tolist()
    if not codes:
        raise ValueError("沪深 300 成分股代码列表为空，请检查 akshare 接口返回。")
    return codes


def load_existing_data(path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """读取已有 parquet；不存在则返回空 DataFrame。"""
    if not path.exists():
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    data = pd.read_parquet(path)
    if data.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    data = data[STANDARD_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["code"] = data["code"].map(normalize_stock_code)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=["date", "code", "close"])


def save_merged_data(existing: pd.DataFrame, new_data: pd.DataFrame, path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """
    合并新旧数据并落盘。

    数学上，每条行情样本由二元索引 (date, code) 唯一确定。
    因此合并时按 ['date', 'code'] 去重，保留最后一次下载结果，避免重复样本污染回测。
    """
    frames = [df for df in [existing, new_data] if not df.empty]
    if not frames:
        raise RuntimeError("没有可保存的数据。")

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged["code"] = merged["code"].map(normalize_stock_code)
    merged = merged[STANDARD_COLUMNS]
    merged = merged.drop_duplicates(subset=["date", "code"], keep="last")
    merged = merged.sort_values(["date", "code"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def rename_akshare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 akshare 中文字段映射为统一英文行情字段。

    计算数学视角：后续因子本质是对时间截面矩阵 X_{t,i} 的变换。
    稳定的字段命名相当于固定矩阵维度语义，避免公式生成阶段出现变量歧义。
    """
    column_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    renamed = df.rename(columns=column_map)

    missing = [col for col in STANDARD_COLUMNS if col != "code" and col not in renamed.columns]
    if missing:
        raise ValueError(f"行情数据缺少必要字段：{missing}，实际字段为：{list(df.columns)}")

    return renamed


def fetch_one_stock_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    下载单只股票的日线后复权数据，并标准化字段。

    adjust='hfq' 表示后复权：价格序列会根据分红送转进行连续化处理，
    有利于时间序列因子计算，减少除权缺口对收益率估计的扰动。
    """
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="hfq",
    )

    if raw.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = rename_akshare_columns(raw)
    df["code"] = code

    # 类型清洗：date 为 pandas 日期；OHLCV/amount 强制转数值，非法值变 NaN。
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # 严格保留统一字段并按时间排序。
    df = df[STANDARD_COLUMNS].dropna(subset=["date", "code", "close"])
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def fetch_one_stock_with_retry(
    code: str,
    start_date: str,
    end_date: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, str | None]:
    """带指数退避的单票下载。成功返回数据和 None；失败返回空数据和错误字符串。"""
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            data = fetch_one_stock_daily(code=code, start_date=start_date, end_date=end_date)
            return data, None
        except Exception as exc:  # noqa: BLE001 - 下载边界需要捕获所有 akshare/网络异常。
            last_error = repr(exc)
            if attempt < max_retries:
                delay = retry_sleep * attempt + random.uniform(0.0, 0.8)
                print(f"    重试 {attempt}/{max_retries - 1}，等待 {delay:.1f}s，错误：{last_error[:160]}")
                sleep(delay)

    return pd.DataFrame(columns=STANDARD_COLUMNS), last_error


def select_codes_to_download(
    all_codes: list[str],
    existing: pd.DataFrame,
    only_missing: bool,
    force_refresh: bool,
) -> list[str]:
    """根据已有数据选择需要下载的股票。"""
    if force_refresh or existing.empty:
        return all_codes

    existing_codes = set(existing["code"].map(normalize_stock_code).unique())
    missing_codes = [code for code in all_codes if code not in existing_codes]

    if only_missing:
        return missing_codes

    # 默认补全策略：已有股票也重新拉一遍，用合并去重覆盖旧值；缺失股票自然会补上。
    # 这样可以修复部分股票历史区间不完整的问题，但仍然支持断点合并。
    return all_codes


def fetch_csi300_daily(
    codes: Iterable[str],
    existing: pd.DataFrame,
    start_date: str,
    end_date: str,
    max_retries: int,
    sleep_seconds: float,
    retry_sleep: float,
    flush_every: int = 10,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    批量下载沪深 300 日线行情，并周期性落盘。

    周期性落盘是断点续传的关键：如果 akshare 远端中途断连或进程中断，
    已成功股票不会丢失，下次运行可继续补缺失股票。
    """
    merged = existing.copy()
    batch_frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    codes = list(codes)

    for idx, code in enumerate(codes, start=1):
        print(f"[{idx:03d}/{len(codes):03d}] 下载 {code} ...")
        one, error = fetch_one_stock_with_retry(
            code=code,
            start_date=start_date,
            end_date=end_date,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        if error:
            failed.append((code, error))
            print(f"[WARN] {code} 下载失败：{error[:240]}")
        elif not one.empty:
            batch_frames.append(one)
            print(f"    成功：{len(one)} 行")
        else:
            print("    返回空数据，可能为新股或接口暂不可用。")

        if batch_frames and (idx % flush_every == 0 or idx == len(codes)):
            merged = save_merged_data(existing=merged, new_data=pd.concat(batch_frames, ignore_index=True))
            print(f"    已落盘：{len(merged):,} 行，{merged['code'].nunique()} 只股票")
            batch_frames.clear()

        # 轻微限速 + 抖动，降低被远端接口拒绝的概率。
        sleep(sleep_seconds + random.uniform(0.0, 0.4))

    if batch_frames:
        merged = save_merged_data(existing=merged, new_data=pd.concat(batch_frames, ignore_index=True))
        batch_frames.clear()

    if failed:
        failed_df = pd.DataFrame(failed, columns=["code", "error"])
        failed_df.to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")

    return merged, failed


def main() -> None:
    """脚本入口：获取成分股、增量下载行情、合并落地 parquet。"""
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    codes = get_csi300_codes()
    if args.limit > 0:
        codes = codes[: args.limit]
    print(f"沪深 300 成分股数量：{len(codes)}")

    existing = pd.DataFrame(columns=STANDARD_COLUMNS) if args.force_refresh else load_existing_data()
    if not existing.empty:
        print(
            f"已有数据：{len(existing):,} 行，{existing['code'].nunique()} 只股票，"
            f"{existing['date'].min().date()} ~ {existing['date'].max().date()}"
        )

    target_codes = select_codes_to_download(
        all_codes=codes,
        existing=existing,
        only_missing=args.only_missing,
        force_refresh=args.force_refresh,
    )
    print(f"本次计划下载：{len(target_codes)} 只股票")

    if not target_codes:
        print("没有需要补全的股票。")
        return

    data, failed = fetch_csi300_daily(
        codes=target_codes,
        existing=existing,
        start_date=args.start_date,
        end_date=args.end_date,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep,
        retry_sleep=args.retry_sleep,
    )

    print(f"\n数据已保存：{OUTPUT_PATH}")
    print(f"数据规模：{len(data):,} 行，{data['code'].nunique()} 只股票")
    print(f"日期范围：{data['date'].min().date()} ~ {data['date'].max().date()}")

    if failed:
        print(f"仍有 {len(failed)} 只股票失败，明细已保存：{FAILED_PATH}")
        for code, error in failed[:20]:
            print(f"- {code}: {error[:180]}")


if __name__ == "__main__":
    main()
