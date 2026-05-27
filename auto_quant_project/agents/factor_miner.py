"""
Alpha 因子自进化挖掘 Agent。

核心闭环：
LLM 生成因子代码 -> 本地沙盒 exec 执行 -> evaluator 计算 Rank IC / Sharpe ->
将报错或指标反馈给 LLM -> LLM 反思并变异下一代因子。

Phase 2 目标是建立可运行的最小闭环，而不是追求生产级搜索效率。
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.backtester import StrategySpec, backtest_factor_strategy
from core.evaluator import evaluate_factor
from core.llm_client import GeminiClient
from core.sandbox import SandboxResult, run_factor_code
from core.scoring import score_metrics
from core.splitter import DateSplit, split_by_date
from core.static_checker import check_factor_code


SYSTEM_PROMPT = """
你是一位严谨的量化研究员与 Python 向量化编程专家。
你的任务是生成 A 股日频 Alpha 因子代码，用于预测下一交易日横截面收益。

硬性约束：
1. 只能输出 JSON，不能输出 Markdown、解释文本或代码围栏。
2. JSON 必须包含字段：name, hypothesis, code, lookback_days, expected_direction。
3. code 必须是 Python 代码字符串，执行后必须产生 pandas.Series 变量 factor。
4. 可用变量只有 data, pd, np。data 字段为：date, code, open, high, low, close, volume, amount。
5. 必须全向量化，严禁逐行 for 循环；允许 groupby、rolling、rank、pct_change、transform。
6. 严禁未来函数：不得使用 shift(-1) 或任何未来价格/未来收益构造 factor。
7. 因子索引必须与 data.index 对齐。推荐写法：
   df = data.sort_values(['code', 'date']).copy()
   ...
   factor = some_series.reindex(data.index)
8. 因子应尽量具备横截面区分度，不要输出常数或几乎全 NaN。
""".strip()


@dataclass(frozen=True)
class FactorCandidate:
    """LLM 生成的因子候选。"""

    name: str
    hypothesis: str
    code: str
    lookback_days: int
    expected_direction: str


@dataclass(frozen=True)
class FactorTrial:
    """一次因子试验记录。"""

    trial_id: str
    timestamp: str
    generation: int
    candidate: FactorCandidate
    sandbox_success: bool
    metrics: dict[str, Any] | None
    error: str | None
    score: float


class FactorMiner:
    """
    控制 Alpha 因子进化循环的 Agent。

    评分函数采用简单线性组合：
        score = Sharpe_Ratio + 5 * abs(Rank_IC)
    这里 Rank_IC 乘以 5 是为了让横截面预测能力在分数中有足够权重。
    生产级系统应加入样本外验证、换手/容量/行业中性等约束。
    """

    def __init__(
        self,
        llm_client: GeminiClient,
        data: pd.DataFrame,
        factors_pool_dir: str | Path,
        top_k: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.data = self._prepare_data(data)
        self.factors_pool_dir = Path(factors_pool_dir)
        self.factors_pool_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.data_splits = split_by_date(self.data)
        self.trials: list[FactorTrial] = []

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化输入行情数据。

        因子计算依赖时间序列窗口，必须确保每只股票内部按 date 升序排列。
        这里保留原始行索引的语义：后续 sandbox 会把 factor reindex 回 data.index。
        """
        required = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"行情数据缺少必要字段：{missing}")

        prepared = data.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])
        prepared["code"] = prepared["code"].astype(str).str.zfill(6)
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        prepared[numeric_cols] = prepared[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return prepared.sort_values(["code", "date"]).reset_index(drop=True)

    def run(self, generations: int = 5) -> list[FactorTrial]:
        """
        执行多代因子挖掘。

        Args:
            generations: 迭代代数。每代生成一个候选因子并立刻回测反馈。

        Returns:
            list[FactorTrial]: 全部试验记录。
        """
        for generation in range(1, generations + 1):
            print(f"\n========== Generation {generation}/{generations} ==========")
            candidate = self.generate_candidate(generation=generation)
            trial = self.evaluate_candidate(candidate=candidate, generation=generation)
            self.trials.append(trial)
            self._persist_trial(trial)
            self._persist_best_factors()

            if trial.sandbox_success:
                print(
                    f"完成：{candidate.name} | score={trial.score:.4f} | "
                    f"valid_Rank_IC={self._metric_for_log(trial.metrics, 'valid', 'Rank_IC')} | "
                    f"valid_Sharpe={self._metric_for_log(trial.metrics, 'valid', 'Sharpe_Ratio')}"
                )
            else:
                print(f"失败：{candidate.name} | {trial.error[:300] if trial.error else '未知错误'}")

        return self.trials

    def generate_candidate(self, generation: int) -> FactorCandidate:
        """让 LLM 基于历史反馈生成下一代因子。"""
        prompt = self._build_generation_prompt(generation=generation)
        raw_text = self.llm_client.generate_text(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        payload = self._parse_json(raw_text)
        return self._candidate_from_payload(payload)

    def evaluate_candidate(self, candidate: FactorCandidate, generation: int) -> FactorTrial:
        """执行候选因子并计算 train/valid/test 指标。"""
        check_result = check_factor_code(candidate.code)
        if not check_result.passed:
            checker_error = self._format_checker_feedback(check_result.errors, check_result.warnings)
            return self._make_trial(
                generation=generation,
                candidate=candidate,
                sandbox_result=SandboxResult(success=False, factor=None, error=checker_error),
                metrics={
                    "static_checker": {
                        "passed": False,
                        "errors": check_result.errors,
                        "warnings": check_result.warnings,
                    }
                },
                score=float("-inf"),
            )

        sandbox_result = run_factor_code(candidate.code, self.data)

        if not sandbox_result.success:
            return self._make_trial(
                generation=generation,
                candidate=candidate,
                sandbox_result=sandbox_result,
                metrics={
                    "static_checker": {
                        "passed": True,
                        "errors": [],
                        "warnings": check_result.warnings,
                    }
                },
                score=float("-inf"),
            )

        assert sandbox_result.factor is not None
        try:
            metrics_by_split = self._evaluate_factor_by_split(sandbox_result.factor)
            metrics_by_split["static_checker"] = {
                "passed": True,
                "errors": [],
                "warnings": check_result.warnings,
            }
            score_breakdown = score_metrics(metrics_by_split)
            metrics_by_split["score_breakdown"] = score_breakdown
            try:
                metrics_by_split["backtest"] = self._backtest_factor_by_split(sandbox_result.factor)
            except Exception as exc:  # noqa: BLE001 - 回测失败应反馈但不阻断因子评分。
                metrics_by_split["backtest_error"] = repr(exc)
            score = float(score_breakdown["final_score"])
            return self._make_trial(
                generation=generation,
                candidate=candidate,
                sandbox_result=sandbox_result,
                metrics=metrics_by_split,
                score=score,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator 异常也要反馈给 LLM。
            return self._make_trial(
                generation=generation,
                candidate=candidate,
                sandbox_result=SandboxResult(success=False, factor=None, error=repr(exc)),
                metrics={
                    "static_checker": {
                        "passed": True,
                        "errors": [],
                        "warnings": check_result.warnings,
                    }
                },
                score=float("-inf"),
            )

    def _build_generation_prompt(self, generation: int) -> str:
        """构造包含数据概况与历史反馈的提示词。"""
        data_profile = {
            "rows": int(len(self.data)),
            "codes": int(self.data["code"].nunique()),
            "min_date": str(self.data["date"].min().date()),
            "max_date": str(self.data["date"].max().date()),
            "columns": list(self.data.columns),
        }

        history = [self._trial_to_prompt_item(trial) for trial in self.trials[-8:]]
        best = [self._trial_to_prompt_item(trial) for trial in self.best_trials(limit=3)]

        return json.dumps(
            {
                "task": "生成下一代 A 股日频 Alpha 因子。",
                "generation": generation,
                "data_profile": data_profile,
                "recent_trials": history,
                "best_trials": best,
                "instructions": [
                    "如果上一代报错，优先修复代码结构、索引对齐、字段名、NaN 问题。",
                    "如果上一代成功但 Sharpe 或 Rank_IC 较差，请从经济假设上变异，而不是微调常数。",
                    "优先探索价量关系、短期反转、动量、波动率压缩/扩张、量价背离。",
                    "严禁使用 shift(-1) 构造 factor；未来收益只由 evaluator 计算。",
                    "输出严格 JSON：name, hypothesis, code, lookback_days, expected_direction。",
                ],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any]:
        """
        从模型输出中解析 JSON。

        虽然我们要求 responseMimeType=application/json，但为了鲁棒性，仍兼容模型偶发输出
        ```json ... ``` 或前后附加文本的情况。
        """
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise ValueError(f"LLM 输出不是合法 JSON：{raw_text[:1000]}")
            return json.loads(match.group(0))

    @staticmethod
    def _candidate_from_payload(payload: dict[str, Any]) -> FactorCandidate:
        """校验并转换 LLM JSON 为 FactorCandidate。"""
        required = ["name", "hypothesis", "code", "lookback_days", "expected_direction"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"候选因子 JSON 缺少字段：{missing}，实际为：{payload}")

        return FactorCandidate(
            name=str(payload["name"]).strip()[:80],
            hypothesis=str(payload["hypothesis"]).strip(),
            code=str(payload["code"]).strip(),
            lookback_days=int(payload["lookback_days"]),
            expected_direction=str(payload["expected_direction"]).strip(),
        )

    @staticmethod
    def _score_metrics(metrics: dict[str, float]) -> float:
        """将指标压缩成单一进化分数。"""
        rank_ic = metrics.get("Rank_IC", float("nan"))
        sharpe = metrics.get("Sharpe_Ratio", float("nan"))

        if math.isnan(rank_ic) and math.isnan(sharpe):
            return float("-inf")
        if math.isnan(rank_ic):
            rank_ic = 0.0
        if math.isnan(sharpe):
            sharpe = 0.0

        return float(sharpe + 5.0 * abs(rank_ic))

    def best_trials(self, limit: int | None = None) -> list[FactorTrial]:
        """按 score 返回当前最佳试验。"""
        valid = [trial for trial in self.trials if trial.sandbox_success and trial.metrics]
        valid = sorted(valid, key=lambda trial: trial.score, reverse=True)
        return valid[: limit or self.top_k]

    def _make_trial(
        self,
        generation: int,
        candidate: FactorCandidate,
        sandbox_result: SandboxResult,
        metrics: dict[str, Any] | None,
        score: float,
    ) -> FactorTrial:
        """统一生成试验记录。"""
        return FactorTrial(
            trial_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            generation=generation,
            candidate=candidate,
            sandbox_success=sandbox_result.success,
            metrics=metrics,
            error=sandbox_result.error,
            score=score,
        )

    @staticmethod
    def _trial_to_prompt_item(trial: FactorTrial) -> dict[str, Any]:
        """压缩 trial 信息，作为下一轮提示词反馈。"""
        return {
            "generation": trial.generation,
            "name": trial.candidate.name,
            "hypothesis": trial.candidate.hypothesis,
            "code": trial.candidate.code,
            "success": trial.sandbox_success,
            "metrics": trial.metrics,
            "score": trial.score if math.isfinite(trial.score) else None,
            "error": trial.error[-1200:] if trial.error else None,
        }

    def _evaluate_factor_by_split(self, factor: pd.Series) -> dict[str, Any]:
        """在 train/valid/test 三段上分别评估因子。"""
        metrics_by_split: dict[str, Any] = {}
        for split_name, split_data in self.data_splits.as_dict().items():
            split_factor = factor.reindex(split_data.index)
            metrics_by_split[split_name] = evaluate_factor(factor=split_factor, price_data=split_data)
        return metrics_by_split

    def _backtest_factor_by_split(self, factor: pd.Series) -> dict[str, Any]:
        """在 train/valid/test 三段上分别回测因子策略并返回可 JSON 序列化摘要。"""
        backtest_by_split: dict[str, Any] = {}
        strategy_spec = StrategySpec()
        for split_name, split_data in self.data_splits.as_dict().items():
            split_factor = factor.reindex(split_data.index)
            result = backtest_factor_strategy(data=split_data, factor=split_factor, spec=strategy_spec)
            backtest_by_split[split_name] = self._summarize_backtest_result(result)
        return backtest_by_split

    @staticmethod
    def _summarize_backtest_result(result: dict[str, object]) -> dict[str, Any]:
        """压缩回测结果，避免 trial JSON 写入完整持仓与净值序列。"""
        daily_returns = result.get("daily_returns")
        equity_curve = result.get("equity_curve")
        metrics = result.get("metrics")
        turnover = result.get("turnover")

        daily_return_count = int(len(daily_returns)) if isinstance(daily_returns, pd.Series) else 0
        final_equity = None
        if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
            final_equity = float(equity_curve.iloc[-1])

        average_turnover = None
        if isinstance(turnover, pd.Series) and not turnover.empty:
            average_turnover = float(turnover.mean())

        return {
            "metrics": dict(metrics) if isinstance(metrics, dict) else {},
            "daily_return_count": daily_return_count,
            "final_equity": final_equity,
            "average_turnover": average_turnover,
        }

    @staticmethod
    def _format_checker_feedback(errors: list[str], warnings: list[str]) -> str:
        """将 static checker 结果压缩成可进入 trial.error 的文本。"""
        payload = {
            "stage": "static_checker",
            "passed": False,
            "errors": errors,
            "warnings": warnings,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _metric_for_log(metrics: dict[str, Any] | None, split: str, key: str) -> Any:
        """安全读取日志展示用指标。"""
        if not metrics:
            return None
        split_metrics = metrics.get(split)
        if not isinstance(split_metrics, dict):
            return None
        return split_metrics.get(key)

    def _persist_trial(self, trial: FactorTrial) -> None:
        """将每次试验保存为 JSON，便于中断后复盘。"""
        file_name = f"trial_gen_{trial.generation:03d}_{trial.trial_id[:8]}.json"
        path = self.factors_pool_dir / file_name
        payload = asdict(trial)
        payload["score"] = trial.score if math.isfinite(trial.score) else None
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_best_factors(self) -> None:
        """保存当前 best factors 索引。"""
        best_payload = []
        for trial in self.best_trials(limit=self.top_k):
            item = asdict(trial)
            item["score"] = trial.score if math.isfinite(trial.score) else None
            best_payload.append(item)

        path = self.factors_pool_dir / "best_factors.json"
        path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
