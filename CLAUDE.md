# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python MVP for an auto-quant alpha factor researcher. The main loop downloads/loads CSI 300 daily market data, asks Gemini to generate vectorized pandas factor code, statically checks the generated code, runs it in a constrained local sandbox, evaluates train/valid/test performance, scores the trial, and persists factor results.

The import layout assumes `auto_quant_project/` is the application root. Entry scripts can be run from the repository root with paths shown below, while internal imports use package-like top-level modules such as `agents`, `core`, and `storage`.

## Common Commands

### Environment setup

Conda environment setup is configured in `auto_quant_project/environment.yml`:

```powershell
conda env create -f auto_quant_project\environment.yml
conda activate auto-quant-agent
```

A local venv also works with `requirements.txt`:

```powershell
python -m venv auto_quant_project\.venv
auto_quant_project\.venv\Scripts\python.exe -m pip install -r auto_quant_project\requirements.txt
```

On macOS/Linux, use the equivalent interpreter path:

```bash
python -m venv auto_quant_project/.venv
auto_quant_project/.venv/bin/python -m pip install -r auto_quant_project/requirements.txt
```

### Download or refresh market data

```powershell
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\download_data.py
```

Useful downloader variants:

```powershell
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\download_data.py --only-missing
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\download_data.py --force-refresh
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\download_data.py --limit 5
```

The downloader writes `auto_quant_project/data/csi300_daily.parquet` and may write `auto_quant_project/data/failed_codes.csv` for failed symbols.

### Run the factor-mining loop

Set one Gemini API key environment variable before running:

```powershell
$env:GEMINI_API_KEY='...'
# or
$env:GOOGLE_API_KEY='...'
```

Run the main loop:

```powershell
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\main.py --generations 5
```

Useful options:

```powershell
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\main.py --data-path auto_quant_project\data\csi300_daily.parquet --generations 1 --model gemini-1.5-flash --temperature 0.8 --top-k 5
```

Results are written to `auto_quant_project/factors_pool/` as per-trial JSON files plus `best_factors.json`.

### Run the sandbox/evaluator benchmark

```powershell
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\scripts\run_full_sandbox_benchmark.py
```

Debug variants:

```powershell
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\scripts\run_full_sandbox_benchmark.py --limit-codes 10
auto_quant_project\.venv\Scripts\python.exe auto_quant_project\scripts\run_full_sandbox_benchmark.py --factor "5日反转"
```

This benchmark requires `auto_quant_project/data/csi300_daily.parquet` and exits non-zero if any benchmark factor fails sandbox execution or index alignment.

### Tests and linting

There is currently no committed pytest, lint, type-check, or build configuration. If pytest is added as a development dependency, run targeted tests with commands such as:

```powershell
auto_quant_project\.venv\Scripts\python.exe -m pytest path\to\test_file.py -q
auto_quant_project\.venv\Scripts\python.exe -m pytest path\to\test_file.py::test_name -q
```

## Architecture Notes

### Data ingestion

`auto_quant_project/download_data.py` is the CSI 300 data ingestion script. It uses akshare to fetch current CSI 300 constituents and per-stock daily post-adjusted data for the configured date range, standardizes rows to `date`, `code`, `open`, `high`, `low`, `close`, `volume`, and `amount`, then merges and deduplicates by `(date, code)` into parquet. It deliberately clears proxy environment variables before importing akshare/requests because the original environment had a stale local proxy.

### Main orchestration

`auto_quant_project/main.py` loads the parquet data, configures `GeminiClient`, instantiates `FactorMiner`, and runs a fixed number of generations. The main input data file defaults to `auto_quant_project/data/csi300_daily.parquet`; the factor result directory defaults to `auto_quant_project/factors_pool/`.

### Factor generation loop

`auto_quant_project/agents/factor_miner.py` owns the alpha-mining loop:

1. Prepare and date-sort market data.
2. Split data by date into train/valid/test with `core.splitter.split_by_date`.
3. Build a JSON prompt containing recent and best trial feedback.
4. Ask Gemini for a JSON factor candidate with `name`, `hypothesis`, `code`, `lookback_days`, and `expected_direction`.
5. Run `core.static_checker.check_factor_code` before execution.
6. Run accepted code through `core.sandbox.run_factor_code`.
7. Evaluate each split with `core.evaluator.evaluate_factor`.
8. Score valid/test performance with `core.scoring.score_metrics` and persist trial JSON.

Generated factor code must assign a pandas `Series` named `factor`, use only `data`, `pd`, and `np`, avoid future-looking operations such as `shift(-1)`, and keep the factor index aligned to `data.index`.

### Static checking and sandboxing

`auto_quant_project/core/static_checker.py` is a conservative AST/string pre-check for LLM-generated factor code. It blocks imports, file/network/process access, reflection escape hatches, pandas external IO, row iteration over `data.iterrows()`/`data.itertuples()`, and negative `shift` usage.

`auto_quant_project/core/sandbox.py` then executes factor code with a small builtins set plus numpy and pandas. This is an MVP guardrail, not a strong security boundary. The sandbox deep-copies input data, requires `factor` to be a pandas `Series`, rejects duplicate factor indexes before reindexing, coerces factor values numeric, and reindexes to the original market data index.

### Evaluation and scoring

`auto_quant_project/core/evaluator.py` computes forward returns as `close.shift(-1) / close - 1` per stock, then calculates average cross-sectional Rank IC, long-short top/bottom quantile returns, annualized Sharpe ratio, ICIR, and sample size. Keep the no-future-function alignment intact when changing evaluator logic.

`auto_quant_project/core/scoring.py` combines split metrics into a final score. Valid/test performance drives the raw score, with penalties for train-test overfit gaps, missing/NaN metrics, insufficient sample size, and optional complexity.

`auto_quant_project/core/splitter.py` performs chronological date splits by unique dates so all rows from the same trading date stay in the same train/valid/test segment. It preserves original indexes by default because factor/evaluator alignment depends on those indexes.

### Persistence

`FactorMiner` currently persists mining-loop results directly to `auto_quant_project/factors_pool/`. `auto_quant_project/storage/experiment_store.py` provides a separate JSONL/JSON experiment store (`experiments/trials.jsonl`, `experiments/best_factors.json`) intended for later integration and uses atomic JSON writes for best-factor snapshots.

## Data and Generated Artifacts

The repository may contain local generated or heavyweight artifacts:

- `auto_quant_project/.venv/` — local virtual environment; do not edit or review vendored site-packages.
- `auto_quant_project/data/csi300_daily.parquet` — downloaded market data required by the main loop and benchmark.
- `auto_quant_project/factors_pool/` — generated trial outputs from `FactorMiner`.
- `experiments/` — optional JSONL/JSON experiment-store output if `storage.ExperimentStore` is used.

Avoid treating generated data/results as source unless the task explicitly asks for artifact inspection or migration. The current `.gitignore` only ignores root-level `data/*.csv`, `data/*.pkl`, and `data/*.parquet`, so be careful not to accidentally stage `auto_quant_project/data/`, `auto_quant_project/factors_pool/`, or `experiments/` outputs.
