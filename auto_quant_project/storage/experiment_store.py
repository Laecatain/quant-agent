"""
Lightweight experiment persistence for alpha-mining trials.

This module is intentionally independent from the existing factors_pool logic in
``agents.factor_miner``. It provides a small JSONL/JSON based store that can be
integrated later without changing the current mining loop.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


DEFAULT_EXPERIMENT_DIR = Path("experiments")
TRIALS_FILE_NAME = "trials.jsonl"
BEST_FACTORS_FILE_NAME = "best_factors.json"


class ExperimentStore:
    """JSONL/JSON backed experiment store.

    Args:
        root_dir: Directory used to store experiment artifacts. It is created on
            demand. Defaults to ``./experiments`` relative to the process cwd.
    """

    def __init__(self, root_dir: str | Path = DEFAULT_EXPERIMENT_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.root_dir / TRIALS_FILE_NAME
        self.best_factors_path = self.root_dir / BEST_FACTORS_FILE_NAME

    def save_trial(self, trial: dict[str, Any]) -> Path:
        """Append a trial record to ``trials.jsonl`` using UTF-8 JSON Lines.

        The input dict is not mutated. Missing basic metadata is filled with
        lightweight defaults so callers can store trial, metrics and lineage
        records without boilerplate.
        """

        if not isinstance(trial, dict):
            raise TypeError("trial must be a dict")

        payload = dict(trial)
        payload.setdefault("trial_id", str(uuid.uuid4()))
        payload.setdefault("timestamp", _utc_timestamp())
        payload.setdefault("metrics", None)
        payload.setdefault("lineage", None)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.trials_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            file.write("\n")
        return self.trials_path

    def load_trials(self) -> list[dict[str, Any]]:
        """Load all trial records from ``trials.jsonl``.

        Blank lines are ignored. Invalid JSON lines raise ``ValueError`` with the
        line number to avoid silently corrupting experiment history.
        """

        if not self.trials_path.exists():
            return []

        trials: list[dict[str, Any]] = []
        with self.trials_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {self.trials_path} at line {line_no}: {exc.msg}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Trial record at line {line_no} must be a JSON object")
                trials.append(item)
        return trials

    def save_best_factors(self, factors: list[dict[str, Any]]) -> Path:
        """Atomically save the current best factor records to JSON.

        ``best_factors.json`` is written via a temporary file in the same
        directory and then replaced, reducing the chance of partial writes.
        """

        if not isinstance(factors, list):
            raise TypeError("factors must be a list of dicts")
        if not all(isinstance(item, dict) for item in factors):
            raise TypeError("each factor must be a dict")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": _utc_timestamp(),
            "count": len(factors),
            "factors": factors,
        }
        _atomic_write_json(self.best_factors_path, payload)
        return self.best_factors_path


def save_trial(trial: dict[str, Any], root_dir: str | Path = DEFAULT_EXPERIMENT_DIR) -> Path:
    """Convenience wrapper for ``ExperimentStore(root_dir).save_trial``."""

    return ExperimentStore(root_dir).save_trial(trial)


def load_trials(root_dir: str | Path = DEFAULT_EXPERIMENT_DIR) -> list[dict[str, Any]]:
    """Convenience wrapper for ``ExperimentStore(root_dir).load_trials``."""

    return ExperimentStore(root_dir).load_trials()


def save_best_factors(factors: list[dict[str, Any]], root_dir: str | Path = DEFAULT_EXPERIMENT_DIR) -> Path:
    """Convenience wrapper for ``ExperimentStore(root_dir).save_best_factors``."""

    return ExperimentStore(root_dir).save_best_factors(factors)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    temp_path.replace(path)


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
