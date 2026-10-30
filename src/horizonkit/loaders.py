"""Load and validate agent run logs.

HorizonKit's estimand is intentionally narrow: each run contributes a
positive horizon ``x`` (task length/time, in whatever unit the source uses)
and a binary success indicator.  The aggregate record is the multiset of
``(x, success)`` pairs, optionally grouped by task/problem/model.

Supported inputs are deliberately boring, user-supplied files:

* generic CSV or JSON/JSONL, one row per run;
* aggregate CSV/JSON rows with ``trials`` and ``successes`` columns;
* METR-style JSONL using ``human_minutes`` and ``score_binarized``;
* HAL-style JSON/JSONL records using a duration/step field and a
  ``success``/``score``/``passed`` field.

No benchmark data is bundled with the package.  METR and HAL files are read
only when the user supplies them.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import InvalidRunLogError

_HORIZON_FIELDS = (
    "horizon", "human_minutes", "task_length", "length", "duration",
    "duration_seconds", "elapsed_time", "elapsed_seconds", "time", "minutes",
    "tokens", "token_count", "steps", "num_steps", "step_count", "x",
)
_SUCCESS_FIELDS = (
    "success", "succeeded", "passed", "pass", "outcome", "y",
    "score_binarized", "score", "correct", "is_success",
)
_GROUP_FIELDS = (
    "group", "problem_id", "task_id", "task", "problem", "benchmark",
    "model", "alias", "id",
)
_TRIAL_FIELDS = ("trials", "n", "count", "attempts", "n_attempts")
_SUCCESSES_FIELDS = ("successes", "n_success", "success_count", "num_successes")


@dataclass(frozen=True)
class AggregateData:
    """Binomial counts at distinct horizons."""

    horizons: np.ndarray
    trials: np.ndarray
    successes: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.horizons, dtype=float)
        n = np.asarray(self.trials, dtype=int)
        s = np.asarray(self.successes, dtype=int)
        if x.ndim != 1 or n.shape != x.shape or s.shape != x.shape:
            raise InvalidRunLogError("horizons, trials and successes must be aligned vectors")
        if len(x) == 0:
            raise InvalidRunLogError("aggregate data is empty")
        if not np.all(np.isfinite(x)) or np.any(x <= 0):
            raise InvalidRunLogError("horizons must be finite and strictly positive")
        if np.any(n <= 0) or np.any(s < 0) or np.any(s > n):
            raise InvalidRunLogError("counts must satisfy 0 <= successes <= trials and trials > 0")
        order = np.argsort(x)
        object.__setattr__(self, "horizons", x[order])
        object.__setattr__(self, "trials", n[order])
        object.__setattr__(self, "successes", s[order])

    @property
    def n(self) -> int:
        return int(np.sum(self.trials))

    @property
    def n_success(self) -> int:
        return int(np.sum(self.successes))

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n if self.n else float("nan")

    def expanded(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one horizon/outcome pair per run (small/medium data)."""
        x = np.repeat(self.horizons, self.trials)
        y = np.concatenate([
            np.concatenate([np.ones(int(s), dtype=int), np.zeros(int(n - s), dtype=int)])
            for n, s in zip(self.trials, self.successes)
        ])
        return x, y


@dataclass
class RunData:
    """One row per run, with optional group labels."""

    horizons: np.ndarray
    successes: np.ndarray
    groups: tuple[str, ...] | None = None
    source_format: str = "generic"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.horizons = np.asarray(self.horizons, dtype=float)
        self.successes = np.asarray(self.successes, dtype=int)
        if self.horizons.ndim != 1 or self.successes.shape != self.horizons.shape:
            raise InvalidRunLogError("horizons and successes must be aligned vectors")
        if len(self.horizons) == 0:
            raise InvalidRunLogError("run log is empty")
        if not np.all(np.isfinite(self.horizons)) or np.any(self.horizons <= 0):
            raise InvalidRunLogError("horizons must be finite and strictly positive")
        if not np.all(np.isin(self.successes, [0, 1])):
            raise InvalidRunLogError("success values must be binary (0/1, true/false)")
        if self.groups is not None:
            self.groups = tuple(str(g) for g in self.groups)
            if len(self.groups) != len(self.horizons):
                raise InvalidRunLogError("groups must align with runs")

    @property
    def n(self) -> int:
        return int(len(self.horizons))

    @property
    def n_success(self) -> int:
        return int(np.sum(self.successes))

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n if self.n else float("nan")

    def aggregate(self) -> AggregateData:
        """Collapse exact duplicate horizons into binomial counts."""
        totals: dict[float, list[int]] = {}
        for x, y in zip(self.horizons, self.successes):
            rec = totals.setdefault(float(x), [0, 0])
            rec[0] += 1
            rec[1] += int(y)
        xs = np.array(sorted(totals), dtype=float)
        return AggregateData(xs,
                             np.array([totals[x][0] for x in xs], dtype=int),
                             np.array([totals[x][1] for x in xs], dtype=int))

    def group_counts(self) -> list[tuple[str, int, int]]:
        """Return ``(group, attempts, successes)`` sorted by group label."""
        if self.groups is None:
            return [("all", self.n, self.n_success)]
        out: dict[str, list[int]] = {}
        for g, y in zip(self.groups, self.successes):
            rec = out.setdefault(g, [0, 0])
            rec[0] += 1
            rec[1] += int(y)
        return [(g, out[g][0], out[g][1]) for g in sorted(out)]


def _first(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for f in fields:
        if f in lower and lower[f] not in (None, ""):
            return lower[f]
    return None


def _as_bool_success(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) in (0.0, 1.0):
            return int(value)
        raise InvalidRunLogError(f"numeric success/score must be 0 or 1, got {value!r}")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "success", "succeeded", "pass", "passed", "correct"}:
            return 1
        if v in {"0", "false", "no", "failure", "failed", "fail", "incorrect"}:
            return 0
    raise InvalidRunLogError(f"unrecognised success value {value!r}")


def _rows_from_json_obj(obj: Any) -> list[Mapping[str, Any]]:
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, Mapping):
        rows = None
        for key in ("runs", "rollouts", "results", "records", "data", "rows"):
            if isinstance(obj.get(key), list):
                rows = obj[key]
                break
        if rows is None:
            rows = [obj]
    else:
        raise InvalidRunLogError("JSON input must be an object, a list, or a recognised container")
    if not all(isinstance(r, Mapping) for r in rows):
        raise InvalidRunLogError("every JSON record must be an object")
    return list(rows)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read CSV, JSON, or JSONL records from a user-supplied path."""
    p = Path(path)
    if not p.exists():
        raise InvalidRunLogError(f"run log not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(newline="", encoding="utf-8-sig") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    if suffix == ".jsonl":
        rows = []
        with p.open(encoding="utf-8-sig") as fh:
            for i, line in enumerate(fh, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise InvalidRunLogError(f"invalid JSON on line {i}: {exc}") from exc
        return _rows_from_json_obj(rows)
    if suffix == ".json":
        return _rows_from_json_obj(json.loads(p.read_text(encoding="utf-8-sig")))
    # Extension-free JSONL is common for benchmark exports.
    text = p.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        try:
            return _rows_from_json_obj([json.loads(ln) for ln in lines])
        except json.JSONDecodeError:
            pass
    return _rows_from_json_obj(json.loads(text))


def rows_to_runs(rows: Iterable[Mapping[str, Any]], *, source_format: str = "generic") -> RunData:
    """Convert record mappings to validated run-level data.

    Aggregate rows (trials/successes) are expanded to Bernoulli runs; the
    expansion is capped to keep accidental huge files from exhausting memory.
    """
    xs: list[float] = []
    ys: list[int] = []
    groups: list[str] = []
    saw_group = False
    total_expanded = 0
    for idx, row in enumerate(rows):
        x_raw = _first(row, _HORIZON_FIELDS)
        if x_raw is None:
            raise InvalidRunLogError(f"record {idx} has no recognised horizon field")
        try:
            x = float(x_raw)
        except (TypeError, ValueError) as exc:
            raise InvalidRunLogError(f"record {idx} has non-numeric horizon {x_raw!r}") from exc

        group_raw = _first(row, _GROUP_FIELDS)
        group = "all" if group_raw is None else str(group_raw)
        saw_group = saw_group or group_raw is not None

        trials_raw = _first(row, _TRIAL_FIELDS)
        successes_raw = _first(row, _SUCCESSES_FIELDS)
        y_raw = _first(row, _SUCCESS_FIELDS)
        if trials_raw is not None and successes_raw is not None and y_raw is None:
            n = int(trials_raw); s = int(successes_raw)
            if n <= 0 or s < 0 or s > n:
                raise InvalidRunLogError(f"record {idx} has impossible aggregate counts")
            total_expanded += n
            if total_expanded > 5_000_000:
                raise InvalidRunLogError("aggregate expansion exceeds 5,000,000 runs; supply run-level rows")
            xs.extend([x] * n)
            ys.extend([1] * s + [0] * (n - s))
            groups.extend([group] * n)
        else:
            if y_raw is None:
                raise InvalidRunLogError(f"record {idx} has no recognised success field")
            xs.append(x); ys.append(_as_bool_success(y_raw)); groups.append(group)
    return RunData(np.array(xs, dtype=float), np.array(ys, dtype=int),
                   tuple(groups) if saw_group else None, source_format=source_format)


def load_runs(path: str | Path, *, format: str = "auto") -> RunData:
    """Load a generic, METR-format, or HAL-format run log.

    ``format`` may be ``auto``, ``generic``, ``metr``, or ``hal``.  METR and
    HAL are field conventions, not bundled data; auto-detection simply uses
    their documented field names when present.
    """
    fmt = format.lower()
    if fmt not in {"auto", "generic", "metr", "hal"}:
        raise InvalidRunLogError("format must be one of: auto, generic, metr, hal")
    rows = read_records(path)
    if fmt == "metr":
        # The flagship paper's frozen METR extract: alias/task_id identify the
        # cell; human_minutes is the horizon and score_binarized the outcome.
        normalised = []
        for r in rows:
            alias = r.get("alias", "")
            task = r.get("task_id", "")
            normalised.append({
                "horizon": r.get("human_minutes"),
                "success": r.get("score_binarized"),
                "group": f"{alias}::{task}" if alias or task else "all",
            })
        return rows_to_runs(normalised, source_format="metr")
    return rows_to_runs(rows, source_format=("generic" if fmt == "auto" else fmt))


def load_aggregate(path: str | Path) -> AggregateData:
    """Load already-aggregated horizon/trials/successes data."""
    data = load_runs(path)
    return data.aggregate()
