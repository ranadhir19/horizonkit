"""Honest pass^k estimation at small n.

``pass^k`` here is the reliability quantity used in agent evaluation: the
probability that *k independent attempts at the same problem all succeed*.
For problem ``i`` with ``n_i`` attempts and ``c_i`` successes, the standard
unbiased estimator (Chen et al.'s pass@k estimator, specialised to
all-success) is ``C(c_i, k) / C(n_i, k)``; HorizonKit averages it over
problems and — the part most tools skip — attaches an interval whose width
tells the truth about the design:

* with at least two problem groups, a cluster bootstrap over problems;
* with exactly one group, the point estimate is reported with the trivial
  [0, 1] interval and an explicit warning — resampling one problem cannot
  manufacture between-problem uncertainty information;
* groups with fewer than k attempts are excluded and counted, never
  silently treated as failures or successes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import InsufficientDataError
from .loaders import RunData


@dataclass
class PassKResult:
    k: int
    estimate: float
    ci_low: float
    ci_high: float
    level: float
    n_groups: int
    n_groups_used: int
    n_groups_excluded: int
    status: str  # "ok", or "single_group_interval_uninformative"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k, "estimate": self.estimate,
            "ci": [self.ci_low, self.ci_high], "level": self.level,
            "n_groups": self.n_groups, "n_groups_used": self.n_groups_used,
            "n_groups_excluded": self.n_groups_excluded,
            "status": self.status, "warnings": list(self.warnings),
        }


def _unbiased(c: int, n: int, k: int) -> float:
    return comb(c, k) / comb(n, k)


def estimate_passk(groups: Iterable[tuple[int, int]] | RunData,
                   k: int, *, level: float = 0.95,
                   n_bootstrap: int = 2000, seed: int = 0) -> PassKResult:
    """Estimate pass^k from ``(attempts, successes)`` per problem.

    ``groups`` may be an iterable of ``(n_attempts, n_successes)`` pairs or
    a :class:`~horizonkit.loaders.RunData` with group labels.
    """
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise InsufficientDataError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    if isinstance(groups, RunData):
        pairs = [(n, c) for _, n, c in groups.group_counts()]
    else:
        pairs = [(int(n), int(c)) for n, c in groups]
    if not pairs:
        raise InsufficientDataError("pass^k requires at least one problem group")
    for n, c in pairs:
        if n <= 0 or c < 0 or c > n:
            raise InsufficientDataError(f"impossible group counts (attempts={n}, successes={c})")

    usable = [(n, c) for n, c in pairs if n >= k]
    excluded = len(pairs) - len(usable)
    if not usable:
        raise InsufficientDataError(
            f"pass^{k} is not estimable: every group has fewer than k={k} attempts")
    per_group = np.array([_unbiased(c, n, k) for n, c in usable], dtype=float)
    estimate = float(np.mean(per_group))

    warnings: list[str] = []
    if excluded:
        warnings.append(
            f"{excluded} group(s) excluded for having fewer than k={k} attempts; "
            "they were not counted as successes or failures.")
    if len(usable) == 1:
        warnings.append(
            "Only one problem group: between-problem uncertainty is not "
            "identified by this design, so the interval is the trivial "
            "[0, 1]. Collect repeated attempts on more problems to narrow it.")
        return PassKResult(k=k, estimate=estimate, ci_low=0.0, ci_high=1.0,
                           level=level, n_groups=len(pairs),
                           n_groups_used=1, n_groups_excluded=excluded,
                           status="single_group_interval_uninformative",
                           warnings=tuple(warnings))
    if len(usable) < 10:
        warnings.append(
            f"Only {len(usable)} problem groups: the bootstrap interval is "
            "itself noisy at this group count.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_group), size=(n_bootstrap, len(per_group)))
    boot = np.mean(per_group[idx], axis=1)
    alpha = 1 - level
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return PassKResult(k=k, estimate=estimate, ci_low=float(lo), ci_high=float(hi),
                       level=level, n_groups=len(pairs), n_groups_used=len(usable),
                       n_groups_excluded=excluded, status="ok",
                       warnings=tuple(warnings))


def passk_from_counts(attempts: int, successes: int, k: int) -> float:
    """The unbiased point estimate for one pooled group (no interval)."""
    if attempts < k:
        raise InsufficientDataError(
            f"pass^{k} needs at least k attempts in the group; got {attempts}")
    return _unbiased(successes, attempts, k)
