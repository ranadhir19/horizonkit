"""Planted-law self-test: the tool graded against known truths.

Following the flagship paper's certificate design, this module plants each
candidate law at a realistic per-model design (n = 1,300 runs over the
paper's horizon window), pushes the simulated aggregate record through the
*same* public pipeline a user runs (fit + identify), and reports what the
classifier actually does — correct names, refusals, and wrong names, with
no cherry-picking.  The Weibull and Lomax plantings are the paper's
matched pair (sup-norm distance 0.0257 over the window): the pair the
paper proves is hard to separate at this design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .identify import identify
from .loaders import AggregateData
from .models import survival

# Matched pair from the flagship certificate (results/phase2_cert_*).
PLANTED = {
    "lomax": {"lambda": 0.03500061704803072, "theta": 0.593161685642301},
    "weibull": {"kappa": 0.7665769445638558, "eta": 42.70111652456997},
    "exponential": {"lambda": float(np.log(2) / 25.0)},
}
WINDOW = (0.02, 1112.0)


def sample_design(n: int, *, seed: int = 0) -> np.ndarray:
    """Log-uniform horizon design over the flagship window."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(WINDOW[0]), np.log(WINDOW[1]), n))


def simulate_aggregate(model: str, params: dict[str, float],
                       x: np.ndarray, rng: np.random.Generator) -> AggregateData:
    """Aggregate Bernoulli draws at fixed distinct design horizons."""
    trials = np.ones(len(x), dtype=int)
    successes = (rng.random(len(x)) < survival(model, params, x)).astype(int)
    return AggregateData(np.asarray(x, dtype=float), trials, successes)


@dataclass
class SelfTestResult:
    n_per_rep: int
    reps: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_per_rep": self.n_per_rep, "reps": self.reps, "rows": self.rows}

    def summary_lines(self) -> list[str]:
        lines = []
        for row in self.rows:
            lines.append(
                f"truth={row['truth']}: named-correct {row['named_correct']:.2f}, "
                f"refused (not identified) {row['refused']:.2f}, "
                f"named-wrong {row['named_wrong']:.2f}")
        return lines


def run_selftest(*, reps: int = 20, n: int = 1300, seed: int = 20261029) -> SelfTestResult:
    """Run the planted-law simulation through the full public pipeline."""
    design = sample_design(n, seed=seed)
    rows: list[dict[str, Any]] = []
    for ti, (truth, params) in enumerate(PLANTED.items()):
        verdicts: list[str] = []
        wrong = 0
        for r in range(reps):
            rng = np.random.default_rng(seed + 1000 * ti + r)
            agg = simulate_aggregate(truth, params, design, rng)
            res = identify(agg)
            verdicts.append(res.verdict)
            if res.verdict not in (truth, "not_identified"):
                wrong += 1
        named_correct = float(np.mean([v == truth for v in verdicts])) if truth != "exponential" else 0.0
        refused = float(np.mean([v == "not_identified" for v in verdicts]))
        named_wrong = wrong / reps
        if truth == "exponential":
            # The classifier only arbitrates Weibull vs Lomax; for an
            # exponential planting both named verdicts are "wrong law" in
            # spirit, so report refusal as the honest outcome and count any
            # naming separately.
            named_correct = 0.0
            named_wrong = 1.0 - refused
        rows.append({
            "truth": truth, "planted_params": params,
            "named_correct": named_correct, "refused": refused,
            "named_wrong": float(named_wrong),
            "verdict_counts": {v: int(verdicts.count(v)) for v in sorted(set(verdicts))},
        })
    return SelfTestResult(n_per_rep=n, reps=reps, rows=rows)
