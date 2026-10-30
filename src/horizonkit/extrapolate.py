"""T50/T90/T99 extrapolation — with the identification warning attached.

For a success curve ``S(x)``, ``Tq`` is the horizon at which predicted
success falls to ``1 - q/100`` — T50 at S=0.50, T90 at S=0.10, T99 at
S=0.01.  The flagship paper's warning is structural: laws that aggregate
data cannot distinguish imply wildly different tails, so a single-law
T99 is not a measurement, it is an assumption.  This module therefore
always reports the three fitted laws side by side, flags material
divergence, and evaluates circulating tail rules of thumb (the
exponential-implied T99/T50 = ln(100)/ln(2) ≈ 6.64, and the
frailty-flavoured T99 ≈ 170 × T50 sometimes quoted from the time-horizon
literature) against each fitted law with explicit relative errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .identify import IdentificationResult, identify
from .loaders import AggregateData, RunData
from .models import FitResult, fit_all

TARGETS = {"T50": 0.50, "T90": 0.10, "T99": 0.01}
EXPONENTIAL_T99_T50 = float(np.log(100) / np.log(2))  # 6.6439
CIRCULATING_RULES = {"exponential_implied": EXPONENTIAL_T99_T50,
                     "circulating_170x": 170.0}
DIVERGENCE_RATIO = 1.25


def quantile_horizon(model: str, params: Mapping[str, float], survival_prob: float) -> float:
    """Horizon x at which the fitted success curve equals ``survival_prob``."""
    p = float(survival_prob)
    if not 0 < p < 1:
        raise ValueError("survival_prob must lie in (0, 1)")
    if model == "exponential":
        return float(-np.log(p) / float(params["lambda"]))
    if model == "weibull":
        return float(float(params["eta"]) * (-np.log(p)) ** (1.0 / float(params["kappa"])))
    if model == "lomax":
        lam = float(params["lambda"]); theta = float(params["theta"])
        return float((p ** (-theta) - 1.0) / (theta * lam))
    raise ValueError(f"unknown model {model!r}")


def t_ratio(model: str, params: Mapping[str, float], *, hi: float = 0.01, lo: float = 0.50) -> float:
    """T(hi)/T(lo) implied by one fitted law (flagship ``t_ratio_*``)."""
    return quantile_horizon(model, params, hi) / quantile_horizon(model, params, lo)


@dataclass
class ExtrapolationResult:
    horizons: dict[str, dict[str, float]]  # model -> target -> horizon
    ratios_t99_t50: dict[str, float]
    rule_errors: dict[str, dict[str, float]]  # rule -> model -> relative error
    divergent_targets: tuple[str, ...]
    identification: IdentificationResult | None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizons": self.horizons,
            "ratios_t99_t50": self.ratios_t99_t50,
            "rule_relative_errors": self.rule_errors,
            "divergent_targets": list(self.divergent_targets),
            "identification": self.identification.to_dict() if self.identification else None,
            "warnings": list(self.warnings),
        }


def extrapolate(data: AggregateData | RunData | Mapping[str, FitResult],
                *, include_identification: bool = True) -> ExtrapolationResult:
    """Fit all laws (or accept pre-fitted ones) and extrapolate honestly."""
    if isinstance(data, Mapping):
        fits = dict(data)
        ident = None
    else:
        fits = fit_all(data)
        ident = identify(data) if include_identification else None

    horizons = {
        model: {target: quantile_horizon(model, fit.params, p) for target, p in TARGETS.items()}
        for model, fit in fits.items()
    }
    ratios = {model: t_ratio(model, fit.params) for model, fit in fits.items()}
    rule_errors = {
        rule: {model: float(rule_value / ratios[model] - 1.0) for model in fits}
        for rule, rule_value in CIRCULATING_RULES.items()
    }
    divergent = []
    for target in TARGETS:
        vals = np.array([horizons[m][target] for m in fits], dtype=float)
        if np.max(vals) / max(np.min(vals), 1e-300) > DIVERGENCE_RATIO:
            divergent.append(target)

    warnings: list[str] = []
    if ident is not None and not ident.identified:
        warnings.append(
            "IDENTIFICATION WARNING: the aggregate record does not identify "
            f"the law ({ident.reason}) Extrapolated horizons below are "
            "law-relative, not measurements; report the range across laws, "
            "never a single law's tail.")
    if divergent:
        warnings.append(
            "Material divergence at " + ", ".join(divergent) +
            f": the fitted laws disagree by more than {DIVERGENCE_RATIO:.2f}× "
            "on the same data.")
    spread = max(ratios.values()) / min(ratios.values())
    warnings.append(
        f"Tail rules of thumb are law-relative: fitted T99/T50 spans "
        f"{min(ratios.values()):.2f}×–{max(ratios.values()):.2f}× across laws "
        f"({spread:.1f}× spread). The exponential-implied "
        f"{EXPONENTIAL_T99_T50:.2f}× and the circulating 170× rule each carry "
        "the per-law relative errors tabulated in this result; neither is a "
        "law-free constant.")
    return ExtrapolationResult(horizons=horizons, ratios_t99_t50=ratios,
                               rule_errors=rule_errors,
                               divergent_targets=tuple(divergent),
                               identification=ident, warnings=tuple(warnings))
