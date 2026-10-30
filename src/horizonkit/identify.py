"""Identification accounting for aggregate horizon curves.

This is a faithful productisation of the flagship paper's aggregate
classifier (``mph.classify_aggregate``): fit Weibull and Lomax, run a
binned chi-square goodness-of-fit test for each, reject a law only when its
GOF p-value falls below alpha, and REFUSE (``not_identified``) when both or
neither law is rejected.

The paper's two scope statements are part of the API, not fine print:

* identification is ONE-DIRECTIONAL at the per-model aggregate design —
  aggregate data cannot exclude the frailty (Lomax) explanation of an
  apparent declining hazard, while declining-hazard data can exclude the
  frailty law; and
* identification is DESIGN-RELATIVE — a property of the sample size,
  horizon window and replication design, not of the laws alone.  Pooling
  or replication can separate laws that this design cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .errors import NotIdentifiedError
from .loaders import AggregateData, RunData
from .models import FitResult, fit_model, survival

ONE_DIRECTIONAL_DISCLOSURE = (
    "One-directional identification: at a per-model aggregate design, "
    "aggregate curves cannot exclude the frailty (Lomax) explanation of an "
    "apparent declining hazard; declining-hazard data can exclude the "
    "frailty law. A Weibull-looking aggregate curve is therefore weak "
    "evidence about the underlying hazard."
)
DESIGN_RELATIVITY_DISCLOSURE = (
    "Design-relativity: identification is a property of this design "
    "(sample size, horizon window, replication), not of the laws alone. "
    "Pooled or replicated designs can separate laws that this aggregate "
    "record cannot."
)


def gof_chisq(data: AggregateData, fit: FitResult, *, n_bins: int = 10) -> tuple[float, int, float]:
    """Binned chi-square GOF, ported from flagship ``mph.gof_chisq``.

    Bins are trial-weighted quantiles of the horizon distribution; the
    degrees of freedom subtract the fitted parameter count, because the
    unadjusted test is anti-conservative (Phase-1 pilot in the paper).
    """
    x = data.horizons; n = data.trials; s = data.successes
    cdf = np.cumsum(n) / np.sum(n)
    edges_idx = []
    targets = np.linspace(0, 1, n_bins + 1)
    for t in targets:
        edges_idx.append(int(np.searchsorted(cdf, t, side="left")))
    edges_idx = sorted(set([0] + [min(i, len(x) - 1) for i in edges_idx] + [len(x) - 1]))
    stat = 0.0; used = 0
    for a, b in zip(edges_idx[:-1], edges_idx[1:]):
        sl = slice(a, b + 1)
        m = int(np.sum(n[sl]))
        if m < 5:
            continue
        p_hat = float(np.sum(n[sl] * survival(fit.model, fit.params, x[sl])) / m)
        exp_s = m * p_hat
        obs_s = float(np.sum(s[sl]))
        var = max(m * p_hat * (1 - p_hat), 1e-9)
        stat += (obs_s - exp_s) ** 2 / var
        used += 1
    df = max(used - 1 - fit.k, 1)
    return float(stat), int(df), float(stats.chi2.sf(stat, df))


def curve_distance(fit_a: FitResult, fit_b: FitResult, data: AggregateData) -> float:
    """Sup-norm distance between two fitted curves over the observed window."""
    grid = np.exp(np.linspace(np.log(data.horizons.min()), np.log(data.horizons.max()), 400))
    return float(np.max(np.abs(fit_a.survival(grid) - fit_b.survival(grid))))


@dataclass
class IdentificationResult:
    verdict: str  # "weibull", "lomax", or "not_identified"
    weibull_fit: FitResult
    lomax_fit: FitResult
    weibull_gof_p: float
    lomax_gof_p: float
    sup_norm_distance: float
    n: int
    reason: str
    disclosures: tuple[str, ...] = (ONE_DIRECTIONAL_DISCLOSURE, DESIGN_RELATIVITY_DISCLOSURE)
    alpha: float = 0.05

    @property
    def identified(self) -> bool:
        return self.verdict != "not_identified"

    def require_identified(self) -> str:
        """Return the verdict, or raise the named refusal."""
        if not self.identified:
            raise NotIdentifiedError(
                f"not identified: {self.reason} Refusal is the honest output "
                "for this aggregate design.")
        return self.verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "identified": self.identified,
            "reason": self.reason, "alpha": self.alpha, "n": self.n,
            "weibull_gof_p": self.weibull_gof_p, "lomax_gof_p": self.lomax_gof_p,
            "sup_norm_distance": self.sup_norm_distance,
            "weibull_fit": self.weibull_fit.to_dict(),
            "lomax_fit": self.lomax_fit.to_dict(),
            "disclosures": list(self.disclosures),
        }


def identify(data: AggregateData | RunData, *, alpha: float = 0.05,
             n_bins: int = 10) -> IdentificationResult:
    """Classify an aggregate record as Weibull, Lomax, or not identified."""
    agg = data.aggregate() if isinstance(data, RunData) else data
    w = fit_model(agg, "weibull")
    l = fit_model(agg, "lomax")
    _, _, p_w = gof_chisq(agg, w, n_bins=n_bins)
    _, _, p_l = gof_chisq(agg, l, n_bins=n_bins)
    dist = curve_distance(w, l, agg)
    rej_w, rej_l = p_w < alpha, p_l < alpha
    if rej_l and not rej_w:
        verdict = "weibull"
        reason = (f"Lomax is rejected by GOF (p={p_l:.3g}) and Weibull is not "
                  f"(p={p_w:.3g}); the declining-hazard law is the one the "
                  "aggregate record can support.")
    elif rej_w and not rej_l:
        verdict = "lomax"
        reason = (f"Weibull is rejected by GOF (p={p_w:.3g}) and Lomax is not "
                  f"(p={p_l:.3g}); this is the direction aggregate data can "
                  "sometimes resolve — but read the one-directional disclosure "
                  "before treating it as hazard evidence.")
    elif rej_w and rej_l:
        verdict = "not_identified"
        reason = (f"both laws are rejected by GOF (Weibull p={p_w:.3g}, "
                  f"Lomax p={p_l:.3g}); neither candidate describes the record, "
                  "so no law is named.")
    else:
        verdict = "not_identified"
        reason = (f"neither law is rejected by GOF (Weibull p={p_w:.3g}, "
                  f"Lomax p={p_l:.3g}) and their fitted curves differ by at "
                  f"most {dist:.3f} in success probability over the observed "
                  "window; the aggregate record cannot distinguish them.")
    return IdentificationResult(verdict=verdict, weibull_fit=w, lomax_fit=l,
                                weibull_gof_p=p_w, lomax_gof_p=p_l,
                                sup_norm_distance=dist, n=agg.n,
                                reason=reason, alpha=alpha)


# Backwards-friendly alias using the flagship code's name.
classify_aggregate = identify
