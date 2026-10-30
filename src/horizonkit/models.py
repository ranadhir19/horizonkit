"""Maximum-likelihood fits of the three aggregate hazard laws.

The three laws are the ones in Ghosh's flagship manuscript:

* exponential / constant hazard: ``S(x) = exp(-lambda x)``;
* Weibull: ``S(x) = exp(-(x/eta)^kappa)`` (``kappa < 1`` is a declining
  hazard);
* Lomax: ``S(x) = (1 + theta lambda x)^(-1/theta)``, the population
  survival curve induced by a gamma frailty mixture of constant hazards
  (Vaupel et al.'s classical identity, used in the paper).

Fits maximise the Bernoulli/binomial likelihood of observed successes at
their horizons — exactly the aggregate fits M1–M3 in the flagship code
(``mph.py``), re-implemented with multiple starts, an observed-information
covariance on the log-parameter scale, and explicit optimizer status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import optimize, stats

from .errors import InsufficientDataError
from .loaders import AggregateData, RunData

MODELS = ("exponential", "weibull", "lomax")
_EPS = 1e-12


def survival(model: str, params: Mapping[str, float], x: np.ndarray | float) -> np.ndarray:
    """Success/survival probability at horizon(s) ``x``."""
    xx = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if model == "exponential":
            cumhaz = np.clip(float(params["lambda"]) * xx, 0.0, 745.0)
            out = np.exp(-cumhaz)
        elif model == "weibull":
            kappa = float(params["kappa"]); eta = float(params["eta"])
            log_cumhaz = np.where(xx > 0, kappa * (np.log(np.where(xx > 0, xx, 1.0)) - np.log(eta)), -np.inf)
            cumhaz = np.exp(np.clip(log_cumhaz, -745.0, 700.0))
            out = np.exp(-cumhaz)
        elif model == "lomax":
            lam = float(params["lambda"]); theta = float(params["theta"])
            log_term = np.where(xx > 0, np.log(theta) + np.log(lam) + np.log(np.where(xx > 0, xx, 1.0)), -np.inf)
            log_base = np.where(xx > 0, np.logaddexp(0.0, np.clip(log_term, -745.0, 700.0)), 0.0)
            out = np.exp(-log_base / theta)
        else:
            raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    return np.clip(out, 0.0, 1.0)


def _loglik(model: str, logparams: np.ndarray, data: AggregateData) -> float:
    params = _params_from_log(model, logparams)
    if params is None:
        return -np.inf
    p = np.clip(survival(model, params, data.horizons), _EPS, 1 - _EPS)
    return float(np.sum(data.successes * np.log(p) +
                        (data.trials - data.successes) * np.log1p(-p)))


def _params_from_log(model: str, logparams: np.ndarray) -> dict[str, float] | None:
    try:
        vals = np.exp(np.asarray(logparams, dtype=float))
    except OverflowError:
        return None
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        return None
    if model == "exponential":
        return {"lambda": float(vals[0])}
    if model == "weibull":
        return {"kappa": float(vals[0]), "eta": float(vals[1])}
    if model == "lomax":
        return {"lambda": float(vals[0]), "theta": float(vals[1])}
    raise ValueError(model)


def _starts(model: str, data: AggregateData) -> list[np.ndarray]:
    med = float(np.median(np.repeat(data.horizons, data.trials)))
    if med <= 0:
        med = float(np.median(data.horizons))
    lam0 = np.log(2.0) / max(med, _EPS)
    if model == "exponential":
        return [np.array([np.log(lam0)]), np.array([np.log(0.5 / med)]),
                np.array([np.log(2.0 / med)])]
    if model == "weibull":
        return [np.array([np.log(k), np.log(med)]) for k in (0.5, 0.8, 1.0, 1.3)]
    return [np.array([np.log(lam0), np.log(t)])
            for t in (0.1, 0.5, 1.0, 2.0)]


def _hessian_logparams(model: str, logparams: np.ndarray, data: AggregateData) -> np.ndarray:
    """Central-difference Hessian of the negative log-likelihood."""
    k = len(logparams)
    h = np.full(k, 1e-4)
    f0 = -_loglik(model, logparams, data)
    H = np.empty((k, k))
    for i in range(k):
        ei = np.zeros(k); ei[i] = h[i]
        fp = -_loglik(model, logparams + ei, data)
        fm = -_loglik(model, logparams - ei, data)
        H[i, i] = (fp - 2 * f0 + fm) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k); ej[j] = h[j]
            fpp = -_loglik(model, logparams + ei + ej, data)
            fpm = -_loglik(model, logparams + ei - ej, data)
            fmp = -_loglik(model, logparams - ei + ej, data)
            fmm = -_loglik(model, logparams - ei - ej, data)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    return H


@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    loglik: float
    n: int
    n_success: int
    param_cis: dict[str, tuple[float, float]] = field(default_factory=dict)
    logparam_covariance: list[list[float]] | None = None
    optimizer_success: bool = True
    optimizer_message: str = ""

    @property
    def k(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.k * np.log(max(self.n, 1)) - 2 * self.loglik

    def survival(self, x: np.ndarray | float) -> np.ndarray:
        return survival(self.model, self.params, x)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "params": self.params,
            "loglik": self.loglik, "aic": self.aic, "bic": self.bic,
            "n": self.n, "n_success": self.n_success,
            "param_cis": {k: list(v) for k, v in self.param_cis.items()},
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }


def _as_aggregate(data: AggregateData | RunData) -> AggregateData:
    if isinstance(data, AggregateData):
        return data
    if isinstance(data, RunData):
        return data.aggregate()
    raise TypeError("data must be AggregateData or RunData")


def fit_model(data: AggregateData | RunData, model: str, *, level: float = 0.95) -> FitResult:
    """Fit one law by maximum likelihood, with log-scale Wald CIs."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    agg = _as_aggregate(data)
    if agg.n < 8:
        raise InsufficientDataError(f"fitting requires at least 8 runs; got {agg.n}")
    if agg.n_success == 0 or agg.n_success == agg.n:
        raise InsufficientDataError(
            "all outcomes are identical; a decay law is not estimable from this log")

    def nll(lp):
        v = _loglik(model, np.asarray(lp, dtype=float), agg)
        return -v if np.isfinite(v) else 1e100

    best = None
    for start in _starts(model, agg):
        for method in ("Nelder-Mead", "L-BFGS-B"):
            try:
                r = optimize.minimize(nll, start, method=method,
                                      options={"maxiter": 2000} if method == "Nelder-Mead" else {})
            except Exception:
                continue
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
    if best is None:
        raise InsufficientDataError(f"{model} fit failed for every start/method")
    params = _params_from_log(model, np.asarray(best.x, dtype=float))
    assert params is not None
    loglik = -float(best.fun)

    # Observed-information covariance on the log-parameter scale.
    cis: dict[str, tuple[float, float]] = {}
    cov_list = None
    try:
        H = _hessian_logparams(model, np.asarray(best.x, dtype=float), agg)
        cov = np.linalg.pinv(H)
        cov_list = cov.tolist()
        z = float(stats.norm.ppf(0.5 + level / 2))
        names = list(params)
        for i, name in enumerate(names):
            se = float(np.sqrt(max(cov[i, i], 0.0))) if cov[i, i] > 0 else np.nan
            if np.isfinite(se) and se < 1e6:
                lo = float(np.exp(np.clip(best.x[i] - z * se, -745.0, 700.0)))
                hi = float(np.exp(np.clip(best.x[i] + z * se, -745.0, 700.0)))
                cis[name] = (lo, hi)
    except Exception:
        cov_list = None
    return FitResult(model=model, params=params, loglik=loglik, n=agg.n,
                     n_success=agg.n_success, param_cis=cis,
                     logparam_covariance=cov_list,
                     optimizer_success=bool(getattr(best, "success", True)),
                     optimizer_message=str(getattr(best, "message", "")))


def fit_all(data: AggregateData | RunData, *, level: float = 0.95) -> dict[str, FitResult]:
    """Fit exponential, Weibull, and Lomax to the same data."""
    return {m: fit_model(data, m, level=level) for m in MODELS}
