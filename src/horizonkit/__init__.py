"""HorizonKit: agent reliability statistics with identification honesty.

Given agent run logs, HorizonKit fits the constant-hazard (exponential),
Weibull, and gamma-frailty (Lomax) laws of task success versus horizon;
reports which laws the aggregate record can actually identify — refusing,
by name, when it cannot; gives honest pass^k intervals at small n; and
extrapolates T50/T90/T99 with the identification warning attached.

Reference implementation for: Ghosh, *What Time-Horizon Curves Cannot
Tell Us* (manuscript in preparation for JMLR).
"""
from .errors import (HorizonKitError, InsufficientDataError,
                     InvalidRunLogError, NotIdentifiedError)
from .identify import (DESIGN_RELATIVITY_DISCLOSURE,
                       ONE_DIRECTIONAL_DISCLOSURE, IdentificationResult,
                       classify_aggregate, identify)
from .loaders import AggregateData, RunData, load_aggregate, load_runs
from .models import MODELS, FitResult, fit_all, fit_model, survival
from .passk import PassKResult, estimate_passk, passk_from_counts

__version__ = "0.1.0"

__all__ = [
    "__version__", "MODELS",
    "HorizonKitError", "InsufficientDataError", "InvalidRunLogError",
    "NotIdentifiedError",
    "AggregateData", "RunData", "load_runs", "load_aggregate",
    "FitResult", "fit_model", "fit_all", "survival",
    "IdentificationResult", "identify", "classify_aggregate",
    "ONE_DIRECTIONAL_DISCLOSURE", "DESIGN_RELATIVITY_DISCLOSURE",
    "PassKResult", "estimate_passk", "passk_from_counts",
]
