import numpy as np
import pytest

from horizonkit.errors import InsufficientDataError
from horizonkit.loaders import AggregateData, RunData
from horizonkit.models import fit_all, fit_model, survival


def test_survival_formulas_and_limits():
    x = np.array([0.0, 10.0, 100.0])
    assert survival("exponential", {"lambda": 0.01}, x)[0] == 1.0
    # Lomax tends to the exponential as theta -> 0 (frailty identity limit).
    s_small_theta = survival("lomax", {"lambda": 0.01, "theta": 1e-6}, x)
    s_exp = survival("exponential", {"lambda": 0.01}, x)
    assert np.allclose(s_small_theta, s_exp, atol=1e-5)
    s_w = survival("weibull", {"kappa": 1.0, "eta": 100.0}, x)
    assert np.allclose(s_w, s_exp, atol=1e-12)


def planted(model, params, n=4000, seed=3):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(np.log(1), np.log(500), n))
    y = (rng.random(n) < survival(model, params, x)).astype(int)
    return RunData(x, y)


def test_fits_recover_planted_laws_approximately():
    cases = [
        ("exponential", {"lambda": 0.01}, "lambda", 0.007, 0.014),
        ("weibull", {"kappa": 0.7, "eta": 80.0}, "kappa", 0.55, 0.9),
        ("lomax", {"lambda": 0.02, "theta": 0.8}, "theta", 0.4, 1.4),
    ]
    for model, params, key, lo, hi in cases:
        fit = fit_model(planted(model, params), model)
        assert lo <= fit.params[key] <= hi
        assert fit.param_cis  # uncertainty is reported, not hidden


def test_fit_all_and_aic_order_on_weibull_data():
    fits = fit_all(planted("weibull", {"kappa": 0.65, "eta": 60.0}, seed=8))
    assert fits["weibull"].aic <= fits["exponential"].aic


def test_degenerate_logs_refuse():
    d = RunData(np.arange(1, 21, dtype=float), np.ones(20, dtype=int))
    with pytest.raises(InsufficientDataError):
        fit_model(d, "weibull")
    tiny = AggregateData(np.array([1.0, 2.0]), np.array([2, 2]), np.array([1, 1]))
    with pytest.raises(InsufficientDataError):
        fit_model(tiny, "exponential")
