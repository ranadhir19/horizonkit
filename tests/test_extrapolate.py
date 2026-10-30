import numpy as np
import pytest

from horizonkit.extrapolate import (EXPONENTIAL_T99_T50, extrapolate,
                                    quantile_horizon, t_ratio)
from horizonkit.loaders import RunData
from horizonkit.models import survival


def test_quantiles_invert_survival():
    cases = [
        ("exponential", {"lambda": 0.02}),
        ("weibull", {"kappa": 0.7, "eta": 50.0}),
        ("lomax", {"lambda": 0.02, "theta": 0.9}),
    ]
    for model, params in cases:
        for p in (0.5, 0.1, 0.01):
            x = quantile_horizon(model, params, p)
            assert float(survival(model, params, np.array([x]))[0]) == pytest.approx(p, rel=1e-9)


def test_exponential_ratio_is_the_naive_constant():
    r = t_ratio("exponential", {"lambda": 0.123})
    assert r == pytest.approx(EXPONENTIAL_T99_T50)
    # A heavy frailty tail makes T99/T50 vastly larger than 6.64x.
    r_lomax = t_ratio("lomax", {"lambda": 0.01, "theta": 2.0})
    assert r_lomax > 25 * r


def test_extrapolation_warns_when_laws_diverge():
    rng = np.random.default_rng(11)
    x = np.exp(rng.uniform(np.log(1), np.log(600), 2500))
    y = (rng.random(len(x)) < survival("lomax", {"lambda": 0.02, "theta": 1.2}, x)).astype(int)
    res = extrapolate(RunData(x, y))
    assert "T99" in res.divergent_targets
    assert any("IDENTIFICATION WARNING" in w or "Material divergence" in w
               for w in res.warnings)
    assert res.horizons["lomax"]["T99"] > res.horizons["exponential"]["T99"]
    # The 170x rule's error is reported per law, not asserted.
    assert set(res.rule_errors["circulating_170x"]) == {"exponential", "weibull", "lomax"}
