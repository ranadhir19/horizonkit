import numpy as np
import pytest

from horizonkit.errors import NotIdentifiedError
from horizonkit.identify import identify
from horizonkit.loaders import RunData
from horizonkit.models import survival

# The flagship certificate's matched pair: the Weibull below minimises the
# sup-norm distance (0.0257) to this Lomax over the design window.
LOMAX = {"lambda": 0.03500061704803072, "theta": 0.593161685642301}
WEIBULL = {"kappa": 0.7665769445638558, "eta": 42.70111652456997}


def matched_sample(model, params, seed, n=1300):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(np.log(0.02), np.log(1112.0), n))
    y = (rng.random(n) < survival(model, params, x)).astype(int)
    return RunData(x, y)


def test_near_miss_matched_lomax_must_refuse():
    # NEAR-MISS NEGATIVE (mandatory): data from the frailty law, at the
    # per-model design size, must NOT be confidently named Weibull.
    result = identify(matched_sample("lomax", LOMAX, seed=20261029))
    assert result.verdict == "not_identified", result.reason
    assert "One-directional" in result.disclosures[0]
    assert "Design-relativity" in result.disclosures[1]
    with pytest.raises(NotIdentifiedError):
        result.require_identified()


def test_matched_weibull_is_never_named_lomax():
    # The asymmetry cuts the other way: declining-hazard data may be named
    # Weibull or refused, but (at this seed/design) must not affirm frailty.
    result = identify(matched_sample("weibull", WEIBULL, seed=20261030))
    assert result.verdict in {"weibull", "not_identified"}, result.reason


def test_strong_weibull_signal_can_be_identified():
    rng = np.random.default_rng(5)
    x = np.exp(rng.uniform(np.log(1), np.log(1000), 6000))
    y = (rng.random(len(x)) < survival("weibull", {"kappa": 0.45, "eta": 30.0}, x)).astype(int)
    result = identify(RunData(x, y))
    assert result.verdict == "weibull", result.reason
