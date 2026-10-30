import numpy as np
import pytest

from horizonkit.errors import InsufficientDataError
from horizonkit.loaders import RunData
from horizonkit.passk import estimate_passk, passk_from_counts


def test_unbiased_estimator_exact_values():
    assert passk_from_counts(5, 4, 2) == pytest.approx(0.6)
    r = estimate_passk([(5, 5), (5, 0), (5, 4)], k=1, n_bootstrap=500)
    assert r.estimate == pytest.approx((1.0 + 0.0 + 0.8) / 3)
    assert r.status == "ok"


def test_single_group_gets_trivial_interval_not_a_fake_one():
    r = estimate_passk([(20, 15)], k=3)
    assert r.status == "single_group_interval_uninformative"
    assert (r.ci_low, r.ci_high) == (0.0, 1.0)
    assert r.warnings


def test_excluded_groups_are_counted_not_hidden():
    r = estimate_passk([(10, 9), (10, 8), (2, 2)], k=5, n_bootstrap=500)
    assert r.n_groups_excluded == 1
    assert any("excluded" in w for w in r.warnings)


def test_inestimable_and_impossible_inputs_refuse():
    with pytest.raises(InsufficientDataError):
        estimate_passk([(3, 3), (4, 2)], k=5)
    with pytest.raises(InsufficientDataError):
        estimate_passk([(5, 7)], k=2)
    with pytest.raises(InsufficientDataError):
        passk_from_counts(2, 2, 3)


def test_interval_widens_as_groups_shrink():
    rng = np.random.default_rng(0)
    many = [(20, int(rng.binomial(20, 0.7))) for _ in range(60)]
    few = many[:4]
    rm = estimate_passk(many, k=4, n_bootstrap=800)
    rf = estimate_passk(few, k=4, n_bootstrap=800)
    assert (rm.ci_high - rm.ci_low) < (rf.ci_high - rf.ci_low)


def test_rundata_groups_feed_passk():
    x = np.tile(np.array([10.0, 20.0]), 12)
    y = np.array([1, 0] * 12)
    groups = tuple(["a"] * 12 + ["b"] * 12)
    d = RunData(x, y, groups)
    r = estimate_passk(d, k=2, n_bootstrap=300)
    assert r.n_groups == 2 and r.n_groups_used == 2
