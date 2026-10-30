from horizonkit.selftest import PLANTED, run_selftest, sample_design, simulate_aggregate
import numpy as np


def test_design_and_simulation_shapes():
    x = sample_design(50, seed=1)
    assert len(x) == 50 and np.all(x >= 0.02) and np.all(x <= 1112.0)
    agg = simulate_aggregate("lomax", PLANTED["lomax"], x, np.random.default_rng(0))
    assert agg.n == 50


def test_selftest_small_run_reports_all_truths():
    res = run_selftest(reps=3, n=400, seed=7)
    assert [r["truth"] for r in res.rows] == ["lomax", "weibull", "exponential"]
    for row in res.rows:
        total = sum(row["verdict_counts"].values())
        assert total == 3
        assert 0.0 <= row["refused"] <= 1.0
    assert res.summary_lines()
