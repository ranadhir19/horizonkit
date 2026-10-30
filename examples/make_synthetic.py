#!/usr/bin/env python3
"""Regenerate the bundled SYNTHETIC example run log.

The planting law is a Lomax (gamma-frailty mixture) with lambda=0.02,
theta=1.0 — chosen so the example exercises the identification machinery
on data whose truth is known.  This file contains no benchmark data: it
is simulated, and regenerating it with this script reproduces it exactly.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from horizonkit.models import survival  # noqa: E402

OUT = Path(__file__).with_name("synthetic_runs.csv")


def main() -> None:
    rng = np.random.default_rng(42)
    n = 800
    x = np.exp(rng.uniform(np.log(1), np.log(600), n))
    p = survival("lomax", {"lambda": 0.02, "theta": 1.0}, x)
    y = (rng.random(n) < p).astype(int)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["horizon", "success", "task_id"])
        for i, (xi, yi) in enumerate(zip(x, y)):
            w.writerow([f"{xi:.5f}", int(yi), f"synthetic-{i % 40:02d}"])
    print(f"wrote {OUT} ({n} rows)")


if __name__ == "__main__":
    main()
