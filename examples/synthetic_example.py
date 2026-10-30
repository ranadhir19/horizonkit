#!/usr/bin/env python3
"""Worked example on the bundled synthetic run log.

Prints the fits, the identification verdict, a pass^k estimate and the
extrapolation table for data planted from a known Lomax law, ending with
a machine-checkable SUMMARY_JSON line (validate.py compares it against
examples/synthetic_expected.json).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from horizonkit import estimate_passk, fit_all, identify, load_runs  # noqa: E402
from horizonkit.extrapolate import extrapolate  # noqa: E402
from horizonkit.report import (extrapolate_report, fit_report,  # noqa: E402
                               identify_report, passk_report)

DATA = Path(__file__).with_name("synthetic_runs.csv")


def main() -> None:
    data = load_runs(DATA)
    fits = fit_all(data)
    ident = identify(data)
    pk = estimate_passk(data, k=3, n_bootstrap=1000, seed=1)
    ext = extrapolate(data)
    print(fit_report(data, fits))
    print()
    print(identify_report(data, ident))
    print()
    print(passk_report(pk))
    print()
    print(extrapolate_report(data, ext))
    summary = {
        "n": data.n,
        "n_success": data.n_success,
        "verdict": ident.verdict,
        "weibull_kappa": round(fits["weibull"].params["kappa"], 3),
        "lomax_theta": round(fits["lomax"].params["theta"], 3),
        "passk3": round(pk.estimate, 4),
        "t99": {m: round(ext.horizons[m]["T99"], 1) for m in ext.horizons},
    }
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
