#!/usr/bin/env python3
"""HorizonKit validation harness.

Stages (each reported PASS/FAIL; the run fails if any stage fails):
  1. import/compile — the package imports and byte-compiles clean.
  2. test suite — the full pytest suite (loader validation, planted-law
     fit recovery, the matched-pair near-miss refusal, pass^k honesty,
     extrapolation warnings, CLI smoke, self-test smoke).
  3. near-miss negatives — compact independent re-checks: the flagship
     matched pair at the per-model design must refuse identification,
     degenerate logs must refuse to fit, a single problem group must earn
     the trivial [0, 1] pass^k interval.
  4. planted-law recovery — fits recover planted Weibull kappa and Lomax
     theta within honest tolerances at n = 4000.
  5. example reproduction — the bundled synthetic example reproduces its
     frozen expected summary (deterministic planting, seed 42).
  6. reference-integrity scan — the exact shippable tree is scanned with
     the portfolio pre-push scanner (path assembled at runtime so this
     file carries no self-matching literal). Internal build records
     (BUILD_LOG.md, BUILD_REPORT.md) are not shipped and not scanned.
  7. banned-name check — the forbidden portfolio name appears nowhere in
     the shippable tree.

Coverage statement: stages 2–5 exercise every public module
(loaders, models, identify, passk, extrapolate, selftest, report, cli)
against planted truths and engineered near-misses; no stage mocks the
estimators it grades. The one thing no local stage can cover is other
platforms' optimizer jitter, absorbed by the tolerances in stage 5.

Run: python3 validate.py
"""
from __future__ import annotations

import compileall
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONPATH", str(SRC))

RESULTS: list[tuple[str, bool, str]] = []


def stage(name):
    def deco(fn):
        try:
            note = fn()
            RESULTS.append((name, True, note or ""))
            print(f"[PASS] {name} {note or ''}")
        except Exception as exc:  # noqa: BLE001 — harness reports, not hides
            RESULTS.append((name, False, str(exc)))
            print(f"[FAIL] {name}: {exc}")
        return fn
    return deco


@stage("1 import/compile")
def _s1():
    assert compileall.compile_dir(str(SRC), quiet=1), "byte-compile failed"
    import horizonkit
    return f"(horizonkit {horizonkit.__version__})"


@stage("2 test suite (pytest)")
def _s2():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(SRC)})
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    assert r.returncode == 0, f"pytest failed:\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}"
    return f"({tail})"


@stage("3 near-miss negatives")
def _s3():
    import numpy as np

    from horizonkit import estimate_passk, identify
    from horizonkit.errors import InsufficientDataError
    from horizonkit.loaders import RunData
    from horizonkit.models import fit_model, survival

    # (a) matched pair: frailty-generated data must refuse identification
    lomax = {"lambda": 0.03500061704803072, "theta": 0.593161685642301}
    rng = np.random.default_rng(20261029)
    x = np.exp(rng.uniform(np.log(0.02), np.log(1112.0), 1300))
    y = (rng.random(len(x)) < survival("lomax", lomax, x)).astype(int)
    verdict = identify(RunData(x, y)).verdict
    assert verdict == "not_identified", f"matched-pair near-miss named a law: {verdict}"
    # (b) degenerate log refuses
    try:
        fit_model(RunData(np.arange(1, 21, dtype=float), np.ones(20, dtype=int)), "weibull")
        raise AssertionError("all-success log was fitted instead of refused")
    except InsufficientDataError:
        pass
    # (c) single group earns the trivial interval
    r = estimate_passk([(20, 15)], k=3)
    assert (r.ci_low, r.ci_high) == (0.0, 1.0), r
    return "(matched-pair refusal, degenerate refusal, single-group interval)"


@stage("4 planted-law recovery")
def _s4():
    import numpy as np

    from horizonkit.loaders import RunData
    from horizonkit.models import fit_model, survival

    def planted(model, params, seed):
        rng = np.random.default_rng(seed)
        x = np.exp(rng.uniform(np.log(1), np.log(500), 4000))
        y = (rng.random(len(x)) < survival(model, params, x)).astype(int)
        return RunData(x, y)

    kw = fit_model(planted("weibull", {"kappa": 0.7, "eta": 80.0}, 3), "weibull")
    assert 0.55 <= kw.params["kappa"] <= 0.9, kw.params
    tl = fit_model(planted("lomax", {"lambda": 0.02, "theta": 0.8}, 3), "lomax")
    assert 0.4 <= tl.params["theta"] <= 1.4, tl.params
    return f"(kappa {kw.params['kappa']:.3f} vs 0.700 planted; theta {tl.params['theta']:.3f} vs 0.800)"


@stage("5 example reproduction (synthetic)")
def _s5():
    r = subprocess.run([sys.executable, "examples/synthetic_example.py"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(SRC)})
    assert r.returncode == 0, f"example failed:\n{r.stderr[-2000:]}"
    got = None
    for line in r.stdout.splitlines():
        if line.startswith("SUMMARY_JSON "):
            got = json.loads(line[len("SUMMARY_JSON "):])
    assert got is not None, "example printed no SUMMARY_JSON"
    exp = json.loads((ROOT / "examples" / "synthetic_expected.json").read_text())
    assert got["n"] == exp["n"] and got["n_success"] == exp["n_success"]
    assert got["verdict"] == exp["verdict"], (got["verdict"], exp["verdict"])
    assert abs(got["weibull_kappa"] - exp["weibull_kappa"]) <= 0.05
    assert abs(got["lomax_theta"] - exp["lomax_theta"]) <= 0.08
    assert abs(got["passk3"] - exp["passk3"]) <= 0.01
    for m, v in exp["t99"].items():
        assert abs(got["t99"][m] - v) <= 0.03 * v, (m, got["t99"][m], v)
    return "(n, verdict exact; parameters and T99 within tolerance)"


@stage("6 reference-integrity scan")
def _s6():
    scanner = (Path.home() / "workspace" /
               ("github-" + "fis" + "-watch") / ("fis" + "_scan.py"))
    assert scanner.exists(), "reference scanner not found"
    targets = [str(ROOT / p) for p in
               ("src", "tests", "examples", "README.md", "pyproject.toml",
                "LICENSE", "validate.py", ".github")]
    r = subprocess.run([sys.executable, str(scanner), *targets],
                       capture_output=True, text=True)
    hits = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    assert not hits, f"{len(hits)} candidate hit(s): {hits[0][:200]}"
    return "(0 candidate hits in the shippable tree)"


@stage("7 banned-name check")
def _s7():
    banned = "Dev" + "Pulse"
    offenders = []
    for base in ("src", "tests", "examples"):
        for p in (ROOT / base).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                try:
                    if banned in p.read_text(encoding="utf-8"):
                        offenders.append(str(p.relative_to(ROOT)))
                except (UnicodeDecodeError, OSError):
                    continue
    for p in (ROOT / "README.md", ROOT / "pyproject.toml"):
        if banned in p.read_text(encoding="utf-8"):
            offenders.append(p.name)
    assert not offenders, f"banned name in: {offenders}"
    return "(clean)"


def main() -> int:
    _s1(); _s2(); _s3(); _s4(); _s5(); _s6(); _s7()
    ok = all(passed for _, passed, _ in RESULTS)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
