# HorizonKit

**Agent reliability statistics, with the identification warning attached.**

Given run logs from your agent evaluations, HorizonKit fits the three
competing laws of task success versus horizon — constant hazard
(exponential), Weibull, and the gamma-frailty mixture (Lomax) — reports
**which of them your data can actually identify**, gives honest pass^k
intervals at small n, and extrapolates T50/T90/T99 **side by side, with
the warning printed next to the numbers**.

HorizonKit is the reference implementation for:

> Ghosh, *What Time-Horizon Curves Cannot Tell Us* (manuscript in preparation for JMLR)

The paper shows that aggregate success-versus-length curves of the kind
the field argues about (METR-style time-horizon plots) often *cannot*
distinguish a genuinely declining hazard from a population of constant
hazards mixed over tasks of different difficulty — and that this
non-identification is **one-directional**: aggregate data cannot exclude
the frailty explanation of an apparent declining hazard. HorizonKit
operationalises that result: when your aggregate record cannot separate
the laws, the tool's output is the named refusal **`not identified`**,
not a confident pick.

## Install

```bash
pip install horizonkit
```

Dependencies: NumPy and SciPy only. Pure CPU.

## Quickstart

```bash
# Fit all three laws to a run log (CSV/JSON/JSONL: horizon + success per run)
horizonkit fit runs.csv

# Which laws can this aggregate record actually identify?
horizonkit identify runs.csv

# Honest pass^k (k attempts per problem, all must succeed)
horizonkit passk runs.csv -k 3

# T50/T90/T99 under every fitted law, with divergence + tail-rule errors
horizonkit extrapolate runs.csv

# Grade the tool itself on planted laws at a realistic design
horizonkit selftest
```

Or as a library:

```python
from horizonkit import load_runs, fit_all, identify, estimate_passk
from horizonkit.extrapolate import extrapolate

data = load_runs("runs.csv")            # generic, METR-format, or HAL-format fields
fits = fit_all(data)                    # exponential / weibull / lomax + CIs
verdict = identify(data)                # "weibull" | "lomax" | "not_identified"
print(verdict.verdict, verdict.reason)  # refusals carry their reason
pk = estimate_passk(data, k=3)          # unbiased estimate + honest interval
ext = extrapolate(data)                 # per-law T50/T90/T99 + warnings
```

Input formats: one row per run with a positive horizon (task length,
minutes, tokens, steps — the unit is yours) and a binary success;
aggregate rows with `trials`/`successes` also work. METR-style files
(`human_minutes`, `score_binarized`) and HAL-style files (duration/steps
plus `success`/`score`) are recognised by field name. Group labels
(`task_id`, `problem_id`, …) feed the pass^k estimator.

## The worked example (synthetic, truth known)

`examples/synthetic_runs.csv` is **simulated** data planted from a Lomax
law (λ=0.02, θ=1.0, n=800, seed 42; regenerate with
`examples/make_synthetic.py`). Running it through the pipeline
(`python examples/synthetic_example.py`) gives the honest result this
tool exists to produce:

- identification verdict: **not identified** — at n=800 the aggregate
  record cannot separate the planted frailty law from its Weibull match;
- T99 by law: exponential **418**, Weibull **705**, Lomax **2,267** —
  a 5.4× spread on identical data, which is why no single law's tail is
  ever reported alone;
- pass^3 ≈ 0.21 with a cluster-bootstrap interval over the 40 problem
  groups.

## Self-test: the tool graded on planted laws

`horizonkit selftest` plants each law at a per-model design in the
flagship paper's regime (n=1,300 runs, horizons log-uniform over the
paper's window [0.02, 1112]; the Weibull and Lomax plantings are the
paper's matched pair, sup-norm distance 0.0257) and pushes every
replicate through the same public pipeline. Measured behaviour
(20 replicates per law, seed 20261029):

| planted truth | named correct | refused | named wrong |
|---|---:|---:|---:|
| Lomax (frailty) | 0.75 | 0.20 | 0.05 |
| Weibull (κ=0.767) | 0.70 | 0.25 | 0.05 |
| Exponential | — | 0.95 | 0.05 |

(For the exponential planting the classifier only arbitrates Weibull vs
Lomax, so "named correct" does not apply; refusal is the honest outcome,
and it refuses 19 times in 20.)

The refusal column is the feature. A tool that always names a law at
this design is wrong a large fraction of the time; HorizonKit says so
instead. Note also what the table does *not* show: at the paper's own
empirical design the refusal rates are higher still — identification is
design-relative, and this log-uniform design is somewhat more informative
than the empirical length distribution the dispute is conducted on.

## What this tool does not establish

- It does not identify a hazard law from aggregate curves when the fitted
  laws are observationally equivalent at your design; in that case the
  honest output is the refusal "not identified", not a confident pick.
- Identification here is one-directional and design-relative: a
  Weibull-looking aggregate curve cannot exclude the frailty explanation
  at per-model designs, and pooling/replication — not a better fit
  statistic — is what restores identification.
- Extrapolated T90/T99 values are law-relative. A single law's tail is
  an assumption, not a measurement; report the range across laws.
- A predictive win by one law (better held-out log-loss, lower AIC) is
  not identification of the data-generating law.
- pass^k intervals are only as real as the problem groups behind them:
  one group earns a [0, 1] interval, by design.
- No METR or HAL data is bundled or redistributed by this tool; it
  analyses files you supply, under those datasets' own licences.

## Validation

`python validate.py` runs the full harness: import/compile, the pytest
suite (including the mandatory near-miss negatives — data engineered so
Weibull and frailty are indistinguishable must refuse), planted-law
recovery, reproduction of the frozen synthetic example, a
reference-integrity scan of the shippable tree, and a banned-name check.

## License

MIT © Ranadhir Ghosh
