# HorizonKit — planted-law self-test

design: n = 1300 runs/replicate over the flagship window, 20 replicates per planted law

| planted truth | named correct | refused | named wrong | verdict counts |
|---|---:|---:|---:|---|
| lomax | 0.75 | 0.20 | 0.05 | lomax: 15, not_identified: 4, weibull: 1 |
| weibull | 0.70 | 0.25 | 0.05 | lomax: 1, not_identified: 5, weibull: 14 |
| exponential | 0.00 | 0.95 | 0.05 | lomax: 1, not_identified: 19 |

## What this tool does not establish

- It does not identify a hazard law from aggregate curves when the fitted
  laws are observationally equivalent at your design; in that case the
  honest output is the refusal "not identified", not a confident pick.
- Identification here is one-directional and design-relative (see the
  disclosures in every identification report): a Weibull-looking
  aggregate curve cannot exclude the frailty explanation at per-model
  designs, and pooling/replication — not a better fit statistic — is
  what restores identification.
- Extrapolated T90/T99 values are law-relative. A single law's tail is
  an assumption, not a measurement; report the range across laws.
- A predictive win by one law (better held-out log-loss, lower AIC) is
  not identification of the data-generating law.
- pass^k intervals are only as real as the problem groups behind them:
  one group earns a [0, 1] interval, by design.
- No METR or HAL data is bundled or redistributed by this tool; it
  analyses files you supply, under those datasets' own licences.

