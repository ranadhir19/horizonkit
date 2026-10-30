# HorizonKit — hazard-law fits

n = 800 runs, 474 successes (0.593), horizon range [1.008, 596.6], source = generic

| model | parameters | log-lik | AIC | BIC |
|---|---|---:|---:|---:|
| exponential | lambda=0.011028 | -351.900 | 705.801 | 710.485 |
| weibull | kappa=0.746836, eta=91.2746 | -338.233 | 680.466 | 689.835 |
| lomax | lambda=0.0196016, theta=0.774571 | -335.792 | 675.584 | 684.954 |

Parameter intervals (log-scale Wald, from the observed information):

- exponential: lambda [0.009696, 0.01254]
- weibull: kappa [0.6635, 0.8406], eta [77.43, 107.6]
- lomax: lambda [0.01498, 0.02564], theta [0.4731, 1.268]

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


# HorizonKit — identification

n = 800 runs, 474 successes (0.593), horizon range [1.008, 596.6], source = generic

**Verdict: not_identified**

neither law is rejected by GOF (Weibull p=0.228, Lomax p=0.556) and their fitted curves differ by at most 0.048 in success probability over the observed window; the aggregate record cannot distinguish them.

Weibull GOF p = 0.2277; Lomax GOF p = 0.5565; sup-norm distance between fitted curves = 0.0481; alpha = 0.05.

Disclosures:
- One-directional identification: at a per-model aggregate design, aggregate curves cannot exclude the frailty (Lomax) explanation of an apparent declining hazard; declining-hazard data can exclude the frailty law. A Weibull-looking aggregate curve is therefore weak evidence about the underlying hazard.
- Design-relativity: identification is a property of this design (sample size, horizon window, replication), not of the laws alone. Pooled or replicated designs can separate laws that this aggregate record cannot.

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


# HorizonKit — pass^k

pass^3 estimate = 0.2115 (95% interval [0.1739, 0.2492])
status = ok; groups used = 40 of 40; excluded = 0.

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


# HorizonKit — T50/T90/T99 extrapolation

n = 800 runs, 474 successes (0.593), horizon range [1.008, 596.6], source = generic

| law | T50 | T90 | T99 | T99/T50 |
|---|---:|---:|---:|---:|
| exponential | 62.8536 | 208.795 | 417.59 | 6.644 |
| weibull | 55.8751 | 278.837 | 705.381 | 12.624 |
| lomax | 46.8083 | 326.076 | 2266.47 | 48.420 |

Tail rule-of-thumb relative errors (rule / fitted ratio - 1):

- exponential_implied: exponential: +0.0%, weibull: -47.4%, lomax: -86.3%
- circulating_170x: exponential: +2458.8%, weibull: +1246.6%, lomax: +251.1%

Warnings:
- IDENTIFICATION WARNING: the aggregate record does not identify the law (neither law is rejected by GOF (Weibull p=0.228, Lomax p=0.556) and their fitted curves differ by at most 0.048 in success probability over the observed window; the aggregate record cannot distinguish them.) Extrapolated horizons below are law-relative, not measurements; report the range across laws, never a single law's tail.
- Material divergence at T50, T90, T99: the fitted laws disagree by more than 1.25× on the same data.
- Tail rules of thumb are law-relative: fitted T99/T50 spans 6.64×–48.42× across laws (7.3× spread). The exponential-implied 6.64× and the circulating 170× rule each carry the per-law relative errors tabulated in this result; neither is a law-free constant.

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

SUMMARY_JSON {"lomax_theta": 0.775, "n": 800, "n_success": 474, "passk3": 0.2115, "t99": {"exponential": 417.6, "lomax": 2266.5, "weibull": 705.4}, "verdict": "not_identified", "weibull_kappa": 0.747}
