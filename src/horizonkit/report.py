"""Markdown/terminal reports with a fixed honesty box.

Every HorizonKit report carries the same box.  It is not decoration: it is
the set of claims the tool refuses to make, printed next to every number
the tool does produce.
"""
from __future__ import annotations

from typing import Mapping

from .identify import IdentificationResult
from .loaders import RunData
from .models import FitResult
from .passk import PassKResult
from .extrapolate import ExtrapolationResult
from .selftest import SelfTestResult

HONESTY_BOX = """## What this tool does not establish

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
"""


def data_summary(data: RunData) -> str:
    return (f"n = {data.n} runs, {data.n_success} successes "
            f"({data.success_rate:.3f}), horizon range "
            f"[{data.horizons.min():.4g}, {data.horizons.max():.4g}], "
            f"source = {data.source_format}")


def fit_report(data: RunData, fits: Mapping[str, FitResult]) -> str:
    lines = ["# HorizonKit — hazard-law fits", "", data_summary(data), "",
             "| model | parameters | log-lik | AIC | BIC |",
             "|---|---|---:|---:|---:|"]
    for model, fit in fits.items():
        params = ", ".join(f"{k}={v:.6g}" for k, v in fit.params.items())
        lines.append(f"| {model} | {params} | {fit.loglik:.3f} | {fit.aic:.3f} | {fit.bic:.3f} |")
    lines += ["", "Parameter intervals (log-scale Wald, from the observed information):", ""]
    for model, fit in fits.items():
        if fit.param_cis:
            cis = ", ".join(f"{k} [{lo:.4g}, {hi:.4g}]" for k, (lo, hi) in fit.param_cis.items())
            lines.append(f"- {model}: {cis}")
        else:
            lines.append(f"- {model}: intervals unavailable (singular information matrix)")
    lines += ["", HONESTY_BOX]
    return "\n".join(lines)


def identify_report(data: RunData, result: IdentificationResult) -> str:
    lines = ["# HorizonKit — identification", "", data_summary(data), "",
             f"**Verdict: {result.verdict}**", "", result.reason, "",
             f"Weibull GOF p = {result.weibull_gof_p:.4g}; "
             f"Lomax GOF p = {result.lomax_gof_p:.4g}; "
             f"sup-norm distance between fitted curves = {result.sup_norm_distance:.4f}; "
             f"alpha = {result.alpha}.", "", "Disclosures:"]
    lines += [f"- {d}" for d in result.disclosures]
    lines += ["", HONESTY_BOX]
    return "\n".join(lines)


def passk_report(result: PassKResult) -> str:
    lines = ["# HorizonKit — pass^k", "",
             f"pass^{result.k} estimate = {result.estimate:.4f} "
             f"({result.level:.0%} interval [{result.ci_low:.4f}, {result.ci_high:.4f}])",
             f"status = {result.status}; groups used = {result.n_groups_used} "
             f"of {result.n_groups}; excluded = {result.n_groups_excluded}."]
    if result.warnings:
        lines += ["", "Warnings:"] + [f"- {w}" for w in result.warnings]
    lines += ["", HONESTY_BOX]
    return "\n".join(lines)


def extrapolate_report(data: RunData, result: ExtrapolationResult) -> str:
    lines = ["# HorizonKit — T50/T90/T99 extrapolation", "", data_summary(data), "",
             "| law | T50 | T90 | T99 | T99/T50 |", "|---|---:|---:|---:|---:|"]
    for model, hz in result.horizons.items():
        lines.append(f"| {model} | {hz['T50']:.6g} | {hz['T90']:.6g} | "
                     f"{hz['T99']:.6g} | {result.ratios_t99_t50[model]:.3f} |")
    lines += ["", "Tail rule-of-thumb relative errors (rule / fitted ratio - 1):", ""]
    for rule, per_model in result.rule_errors.items():
        errs = ", ".join(f"{m}: {e:+.1%}" for m, e in per_model.items())
        lines.append(f"- {rule}: {errs}")
    if result.warnings:
        lines += ["", "Warnings:"] + [f"- {w}" for w in result.warnings]
    lines += ["", HONESTY_BOX]
    return "\n".join(lines)


def selftest_report(result: SelfTestResult) -> str:
    lines = ["# HorizonKit — planted-law self-test", "",
             f"design: n = {result.n_per_rep} runs/replicate over the flagship "
             f"window, {result.reps} replicates per planted law", "",
             "| planted truth | named correct | refused | named wrong | verdict counts |",
             "|---|---:|---:|---:|---|"]
    for row in result.rows:
        counts = ", ".join(f"{k}: {v}" for k, v in row["verdict_counts"].items())
        lines.append(f"| {row['truth']} | {row['named_correct']:.2f} | "
                     f"{row['refused']:.2f} | {row['named_wrong']:.2f} | {counts} |")
    lines += ["", HONESTY_BOX]
    return "\n".join(lines)
