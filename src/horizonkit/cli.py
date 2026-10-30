"""Command line interface: ``horizonkit fit|identify|passk|extrapolate|selftest``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .errors import HorizonKitError
from .extrapolate import extrapolate
from .identify import identify
from .loaders import load_runs
from .models import fit_all
from .passk import estimate_passk
from .report import (extrapolate_report, fit_report, identify_report,
                     passk_report, selftest_report)
from .selftest import run_selftest


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    print(text)


def _add_file_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="run log (CSV, JSON, or JSONL)")
    p.add_argument("--format", default="auto",
                   choices=["auto", "generic", "metr", "hal"],
                   help="field convention of the run log (default: auto)")
    p.add_argument("--out", default=None, help="also write the report to this Markdown file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="horizonkit",
        description="Agent reliability statistics: hazard-law fits, "
                    "identification verdicts, honest pass^k, and "
                    "extrapolation with the warning attached.")
    parser.add_argument("--version", action="version", version=f"horizonkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="fit exponential / Weibull / Lomax laws")
    _add_file_args(p)
    p = sub.add_parser("identify", help="which laws can this aggregate record identify?")
    _add_file_args(p)
    p = sub.add_parser("passk", help="honest pass^k from repeated attempts per problem")
    _add_file_args(p)
    p.add_argument("-k", type=int, required=True, help="all k attempts must succeed")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("extrapolate", help="T50/T90/T99 under every fitted law, side by side")
    _add_file_args(p)
    p = sub.add_parser("selftest", help="planted-law simulation through the full pipeline")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--n", type=int, default=1300)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "selftest":
            _emit(selftest_report(run_selftest(reps=args.reps, n=args.n, seed=args.seed)), args.out)
            return 0
        data = load_runs(args.file, format=args.format)
        if args.command == "fit":
            _emit(fit_report(data, fit_all(data)), args.out)
        elif args.command == "identify":
            _emit(identify_report(data, identify(data)), args.out)
        elif args.command == "passk":
            _emit(passk_report(estimate_passk(data, args.k,
                                              n_bootstrap=args.bootstrap,
                                              seed=args.seed)), args.out)
        elif args.command == "extrapolate":
            _emit(extrapolate_report(data, extrapolate(data)), args.out)
        return 0
    except HorizonKitError as exc:
        print(f"horizonkit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
