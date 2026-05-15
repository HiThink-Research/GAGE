#!/usr/bin/env python
"""Dry-run helper for inspecting report pack rollout state.

This script intentionally does not modify source files or run directories. It is
kept as an operational checklist entry for reverting the ReportStep split by
feature flag: use `GAGE_EVAL_REPORT_PACK=0` or `run.py --no-report-pack` to keep
legacy summary-only output while preserving the new code paths for review.
"""

from __future__ import annotations

import argparse
from pathlib import Path


STAGE_GUIDES = {
    "04a-facade": [
        "disable report pack with GAGE_EVAL_REPORT_PACK=0",
        "verify legacy summary.json is still written by ReportStep.finalize",
    ],
    "04b-collectors": [
        "restore runtime health and metric collection to the ReportStep facade",
        "verify summary.runtime_health counts against samples.jsonl",
    ],
    "04c-context": [
        "bypass ReportContextBuilder and SummaryExtensionRunner",
        "verify summary generator legacy payloads still merge into summary.json",
    ],
    "04d-writer": [
        "restore direct EvalCache.write_summary from ReportStep",
        "verify report_pack diagnostics are absent when pack is disabled",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect report pack rollout state without mutating files.")
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_GUIDES),
        help="ReportStep split stage to inspect.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print supported rollback inspection stages.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional run directory to inspect for report_pack output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_stages:
        for stage in STAGE_GUIDES:
            print(stage)
        return

    print("rollback mode: dry-run only")
    print("disable flag: GAGE_EVAL_REPORT_PACK=0")
    print("CLI flag: run.py --no-report-pack")
    if args.stage:
        print(f"stage: {args.stage}")
        print("stage checklist:")
        for item in STAGE_GUIDES[args.stage]:
            print(f"- {item}")
    if args.run_dir:
        pack = args.run_dir / "report_pack"
        print(f"run_dir: {args.run_dir}")
        print(f"report_pack_exists: {pack.exists()}")
        print(f"summary_exists: {(args.run_dir / 'summary.json').exists()}")


if __name__ == "__main__":
    main()
