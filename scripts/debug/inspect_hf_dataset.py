#!/usr/bin/env python3
"""Inspect an HF dataset schema and print a compact JSON report."""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _build_report(records: Iterable[dict[str, Any]], *, hub_id: str, split: str, config: str | None, revision: str | None) -> dict[str, Any]:
    sample_count = 0
    key_union: set[str] = set()
    problem_statement_present = 0
    problem_statement_chars = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        sample_count += 1
        key_union.update(record.keys())
        text = _coerce_text(record.get("problem_statement"))
        if text:
            problem_statement_present += 1
            problem_statement_chars += len(text)

    return {
        "hub_id": hub_id,
        "split": split,
        "config": config,
        "revision": revision,
        "sample_count": sample_count,
        "key_union": sorted(key_union),
        "problem_statement_present_ratio": (
            problem_statement_present / sample_count if sample_count else 0.0
        ),
        "problem_statement_avg_chars": (
            problem_statement_chars / problem_statement_present if problem_statement_present else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hub_id", help="HF dataset id, e.g. SWE-bench/SWE-bench")
    parser.add_argument("--split", default="train", help="Dataset split to inspect")
    parser.add_argument("--config", default=None, help="Optional dataset config name")
    parser.add_argument("--revision", default=None, help="Optional dataset revision")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for quick sampling")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        args.hub_id,
        name=args.config,
        split=args.split,
        revision=args.revision,
    )
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    report = _build_report(
        dataset,
        hub_id=args.hub_id,
        split=args.split,
        config=args.config,
        revision=args.revision,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
