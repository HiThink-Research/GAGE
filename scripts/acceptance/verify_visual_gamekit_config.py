#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gage_eval.tools.gamekit_acceptance import verify_visual_gamekit_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run config-driven arena_visual acceptance for a GameKit config."
    )
    parser.add_argument("--config", required=True, help="Path to a GameKit PipelineConfig YAML.")
    parser.add_argument(
        "--expect-plugin",
        required=True,
        help="Expected arena_visual plugin id for the config.",
    )
    parser.add_argument("--gpus", type=int, default=0, help="GPU count for the local resource profile.")
    parser.add_argument("--cpus", type=int, default=1, help="CPU count for the local resource profile.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs" / "final_acceptance"),
        help="Directory that will receive run artifacts.",
    )
    parser.add_argument("--run-id", help="Optional run identifier override.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Supported for parity with run.py; only value 1 is accepted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_visual_gamekit_config(
            args.config,
            expect_plugin=args.expect_plugin,
            output_dir=args.output_dir,
            run_id=args.run_id,
            gpus=args.gpus,
            cpus=args.cpus,
            max_samples=args.max_samples,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[gage-eval] visual acceptance failed: {exc}", file=sys.stderr)
        return 1

    probe = result.browser_probe
    summary = {
        "config": str(Path(args.config).expanduser()),
        "run_id": result.run_id,
        "viewer_url": None if probe is None else probe.viewer_url,
        "plugin_id": None if probe is None else probe.session_payload.get("pluginId"),
        "first_seq": None if probe is None else probe.first_scene.get("seq"),
        "last_seq": None if probe is None else probe.last_scene.get("seq"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
