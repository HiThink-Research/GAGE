#!/usr/bin/env python
"""汇总 run_all_models(_mock) / run_piqa_litellm_local 产出的 summary.json。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def _collect_summaries(root: Path) -> List[Dict[str, Any]]:
    """扫描 root 下的 summary.json，返回汇总列表。"""

    items: List[Dict[str, Any]] = []
    for path in root.glob("**/summary.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        rel = path.relative_to(root)
        parts = rel.parts
        # 形如 <model>/<run_id>/summary.json，或 <run_id>/summary.json
        model = parts[0] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else "default")
        run_id = parts[-2] if len(parts) >= 2 else path.parent.name

        meta = data.get("run", {}).get("metadata", {}) if isinstance(data, dict) else {}
        backends = [b.get("backend_id") for b in meta.get("backends", []) if isinstance(b, dict) and b.get("backend_id")]
        metrics = data.get("metrics") if isinstance(data, dict) else None
        sample_count = data.get("sample_count") if isinstance(data, dict) else None

        items.append(
            {
                "model": model,
                "run_id": run_id,
                "path": str(path),
                "sample_count": sample_count,
                "metrics": metrics,
                "backends": backends,
            }
        )

    # 按模型名/修改时间排序，便于阅读
    items.sort(key=lambda x: (x["model"], x["run_id"]))
    return items


def main():
    parser = argparse.ArgumentParser(description="汇总指定根目录下的 summary.json")
    parser.add_argument("--root", type=Path, default=Path("runs/litellm_mock_matrix"), help="包含各模型子目录的根路径")
    parser.add_argument("--output", type=Path, default=None, help="输出索引文件路径（默认写入 root/summary.index.json）")
    args = parser.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    items = _collect_summaries(root)
    if not items:
        raise SystemExit(f"no summary.json found under {root}")

    output = args.output or (root / "summary.index.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"root": str(root), "count": len(items), "items": items}, ensure_ascii=False, indent=2))

    # 控制台预览
    print(f"[summary] collected {len(items)} summaries under {root}")
    for item in items:
        label = item["model"]
        run_id = item["run_id"]
        samples = item.get("sample_count")
        print(f"- {label:<24} run={run_id:<12} samples={samples!s:<4} -> {item['path']}")
    print(f"[summary] index written to {output}")


if __name__ == "__main__":
    main()
