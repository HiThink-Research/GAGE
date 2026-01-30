"""ScreenSpot(-like) GUI grounding metrics.

This implements the "point ∈ bbox" scoring used by the original ScreenSpot eval scripts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, List

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QWEN_BOX_RE = re.compile(
    r"<\|box_start\|>\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*"
    r"\(?\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>",
    flags=re.IGNORECASE,
)
_LABELED_BBOX_RE = re.compile(
    r"\bx1\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b[\s,;]*"
    r"\by1\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b[\s,;]*"
    r"\bx2\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b[\s,;]*"
    r"\by2\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_CLICK_RE = re.compile(
    r"\bClick\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)",
    flags=re.IGNORECASE,
)


def _as_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for v in value:
            try:
                out.append(float(v))
            except Exception:
                return None
        return out
    return None


def _normalize_bbox(
    bbox: List[float],
    *,
    img_size: Optional[List[float]] = None,
) -> Optional[List[float]]:
    """Normalize bbox to [0,1] if it looks like pixels or 0..1000."""
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox

    # If already normalized.
    if all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    # Qwen-VL style often uses 0..1000 coordinates.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0 and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000.0:
        x1n, y1n, x2n, y2n = [v / 1000.0 for v in (x1, y1, x2, y2)]
        return [min(x1n, x2n), min(y1n, y2n), max(x1n, x2n), max(y1n, y2n)]

    # Pixel coordinates with known image size.
    if img_size and len(img_size) == 2 and img_size[0] and img_size[1]:
        w, h = img_size[0], img_size[1]
        x1n, y1n, x2n, y2n = [x1 / w, y1 / h, x2 / w, y2 / h]
        return [min(x1n, x2n), min(y1n, y2n), max(x1n, x2n), max(y1n, y2n)]

    return None


def _parse_pred_point_and_bbox(text: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Parse model output text and return (point, bbox) in *raw* coordinates.

    - If bbox is available, point is computed as bbox center.
    - If only 2 numbers are found, treat them as a point.
    - If 4 numbers are found, treat them as a bbox (x1,y1,x2,y2).
    """
    if not text:
        return None, None

    # 1) Labeled bbox like:
    # x1: 1000
    # y1: 1000
    # x2: 1200
    # y2: 1200
    m = _LABELED_BBOX_RE.search(text)
    if m:
        x1, y1, x2, y2 = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
        bbox = [x1, y1, x2, y2]
        point = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        return point, bbox

    # 2) Click(x, y)
    m = _CLICK_RE.search(text)
    if m:
        return [float(m.group(1)), float(m.group(2))], None

    # 3) Qwen-style <|box_start|> ... <|box_end|>
    m = _QWEN_BOX_RE.search(text)
    if m:
        nums = [float(m.group(i)) for i in range(1, 5)]
        x1, y1, x2, y2 = nums
        bbox = [x1, y1, x2, y2]
        point = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        return point, bbox

    nums = [float(x) for x in _NUM_RE.findall(text)]
    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[0], nums[1], nums[2], nums[3]
        bbox = [x1, y1, x2, y2]
        point = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        return point, bbox
    if len(nums) >= 2:
        return [nums[0], nums[1]], None
    return None, None


@registry.asset(
    "metrics",
    "screenspot_point_in_bbox",
    desc="ScreenSpot-style GUI grounding accuracy (point inside GT bbox)",
    tags=("vision", "gui-grounding", "screenspot"),
    default_aggregation="mean",
)
class ScreenSpotPointInBboxMetric(SimpleMetric):
    """Score 1.0 if predicted point lies inside GT bbox, else 0.0.

    Expected fields:
    - GT bbox: sample.metadata.bbox (either normalized [0,1] or pixel coords or 0..1000)
    - GT image size (optional for pixel bbox): sample.metadata.img_size as [W, H]
    - Prediction text: model_output.answer
    """

    def compute_value(self, context: MetricContext) -> tuple[float, Dict[str, Any]]:
        gt_bbox_raw = _as_float_list(context.get("sample.metadata.bbox"))
        img_size_raw = _as_float_list(context.get("sample.metadata.img_size"))
        pred_text = context.get("model_output.answer") or ""

        meta: Dict[str, Any] = {
            "prediction_text": (str(pred_text)[:500] if pred_text is not None else ""),
            "gt_bbox_raw": gt_bbox_raw,
            "gt_img_size": img_size_raw,
        }

        if not gt_bbox_raw:
            meta["reason"] = "missing_gt_bbox(sample.metadata.bbox)"
            return 0.0, meta

        gt_bbox = _normalize_bbox(gt_bbox_raw, img_size=img_size_raw)
        if not gt_bbox:
            meta["reason"] = "invalid_gt_bbox"
            return 0.0, meta
        meta["gt_bbox_norm"] = gt_bbox

        pred_point_raw, pred_bbox_raw = _parse_pred_point_and_bbox(str(pred_text))
        meta["pred_point_raw"] = pred_point_raw
        meta["pred_bbox_raw"] = pred_bbox_raw

        # Normalize prediction if possible.
        pred_bbox_norm = _normalize_bbox(pred_bbox_raw, img_size=img_size_raw) if pred_bbox_raw else None
        if pred_bbox_norm:
            px = (pred_bbox_norm[0] + pred_bbox_norm[2]) / 2.0
            py = (pred_bbox_norm[1] + pred_bbox_norm[3]) / 2.0
            pred_point_norm = [px, py]
        else:
            # Try interpreting point directly (normalized / 0..1000 / pixels).
            pred_point_norm = None
            if pred_point_raw and len(pred_point_raw) == 2:
                x, y = pred_point_raw
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    pred_point_norm = [x, y]
                elif abs(x) <= 1000.0 and abs(y) <= 1000.0:
                    pred_point_norm = [x / 1000.0, y / 1000.0]
                elif img_size_raw and len(img_size_raw) == 2 and img_size_raw[0] and img_size_raw[1]:
                    pred_point_norm = [x / img_size_raw[0], y / img_size_raw[1]]

        meta["pred_bbox_norm"] = pred_bbox_norm
        meta["pred_point_norm"] = pred_point_norm

        if not pred_point_norm:
            meta["reason"] = "could_not_parse_prediction_point_or_bbox"
            return 0.0, meta

        x1, y1, x2, y2 = gt_bbox
        px, py = pred_point_norm
        inside = (x1 <= px <= x2) and (y1 <= py <= y2)
        meta["inside"] = inside
        return (1.0 if inside else 0.0), meta

