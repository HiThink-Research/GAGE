"""Utilities for capturing and persisting replay frame artifacts."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

_FRAME_RGB_KEYS = ("_rgb", "rgb", "rgb_array", "frame_rgb")
_FRAME_INTERNAL_KEYS = {"_rgb"}


class FrameCaptureRecorder:
    """Capture replay frame events and persist RGB images when available."""

    def __init__(
        self,
        *,
        replay_dir: Path,
        frame_dir_name: str = "frames",
        enabled: bool = False,
        frame_stride: int = 1,
        max_frames: int = 0,
        image_format: str = "jpeg",
        jpeg_quality: int = 75,
        include_frame_snapshot: bool = True,
    ) -> None:
        """Initialize frame capture options.

        Args:
            replay_dir: Replay directory (`runs/<run_id>/replays/<sample_id>`).
            frame_dir_name: Relative frame artifact directory under replay_dir.
            enabled: Whether frame capture is enabled.
            frame_stride: Capture every N frames.
            max_frames: Max number of encoded image files (0 means unlimited).
            image_format: Encoded image format (`jpeg` or `png`).
            jpeg_quality: JPEG quality (1-95).
            include_frame_snapshot: Whether to include sanitized frame snapshot in events.
        """

        self._replay_dir = Path(replay_dir).expanduser().resolve()
        self._frame_dir_name = str(frame_dir_name or "frames").strip() or "frames"
        self._enabled = bool(enabled)
        self._frame_stride = max(1, int(frame_stride))
        self._max_frames = max(0, int(max_frames))
        normalized_format = str(image_format or "jpeg").strip().lower()
        if normalized_format in {"jpg", "jpeg"}:
            normalized_format = "jpeg"
        if normalized_format not in {"jpeg", "png"}:
            normalized_format = "jpeg"
        self._image_format = normalized_format
        self._jpeg_quality = max(1, min(95, int(jpeg_quality)))
        self._include_frame_snapshot = bool(include_frame_snapshot)

        self._capture_attempts = 0
        self._saved_images = 0
        self._events: list[dict[str, Any]] = []

    def capture(
        self,
        frame_payload: Any,
        *,
        step: int,
        actor: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Capture one replay frame event from a runtime frame payload.

        Args:
            frame_payload: Runtime frame payload from environment `get_last_frame()`.
            step: Logical step index for this capture.
            actor: Optional actor/player identifier.
            force: Whether to bypass frame stride checks.
        """

        if not self._enabled:
            return

        # STEP 1: Apply capture stride policy.
        self._capture_attempts += 1
        if not force and (self._capture_attempts - 1) % self._frame_stride != 0:
            return

        # STEP 2: Build event payload with sanitized frame snapshot.
        event: dict[str, Any] = {
            "type": "frame",
            "ts_ms": int(time.time() * 1000),
            "step": int(step),
            "actor": str(actor or "unknown"),
        }
        if self._include_frame_snapshot:
            event["frame"] = _sanitize_frame_snapshot(frame_payload)

        # STEP 3: Persist RGB frame image when available and within limits.
        rgb_frame = _extract_rgb_frame(frame_payload)
        if rgb_frame is not None:
            if self._max_frames > 0 and self._saved_images >= self._max_frames:
                event["meta"] = {"image_skipped": "max_frames_reached"}
            else:
                image_bytes, error, width, height = _encode_image_bytes(
                    rgb_frame,
                    image_format=self._image_format,
                    jpeg_quality=self._jpeg_quality,
                )
                if image_bytes is None:
                    event["meta"] = {"image_error": error or "image_encode_failed"}
                else:
                    image_rel_path = self._save_image_bytes(image_bytes=image_bytes)
                    event["image"] = {
                        "path": image_rel_path,
                        "format": self._image_format,
                        "width": width,
                        "height": height,
                    }

        self._events.append(event)

    def build_frame_events(self) -> list[dict[str, Any]]:
        """Return captured frame events."""

        return [dict(event) for event in self._events]

    def _save_image_bytes(self, *, image_bytes: bytes) -> str:
        frame_dir = self._replay_dir / self._frame_dir_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        extension = "jpg" if self._image_format == "jpeg" else "png"
        filename = f"frame_{self._saved_images:06d}.{extension}"
        output_path = frame_dir / filename
        output_path.write_bytes(image_bytes)
        self._saved_images += 1
        return f"{self._frame_dir_name}/{filename}"


def _sanitize_frame_snapshot(frame_payload: Any) -> Any:
    """Convert frame payload into a compact JSON-serializable snapshot."""

    if not isinstance(frame_payload, Mapping):
        return _json_safe_value(frame_payload)
    snapshot: dict[str, Any] = {}
    for key, value in frame_payload.items():
        key_text = str(key)
        if key_text in _FRAME_INTERNAL_KEYS:
            continue
        snapshot[key_text] = _json_safe_value(value)
    return snapshot


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"frame_type": "array", "shape": _normalize_shape(shape)}
    if isinstance(value, Mapping):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _extract_rgb_frame(frame_payload: Any) -> Optional[Any]:
    if isinstance(frame_payload, Mapping):
        for key in _FRAME_RGB_KEYS:
            candidate = frame_payload.get(key)
            if candidate is not None:
                return candidate
        return None
    return frame_payload


def _normalize_shape(shape: Any) -> list[int]:
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return []


def _encode_image_bytes(
    frame: Any,
    *,
    image_format: str,
    jpeg_quality: int,
) -> tuple[Optional[bytes], Optional[str], Optional[int], Optional[int]]:
    """Encode an RGB frame into image bytes.

    Args:
        frame: RGB frame payload.
        image_format: Target image format (`jpeg` or `png`).
        jpeg_quality: JPEG quality.

    Returns:
        Tuple of (encoded_bytes, error, width, height).
    """

    if Image is None:
        return None, "pillow_missing", None, None
    if frame is None:
        return None, "frame_missing", None, None

    candidate = frame
    if np is not None and isinstance(candidate, (list, tuple)):
        try:
            candidate = np.asarray(candidate)
        except Exception:
            return None, "frame_asarray_failed", None, None
    if np is not None and hasattr(candidate, "dtype"):
        try:
            if str(candidate.dtype) != "uint8":
                candidate = np.asarray(candidate)
                candidate = np.clip(candidate, 0, 255).astype("uint8")
        except Exception:
            return None, "frame_dtype_normalize_failed", None, None

    if not hasattr(candidate, "shape"):
        return None, "frame_missing_shape", None, None
    shape = getattr(candidate, "shape", None)
    dims = _normalize_shape(shape)
    if len(dims) == 2:
        pass
    elif len(dims) == 3 and dims[2] in {1, 3, 4}:
        pass
    else:
        return None, "frame_invalid_shape", None, None

    try:
        image = Image.fromarray(candidate).convert("RGB")
    except Exception:
        return None, "image_from_array_failed", None, None

    width, height = image.size
    buffer = io.BytesIO()
    try:
        if image_format == "png":
            image.save(buffer, format="PNG")
        else:
            image.save(buffer, format="JPEG", quality=int(jpeg_quality), optimize=True)
    except Exception:
        return None, "image_encode_failed", None, None
    return buffer.getvalue(), None, int(width), int(height)
