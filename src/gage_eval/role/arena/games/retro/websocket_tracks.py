"""WebSocket helpers for retro game streaming."""

from __future__ import annotations

import io
import json
from typing import Any, Mapping, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


def parse_key_payload(message: object) -> Optional[Mapping[str, object]]:
    """Parse a keyboard payload sent over WebSocket.

    Args:
        message: WebSocket message (text or bytes-like).

    Returns:
        Parsed mapping payload, or None if parsing failed.
    """

    if isinstance(message, bytes):
        text = message.decode("utf-8", errors="ignore")
    else:
        text = str(message)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def encode_jpeg(
    frame: Any,
    *,
    quality: int = 80,
) -> tuple[Optional[bytes], Optional[str]]:
    """Encode an RGB frame into JPEG bytes.

    Args:
        frame: RGB frame as a numpy array.
        quality: JPEG quality setting (1-95).

    Returns:
        Tuple of (jpeg_bytes, error_message).
    """

    if Image is None:
        return None, "pillow_missing"
    if np is None:
        return None, "numpy_missing"
    if not hasattr(frame, "shape"):
        return None, "frame_missing_shape"

    try:
        # STEP 1: Build the base image from the RGB array.
        image = Image.fromarray(frame, mode="RGB")
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"jpeg_encode_failed:{exc}"

    # STEP 2: Serialize the image to JPEG bytes.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return buffer.getvalue(), None


__all__ = ["encode_jpeg", "parse_key_payload"]
