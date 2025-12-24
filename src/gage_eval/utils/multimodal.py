"""Utilities for loading and normalizing multimodal inputs."""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any, Optional

from PIL import Image

try:  # pragma: no cover - optional dependency
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None


IMAGE_FACTOR = 28
# NOTE: The default minimum pixel area is aligned with the minimum-side=28
# constraint used by Qwen-VL reference implementations.
MIN_PIXELS = 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def load_multimodal_data(
    processor: Any,
    image: list[str] | None = None,
    audio: list[str] | None = None,
    return_dict: bool = False,
    *,
    resize_opts: Optional[dict[str, Any]] = None,
) -> Any:
    """Loads and normalizes multimodal inputs (image/audio) for model backends.

    Args:
        processor: Backend tokenizer/processor that may include audio feature settings.
        image: A list of image sources. Each element can be a local file path or a
            `data:` URL.
        audio: A list of audio file paths.
        return_dict: Whether to return a dict `{image: ..., audio: ...}` instead of
            a tuple `(image, audio)`.
        resize_opts: Optional resizing overrides. Values can also be set by env vars:
            - `GAGE_EVAL_IMAGE_MIN_PIXELS`
            - `GAGE_EVAL_IMAGE_MAX_PIXELS`
            - `GAGE_EVAL_IMAGE_FACTOR`

    Returns:
        Either a `(image, audio)` tuple or a `{image, audio}` dict depending on
        `return_dict`.

    Raises:
        RuntimeError: If `audio` is provided but `librosa` is not installed.
    """

    # STEP 1: Resolve resize policy (args override env override defaults).
    resize_opts = resize_opts or {}
    min_pixels = int(
        resize_opts.get("min_pixels")
        or _env_int("GAGE_EVAL_IMAGE_MIN_PIXELS")
        or MIN_PIXELS
    )
    max_pixels = int(
        resize_opts.get("max_pixels")
        or _env_int("GAGE_EVAL_IMAGE_MAX_PIXELS")
        or MAX_PIXELS
    )
    factor = int(
        resize_opts.get("factor")
        or _env_int("GAGE_EVAL_IMAGE_FACTOR")
        or IMAGE_FACTOR
    )

    if image is not None:
        # STEP 2: Load images from paths or data URLs.
        image_objs = []
        for src in image:
            if src is None:
                continue
            if isinstance(src, str) and src.startswith("data:"):
                # NOTE: data URL -> base64 decode.
                try:
                    _, b64 = src.split(",", 1)
                    binary = base64.b64decode(b64)
                    image_objs.append(Image.open(io.BytesIO(binary)))
                    continue
                except Exception:
                    # Fall back to the "treat as path" logic.
                    pass
            if isinstance(src, str):
                image_objs.append(Image.open(src))
            elif isinstance(src, Image.Image):
                image_objs.append(src)
        image = [to_rgb(x) for x in image_objs]

        # STEP 3: Resize for model compatibility and stable throughput.
        # NOTE: Some model families (for example Qwen2-VL / Qwen2.5-VL) are
        # sensitive to very small images. Enforcing `MIN_PIXELS` avoids edge
        # cases where the processor produces invalid shapes.
        # TODO(team): Tune resize defaults per model/dataset if needed.
        image = [resize_image(x, min_pixels=min_pixels, max_pixels=max_pixels, factor=factor) for x in image]

    if audio is not None:
        # STEP 4: Load audio as a waveform aligned to the processor sampling rate.
        if librosa is None:
            raise RuntimeError("librosa is required to process audio inputs")
        audio = [
            librosa.load(x, sr=processor.feature_extractor.sampling_rate)[0]
            for x in audio
        ]

    # STEP 5: Return outputs in the requested container format.
    if return_dict:
        return {k: v for k, v in [('image', image), ('audio', audio)] if v is not None}
    else:
        return image, audio


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Converts an image to RGB and flattens alpha channels against white."""

    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask.
        return white_background
    return pil_image.convert("RGB")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    image: Image.Image,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    factor: int | None = None,
) -> Image.Image:
    """Resizes an image using the `smart_resize` policy."""

    width, height = image.size
    min_pixels = MIN_PIXELS if min_pixels is None else int(min_pixels)
    if min_pixels <= 0:
        return image
    max_pixels = MAX_PIXELS if max_pixels is None else int(max_pixels)
    factor = IMAGE_FACTOR if factor is None else int(factor)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
