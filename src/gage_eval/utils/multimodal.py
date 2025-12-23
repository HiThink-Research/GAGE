import math
import os
import base64
import io
from PIL import Image

try:  # pragma: no cover - optional dependency
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None


IMAGE_FACTOR = 28
# 默认最小像素面积仅对应最小边 28（对齐参考实现对 Qwen VL 的最小边要求）
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
    processor,
    image: list[str] | None = None,
    audio: list[str] | None = None,
    return_dict=False,
    *,
    resize_opts: dict | None = None,
):
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
        image_objs = []
        for src in image:
            if src is None:
                continue
            if isinstance(src, str) and src.startswith("data:"):
                # data URL -> base64 解码
                try:
                    _, b64 = src.split(",", 1)
                    binary = base64.b64decode(b64)
                    image_objs.append(Image.open(io.BytesIO(binary)))
                    continue
                except Exception:
                    # 回落到原逻辑尝试 as path
                    pass
            if isinstance(src, str):
                image_objs.append(Image.open(src))
            elif isinstance(src, Image.Image):
                image_objs.append(src)
        image = [to_rgb(x) for x in image_objs]
        # For Qwen2-VL, Qwen2.5-VL, Omni, 解决processor短边小于28的bug
        # OCRBench数据集需要调整，MIN_PIXELS
        # TODO:需要根据模型和数据集类型调整
        image = [resize_image(x, min_pixels=min_pixels, max_pixels=max_pixels, factor=factor) for x in image]

    if audio is not None:
        if librosa is None:
            raise RuntimeError("librosa is required to process audio inputs")
        audio = [
            librosa.load(x, sr=processor.feature_extractor.sampling_rate)[0]
            for x in audio
        ]

    if return_dict:
        return {k: v for k, v in [('image', image), ('audio', audio)] if v is not None}
    else:
        return image, audio


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
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


def resize_image(image, min_pixels=None, max_pixels=None, factor=None):
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


def _env_int(name: str):
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
