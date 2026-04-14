"""Video-MME preprocessor built on BasePreprocessor with video handling."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.preprocessors.video_mme.this import create_prompt
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Message,
    MessageContent,
    Sample,
)
from gage_eval.assets.datasets.utils.multimodal import resolve_media_path
from loguru import logger


def _fetch_url_as_data_url(url: str, timeout: int = 30) -> Optional[str]:
    """Download a remote media file and return it as a base64 data URL."""
    try:
        with urlopen(url, timeout=timeout) as response:
            data = response.read()
        content_type = (
            response.headers.get("Content-Type", "").split(";")[0].strip()
        )
        if not content_type:
            content_type = mimetypes.guess_type(url)[0] or "application/octet-stream"
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"
    except Exception as exc:
        logger.warning(f"Failed to fetch video URL {url}: {exc}")
        return None


class VideoMMEChatPreprocessor(BasePreprocessor):
    """Preprocess Video-MME records into a chat-style multimodal Sample."""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        system_prompt: str | None = None,
        content_root: str | None = None,
        pre_encode_video: bool = True,
        include_subtitles: bool = False,
        subtitles: str | None = None,
        schema_version: str = SCHEMA_VERSION,
        **kwargs: Any,
    ) -> Sample:
        sample = dict(record)

        # ------------------------------------------------------------------
        # STEP 1: Build a stable sample id from the available id fields.
        # ------------------------------------------------------------------
        id_parts: List[str] = []
        for key in ("video_id", "videoID", "question_id"):
            val = sample.get(key)
            if val is not None and str(val).strip():
                id_parts.append(str(val).strip())
        sample_id = "_".join(id_parts) if id_parts else str(hash(str(sample)))

        # ------------------------------------------------------------------
        # STEP 2: Extract core fields.
        # ------------------------------------------------------------------
        question = sample.get("question", "")
        options = sample.get("options") or []
        answer = str(sample.get("answer", "")).strip()
        url = sample.get("url")
        # HuggingFace datasets often use the column name ``video`` instead of ``url``.
        # Normalise it here so the preprocessor can consume a single ``url`` field.
        if not url:
            video_value = sample.get("video")
            if isinstance(video_value, dict):
                url = video_value.get("path") or video_value.get("bytes")
            elif isinstance(video_value, str):
                url = video_value

        # ------------------------------------------------------------------
        # STEP 3: Collect metadata.
        # ------------------------------------------------------------------
        metadata: Dict[str, Any] = {
            "duration": sample.get("duration"),
            "domain": sample.get("domain"),
            "sub_category": sample.get("sub_category"),
            "task_type": sample.get("task_type"),
            "videoID": sample.get("videoID"),
        }
        if content_root:
            metadata["content_root"] = content_root

        # ------------------------------------------------------------------
        # STEP 4: Build prompt text.
        # ------------------------------------------------------------------
        prompt = create_prompt(
            question=question,
            options=options,
            include_subtitles=include_subtitles,
            subtitles=subtitles,
        )

        # ------------------------------------------------------------------
        # STEP 5: Resolve video fragment (prefer local path, then base64 data URL).
        # ------------------------------------------------------------------
        video_frag: Optional[MessageContent] = None
        local_video_path = sample.get("local_video_path")
        if local_video_path:
            if not Path(local_video_path).exists():
                return None
            video_frag = MessageContent(
                type="video_url", video_url={"url": local_video_path}
            )
        else:
            return None

        # ------------------------------------------------------------------
        # STEP 6: Build messages.
        # ------------------------------------------------------------------
        content: List[MessageContent] = [
            MessageContent(type="text", text=prompt.strip())
        ]
        if video_frag:
            content.append(video_frag)

        messages: List[Message] = []
        if system_prompt:
            messages.append(
                Message(
                    role="system",
                    content=[
                        MessageContent(
                            type="text", text=system_prompt.strip()
                        )
                    ],
                )
            )
        messages.append(Message(role="user", content=content))
        # ------------------------------------------------------------------
        # STEP 7: Normalize options (strip leading letter prefix if present).
        # ------------------------------------------------------------------
        import re as _re
        clean_options: List[str] = []
        for opt in options:
            opt_str = str(opt).strip()
            # e.g. "A. Apples.", "A) Apples.", "A Apples." -> "Apples."
            cleaned = _re.sub(r"^[A-Da-d][\.\)\s]\s*", "", opt_str)
            clean_options.append(cleaned)

        return Sample(
            id=sample_id,
            schema_version=schema_version,
            options=clean_options or None,
            messages=messages,
            references=[answer],
            label=answer,
            metadata=metadata,
        )


__all__ = ["VideoMMEChatPreprocessor"]
