"""Video-MME resource provider that injects local video paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

from gage_eval.assets.datasets.bundles.base import BaseBundle
from loguru import logger


class VideoMMEBundle(BaseBundle):
    """Provide Video-MME related resources.

    This bundle scans a local directory for video files named
    ``{video_id}.mp4`` and injects ``local_video_path`` into each sample.
    Filtering (dropping samples without a local video) is intentionally left
    to the preprocessor so that responsibilities remain aligned with the
    GAGE framework design.
    """

    def __init__(
        self,
        video_dir: str = "/mnt/aime_data_ssd/user_workspace/zhuwenqiao/data/video-mme",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.video_dir = Path(video_dir).expanduser().resolve()
        self.available_video_ids: Set[str] = set()

    def load(self) -> None:
        """Scan the local video directory and cache available video IDs."""
        if not self.video_dir.exists():
            logger.warning(f"Video directory does not exist: {self.video_dir}")
            return

        for entry in self.video_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".mp4":
                self.available_video_ids.add(entry.stem)

        logger.info(
            f"VideoMMEBundle loaded {len(self.available_video_ids)} videos from {self.video_dir}"
        )

    def provide(self, sample: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Return the sample augmented with ``local_video_path`` when available."""
        sample_dict = dict(sample)
        video_id = sample_dict.get("videoID") or sample_dict.get("video_id")
        if video_id:
            video_id_str = str(video_id).strip()
            if video_id_str in self.available_video_ids:
                local_path = self.video_dir / f"{video_id_str}.mp4"
                sample_dict["local_video_path"] = str(local_path)
        return sample_dict
