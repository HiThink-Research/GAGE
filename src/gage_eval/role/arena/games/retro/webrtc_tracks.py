"""WebRTC track helpers for retro game streaming."""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame


class GameVideoTrack(VideoStreamTrack):
    """Video track that pulls frames from an asyncio queue."""

    def __init__(self, frame_queue: "asyncio.Queue[np.ndarray]", *, fps: int = 60) -> None:
        super().__init__()
        self._frame_queue = frame_queue
        self._fps = max(1, int(fps))
        self._last_frame: Optional[np.ndarray] = None

    async def recv(self) -> VideoFrame:
        """Return the next video frame for the WebRTC peer."""

        # STEP 1: Await the next frame from the producer.
        frame = await self._frame_queue.get()
        if frame is None:
            frame = self._last_frame

        # STEP 2: Provide a fallback frame if no data is available yet.
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self._last_frame = frame
        pts, time_base = await self.next_timestamp()
        video = VideoFrame.from_ndarray(frame, format="rgb24")
        video.pts = pts
        video.time_base = time_base
        return video


__all__ = ["GameVideoTrack"]
