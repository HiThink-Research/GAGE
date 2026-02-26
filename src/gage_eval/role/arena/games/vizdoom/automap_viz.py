from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


class _Cv2Backend:
    def __init__(self, window_name: str, bgr: bool):
        self.window_name = window_name
        self._bgr = bgr
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _normalize(self, frame):
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] not in (3, 4) and arr.shape[0] in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] not in (3, 4):
            arr = np.squeeze(arr)
        return arr

    def update(self, frame):
        frame = self._normalize(frame)
        if frame is None:
            return None
        if not self._bgr:
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)


class _MplBackend:
    def __init__(self, window_name: str):
        self.window_name = window_name
        plt.ion()
        self.fig, self.ax = plt.subplots()
        try:
            self.fig.canvas.manager.set_window_title(window_name)
        except Exception:
            pass
        self.im = None

    def update(self, frame):
        if frame is None:
            return None
        if self.im is None:
            self.im = self.ax.imshow(frame)
            self.ax.axis("off")
        else:
            self.im.set_data(frame)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        return None

    def close(self):
        plt.close(self.fig)


class _NoopBackend:
    def update(self, frame):
        return None

    def close(self):
        return


class AutomapWindow:
    def __init__(self, window_name: str, bgr: bool):
        if cv2 is not None:
            self._impl = _Cv2Backend(window_name, bgr)
        elif plt is not None:
            self._impl = _MplBackend(window_name)
        else:
            self._impl = _NoopBackend()

    def update(self, frame):
        return self._impl.update(frame)

    def close(self):
        self._impl.close()


def init_window(window_name: str = "ViZDoom Automap", bgr: bool = True) -> AutomapWindow:
    return AutomapWindow(window_name, bgr)
