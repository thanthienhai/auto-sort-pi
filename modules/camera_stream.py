"""Threaded camera capture with a latest-frame buffer."""

from __future__ import annotations

import threading
import time
from typing import Optional


class CameraStream:
    """Continuously capture frames and expose the newest frame to callers."""

    def __init__(self, src: int | str = 0, width: int = 1920, height: int = 1080) -> None:
        import cv2  # lazy: chỉ cần OpenCV khi thực sự mở camera trên thiết bị.

        self.src = src
        self.width = width
        self.height = height
        self._capture = cv2.VideoCapture(src)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: Optional[object] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CameraStream":
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.src}")
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update, name="camera-stream", daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._capture.release()
