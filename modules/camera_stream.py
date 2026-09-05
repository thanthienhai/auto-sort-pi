"""Threaded camera capture with a latest-frame buffer.

Yêu cầu (§4 task.md): threaded capture, latest-frame buffer, FPS cấu hình
thực sự được áp dụng, timestamp/frame_id/capture_latency/age cho mỗi frame,
phát hiện disconnect, tự reconnect với giới hạn retry, graceful shutdown.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from .exceptions import CameraError
from .logging_setup import get_logger

logger = get_logger("autosort.camera")


@dataclass
class Frame:
    """Một khung hình kèm metadata truy vết (§4)."""

    image: object
    frame_id: int
    timestamp: float  # time.monotonic() lúc capture xong
    capture_latency: float  # giây, thời gian gọi VideoCapture.read()

    def age(self, now: float | None = None) -> float:
        """Tuổi của frame (giây) tính từ lúc capture tới ``now``."""
        return (now if now is not None else time.monotonic()) - self.timestamp

    def is_stale(self, max_age: float, now: float | None = None) -> bool:
        return self.age(now) > max_age


class CameraStream:
    """Continuously capture frames and expose the newest frame to callers.

    Không cho inference xử lý frame quá cũ: ``read(max_age=...)`` trả về
    ``None`` nếu frame mới nhất đã cũ hơn ``max_age`` giây.
    """

    def __init__(
        self,
        src: int | str = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 15.0,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 1.0,
    ) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.fps = float(fps) if fps and fps > 0 else 0.0
        self._frame_interval = (1.0 / self.fps) if self.fps > 0 else 0.0
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_backoff_seconds = reconnect_backoff_seconds

        self._capture = None
        self._frame: Optional[Frame] = None
        self._frame_counter = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._connected = False

        self._open_capture()

    # ------------------------------------------------------------- capture

    def _open_capture(self) -> None:
        import cv2  # lazy: chỉ cần OpenCV khi thực sự mở camera trên thiết bị.

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass

        capture = cv2.VideoCapture(self.src)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            self._connected = False
            raise CameraError(f"Unable to open camera source: {self.src}")
        self._capture = capture
        self._connected = True

    def start(self) -> "CameraStream":
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update, name="camera-stream", daemon=True)
        self._thread.start()
        return self

    def _reconnect(self) -> bool:
        for attempt in range(1, self.max_reconnect_attempts + 1):
            if self._stop_event.is_set():
                return False
            logger.warning(
                "step.camera.reconnect_attempt",
                attempt=attempt,
                max_attempts=self.max_reconnect_attempts,
            )
            try:
                self._open_capture()
                logger.info("step.camera.reconnected", attempt=attempt)
                self._consecutive_failures = 0
                return True
            except CameraError as exc:
                logger.warning("step.camera.reconnect_failed", attempt=attempt, error=str(exc))
                time.sleep(self.reconnect_backoff_seconds)
        logger.error(
            "step.camera.disconnected",
            attempts=self.max_reconnect_attempts,
        )
        self._connected = False
        return False

    def _update(self) -> None:
        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            started = time.perf_counter()
            ok, image = (False, None)
            try:
                ok, image = self._capture.read()
            except Exception as exc:  # noqa: BLE001 - lỗi driver không được crash thread
                logger.warning("step.camera.read_exception", error=str(exc))
                ok = False

            if not ok or image is None:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 5:
                    self._connected = False
                    if not self._reconnect():
                        # Hết retry: ngừng cố capture nhưng để thread sống để
                        # main loop vẫn gọi được stop() một cách graceful.
                        time.sleep(self.reconnect_backoff_seconds)
                        continue
                time.sleep(0.05)
                continue

            self._consecutive_failures = 0
            capture_latency = time.perf_counter() - started
            self._frame_counter += 1
            frame = Frame(
                image=image,
                frame_id=self._frame_counter,
                timestamp=time.monotonic(),
                capture_latency=capture_latency,
            )
            with self._lock:
                self._frame = frame

            if self._frame_interval:
                elapsed = time.monotonic() - loop_started
                remaining = self._frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    def read(self, max_age: float | None = None) -> Optional[Frame]:
        """Trả về ``Frame`` mới nhất (copy ảnh), hoặc ``None`` nếu chưa có / quá cũ."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        if max_age is not None and frame.is_stale(max_age):
            logger.warning(
                "step.camera.stale_frame",
                frame_id=frame.frame_id,
                age=round(frame.age(), 3),
                max_age=max_age,
            )
            return None
        copied_image = frame.image.copy() if hasattr(frame.image, "copy") else frame.image
        return Frame(
            image=copied_image,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            capture_latency=frame.capture_latency,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
