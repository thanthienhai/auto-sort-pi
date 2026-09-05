"""Kiểm thử CameraStream: latest-frame buffer, frame age, reconnect.

Dùng một ``FakeVideoCapture`` giả lập ``cv2.VideoCapture`` (monkeypatch) để
không cần camera thật, nhưng vẫn chạy qua đúng logic thread capture / reconnect
của ``CameraStream``.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from modules.camera_stream import CameraStream, Frame


class FakeVideoCapture:
    """Giả lập cv2.VideoCapture: có thể "rút camera" giữa chừng để test reconnect."""

    instances: list["FakeVideoCapture"] = []

    def __init__(self, src):
        self.src = src
        self._opened = True
        self.reads = 0
        self.fail_reads = 0  # số lần .read() kế tiếp trả về thất bại
        self.props: dict[int, float] = {}
        FakeVideoCapture.instances.append(self)

    def set(self, prop_id, value):
        self.props[prop_id] = value
        return True

    def isOpened(self):
        return self._opened

    def read(self):
        self.reads += 1
        if self.fail_reads > 0:
            self.fail_reads -= 1
            return False, None
        frame = np.full((4, 4, 3), self.reads, dtype=np.uint8)
        return True, frame

    def release(self):
        self._opened = False


@pytest.fixture(autouse=True)
def fake_cv2(monkeypatch):
    FakeVideoCapture.instances.clear()
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    yield


def test_frame_age_and_staleness():
    frame = Frame(image=None, frame_id=1, timestamp=time.monotonic() - 5.0, capture_latency=0.01)
    assert frame.age() >= 5.0
    assert frame.is_stale(max_age=1.0) is True
    assert frame.is_stale(max_age=10.0) is False


def test_read_returns_none_before_first_frame():
    stream = CameraStream(src=0)
    assert stream.read() is None
    stream.stop()


def test_threaded_capture_produces_incrementing_frame_ids():
    stream = CameraStream(src=0, fps=0).start()
    deadline = time.monotonic() + 2.0
    frame = None
    while time.monotonic() < deadline:
        frame = stream.read()
        if frame is not None and frame.frame_id >= 3:
            break
        time.sleep(0.01)
    stream.stop()
    assert frame is not None
    assert frame.frame_id >= 3
    assert frame.age() < 1.0
    assert frame.capture_latency >= 0.0


def test_stale_frame_rejected_by_max_age():
    stream = CameraStream(src=0, fps=0).start()
    deadline = time.monotonic() + 2.0
    while stream.read() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    stream.stop()
    # Frame tồn tại nhưng coi như quá cũ với max_age cực nhỏ.
    assert stream.read(max_age=0.0) is None


def test_reconnect_after_disconnect_then_success():
    stream = CameraStream(src=0, max_reconnect_attempts=3, reconnect_backoff_seconds=0.01)
    cap = FakeVideoCapture.instances[0]
    # Giả lập rút camera: 10 lần read kế tiếp thất bại (đủ trigger reconnect ở >=5).
    cap.fail_reads = 10
    stream.start()
    deadline = time.monotonic() + 3.0
    while stream.is_connected is True and time.monotonic() < deadline:
        # Chờ tới khi reconnect logic kích hoạt (is_connected chuyển False rồi True lại).
        time.sleep(0.02)
    time.sleep(0.2)
    stream.stop()
    # _open_capture() được gọi lại nghĩa là có instance FakeVideoCapture mới.
    assert len(FakeVideoCapture.instances) >= 2


def test_stop_is_idempotent_and_releases_capture():
    stream = CameraStream(src=0)
    stream.start()
    stream.stop()
    cap = FakeVideoCapture.instances[0]
    assert cap._opened is False
    stream.stop()  # không được raise khi gọi lần 2
