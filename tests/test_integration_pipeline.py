"""End-to-end integration test cho toàn bộ pipeline FSM (§15 task.md).

Dùng fake frame -> fake detection -> mapping -> calibration -> scheduling ->
target validation -> fake UART -> ACK -> DONE -> cycle success, và test riêng
các failure mode: camera failure, detector failure, invalid calibration,
invalid target, UART timeout, UART retry, MCU error, DONE timeout.

Không cần camera/model/MCU thật (§14 hardware abstraction): ``FakeCamera``,
``FakeDetector`` được inject thẳng vào ``AutoSortPi``; UART dùng
``SerialCommunicator`` thật với ``FakeSerial`` injected (giữ đúng logic
handshake/protocol, chỉ giả transport).
"""

from __future__ import annotations

import time

from modules.exceptions import ModelInferenceError
from modules.serial_comm import SerialCommunicator
from main import AutoSortPi, State


class FakeFrame:
    def __init__(self, image, frame_id=1, timestamp=None):
        self.image = image
        self.frame_id = frame_id
        # time.monotonic(): TargetValidator so sánh captured_at với
        # time.monotonic() hiện tại (cùng "đồng hồ" với CameraStream thật).
        self.timestamp = timestamp if timestamp is not None else time.monotonic()


class FakeCamera:
    """Trả về ``frames[i]`` lần lượt mỗi lần ``read()``; hết danh sách -> None mãi."""

    def __init__(self, frames):
        self.frames = list(frames)
        self._i = 0
        self.stopped = False

    def read(self, max_age=None):
        if self._i < len(self.frames):
            frame = self.frames[self._i]
            self._i += 1
            return frame
        return None

    def stop(self):
        self.stopped = True


class FakeDetector:
    def __init__(self, detections=None, raise_exc=None):
        self.detections = detections or []
        self.raise_exc = raise_exc
        self.calls = 0

    def infer(self, image):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.detections


class FakeTransformer:
    def __init__(self, mapping_fn=None):
        self.mapping_fn = mapping_fn or (lambda u, v: (u, v))

    def pixel_to_world(self, u, v):
        return self.mapping_fn(u, v)


class FakeSerial:
    """Serial-like giả (giống tests/test_serial.py): mỗi write() nạp một batch
    response tương ứng lần thử (attempt) đó."""

    def __init__(self, batches):
        self.batches = [list(b) for b in batches]
        self.queue: list[bytes] = []
        self.writes = 0
        self.is_open = True

    def reset_input_buffer(self):
        self.queue.clear()

    def write(self, data):
        idx = self.writes
        self.writes += 1
        if idx < len(self.batches):
            self.queue.extend(self.batches[idx])

    def flush(self):
        pass

    def readline(self):
        return self.queue.pop(0) if self.queue else b""

    def close(self):
        self.is_open = False


def _make_app(camera, detector, comm, transformer=None):
    app = AutoSortPi(config_dir="config", camera=camera, detector=detector, comm=comm, transformer=transformer)
    # Rút ngắn mọi timeout/delay để test chạy tức thời, không phụ thuộc phần cứng thật.
    app.settings["runtime"]["loop_delay_seconds"] = 0.001
    app.settings["runtime"]["waiting_poll_seconds"] = 0.001
    app.settings["runtime"]["error_backoff_seconds"] = 0.001
    app.settings["uart"]["ack_timeout"] = 0.01
    app.settings["uart"]["action_timeout"] = 0.01
    app.settings["uart"]["max_retries"] = 1
    return app


def _drive(app, max_steps=60, stop_states=()):
    states = []
    for _ in range(max_steps):
        s = app.step()
        states.append(s)
        if s == State.SHUTDOWN or s in stop_states:
            break
    return states


def _one_can_detection():
    # class_id 0 -> "can" trong config/model_classes.json -> group_id 1 (config/class_mapping.json).
    return [{"class_id": 0, "conf": 0.9, "box": (40.0, 40.0, 20.0, 20.0), "center": (50.0, 50.0)}]


def test_happy_path_full_cycle_success():
    camera = FakeCamera([FakeFrame(image="frame0")])
    detector = FakeDetector(detections=_one_can_detection())
    transformer = FakeTransformer(lambda u, v: (100.0, 100.0))
    comm = SerialCommunicator(ser=FakeSerial([[b"!ACK\n", b"!DONE\n"]]))

    app = _make_app(camera, detector, comm, transformer)
    _drive(app, max_steps=40, stop_states=())

    assert app.health.last_pick_at is not None
    assert app.tracker.active_cooldowns() == 1
    assert app.health.last_error is None


def test_camera_failure_never_crashes_stays_scanning():
    camera = FakeCamera([])  # không bao giờ trả frame -> luôn None
    detector = FakeDetector(detections=_one_can_detection())
    comm = SerialCommunicator(ser=FakeSerial([[b"!ACK\n", b"!DONE\n"]]))

    app = _make_app(camera, detector, comm, FakeTransformer())
    states = _drive(app, max_steps=30, stop_states=())

    assert states[-1] == State.SCANNING
    assert app.health.last_frame_at is None
    assert detector.calls == 0  # không có frame -> không bao giờ gọi infer


def test_detector_failure_escalates_to_safe_stop_after_threshold():
    camera = FakeCamera([FakeFrame(image=f"frame{i}") for i in range(10)])
    detector = FakeDetector(raise_exc=ModelInferenceError("simulated hardware failure"))
    comm = SerialCommunicator(ser=FakeSerial([]))

    app = _make_app(camera, detector, comm, FakeTransformer())
    app.settings["inference"]["max_consecutive_errors"] = 2

    states = _drive(app, max_steps=30, stop_states=(State.SHUTDOWN,))

    assert states[-1] == State.SHUTDOWN
    assert "ModelInferenceError" in (app.health.last_error or "")


def test_invalid_calibration_blocks_before_ready():
    # Không inject transformer -> dùng CoordinateTransformer thật đọc
    # config/calibration_matrix.json — file này trong repo vẫn là placeholder
    # identity (chưa calibration thật), nên phải bị reject trước READY (§7).
    camera = FakeCamera([])
    detector = FakeDetector()
    comm = SerialCommunicator(ser=FakeSerial([]))

    app = AutoSortPi(config_dir="config", camera=camera, detector=detector, comm=comm)
    states = _drive(app, max_steps=10, stop_states=(State.SHUTDOWN,))

    assert states[-1] == State.SHUTDOWN
    assert "CalibrationError" in (app.health.last_error or "")
    assert State.READY not in states


def test_invalid_target_out_of_workspace_is_dropped_before_scheduling():
    camera = FakeCamera([FakeFrame(image="frame0")])
    detector = FakeDetector(detections=_one_can_detection())
    transformer = FakeTransformer(lambda u, v: (999999.0, 999999.0))  # ngoài workspace
    comm = SerialCommunicator(ser=FakeSerial([[b"!ACK\n", b"!DONE\n"]]))

    app = _make_app(camera, detector, comm, transformer)
    states = _drive(app, max_steps=20, stop_states=())

    assert State.SCHEDULING not in states
    assert app.health.last_pick_at is None


def test_uart_ack_timeout_then_retry_succeeds():
    camera = FakeCamera([FakeFrame(image="frame0")])
    detector = FakeDetector(detections=_one_can_detection())
    transformer = FakeTransformer(lambda u, v: (100.0, 100.0))
    # Lần thử đầu (writes[0]) không có gì -> timeout; lần retry (writes[1]) có ACK+DONE.
    comm = SerialCommunicator(ser=FakeSerial([[], [b"!ACK\n", b"!DONE\n"]]))

    app = _make_app(camera, detector, comm, transformer)
    app.settings["uart"]["max_retries"] = 2
    _drive(app, max_steps=40, stop_states=())

    assert app.health.last_pick_at is not None


def test_uart_ack_timeout_exhausts_retries_and_drops_target_gracefully():
    camera = FakeCamera([FakeFrame(image="frame0")])
    detector = FakeDetector(detections=_one_can_detection())
    transformer = FakeTransformer(lambda u, v: (100.0, 100.0))
    comm = SerialCommunicator(ser=FakeSerial([[], [], []]))  # không bao giờ ACK

    app = _make_app(camera, detector, comm, transformer)
    app.settings["uart"]["max_retries"] = 2
    states = _drive(app, max_steps=40, stop_states=())

    assert app.health.last_pick_at is None
    assert "UartTimeoutError" in (app.health.last_error or "")
    assert states[-1] == State.SCANNING


def test_mcu_error_on_ack_drops_target_without_retry():
    camera = FakeCamera([FakeFrame(image="frame0")])
    detector = FakeDetector(detections=_one_can_detection())
    transformer = FakeTransformer(lambda u, v: (100.0, 100.0))
    comm = SerialCommunicator(ser=FakeSerial([[b"!ERR,42\n"]]))

    app = _make_app(camera, detector, comm, transformer)
    _drive(app, max_steps=40, stop_states=())

    assert app.health.last_pick_at is None
    assert "McuError" in (app.health.last_error or "")


def test_done_timeout_after_ack_drops_target():
    camera = FakeCamera([FakeFrame(image="frame0")])
    detector = FakeDetector(detections=_one_can_detection())
    transformer = FakeTransformer(lambda u, v: (100.0, 100.0))
    comm = SerialCommunicator(ser=FakeSerial([[b"!ACK\n"]]))  # ACK nhưng không bao giờ DONE

    app = _make_app(camera, detector, comm, transformer)
    _drive(app, max_steps=40, stop_states=())

    assert app.health.last_pick_at is None
    assert "UartTimeoutError" in (app.health.last_error or "")
