"""Main state machine for the Auto Sort Pi pipeline (§2/§3 task.md).

FSM production-oriented với tối thiểu các state theo task.md:

    BOOT -> SELF_CHECK -> INITIALIZING -> READY -> SCANNING -> SCHEDULING
      -> VALIDATING_TARGET -> EXECUTING -> WAITING_ACK -> WAITING_DONE
      -> VERIFYING_RESULT -> (VALIDATING_TARGET | SCANNING)
    ERROR / RECOVERING / SAFE_STOP xử lý lỗi theo từng nhóm (camera / model /
    calibration / target / UART / MCU / timeout / cấu hình) với recovery
    policy riêng — không để lỗi một frame hay một target làm chết cả tiến
    trình (xem ``_ERROR_CLASSIFICATION`` và ``_state_error``/``_state_recovering``).

Mỗi lần vào SCANNING với ít nhất 1 frame hợp lệ mở một *chu kỳ* mang
``request_id`` (cycle id) để structlog ghi toàn bộ input/output từng step của
chu kỳ đó ra một file riêng (giữ nguyên từ kiến trúc cũ).

§14 task.md: các thành phần chính (camera/detector/transformer/comm) có thể
được dependency-inject (``AutoSortPi(camera=FakeCamera(), ...)``) để chạy
integration test không cần phần cứng.
"""

from __future__ import annotations

import json
import signal
import time
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from uuid import uuid4

import structlog
import yaml

from modules import (
    CameraStream,
    CoordinateTransformer,
    SerialCommunicator,
    WasteDetector,
)
from modules.config_validation import assert_valid_settings
from modules.dedup import ObjectTracker
from modules.exceptions import (
    CalibrationError,
    CameraError,
    ConfigError,
    ModelDecodingError,
    ModelError,
    ModelInferenceError,
    McuError,
    UartDisconnectedError,
    UartTimeoutError,
)
from modules.logging_setup import configure_logging, get_logger, prune_old_request_logs
from modules.safety_validator import SafetyLimits, SafetyValidator
from modules.scheduler import Scheduler, SchedulerConfig
from modules.target_validator import TargetValidator, WorkspaceBounds
from modules.uart_protocol import ResponseKind, build_pick_payload

logger = get_logger("autosort.main")

# Lỗi phát sinh ở BOOT/SELF_CHECK/INITIALIZING (cấu hình sai, model không
# tương thích, calibration placeholder) không thể tự phục hồi — cần con người
# sửa file rồi khởi động lại tiến trình. Mọi lỗi khác (camera thoáng qua, UART
# mất kết nối, MCU báo lỗi, timeout) được coi là có thể retry/recover.
_FATAL_ERROR_TYPES = (ConfigError, ModelError, CalibrationError)


class State(Enum):
    BOOT = auto()
    SELF_CHECK = auto()
    INITIALIZING = auto()
    READY = auto()
    SCANNING = auto()
    SCHEDULING = auto()
    VALIDATING_TARGET = auto()
    EXECUTING = auto()
    WAITING_ACK = auto()
    WAITING_DONE = auto()
    VERIFYING_RESULT = auto()
    RECOVERING = auto()
    ERROR = auto()
    SAFE_STOP = auto()
    SHUTDOWN = auto()


# States đáng ghi health snapshot — bỏ qua WAITING_ACK/WAITING_DONE (poll mỗi
# 0.05s, ghi file mỗi tick sẽ tốn I/O vô ích trên thẻ SD của Pi).
_SNAPSHOT_STATES = frozenset(
    {
        State.READY,
        State.SCANNING,
        State.SCHEDULING,
        State.VALIDATING_TARGET,
        State.VERIFYING_RESULT,
        State.RECOVERING,
        State.ERROR,
        State.SAFE_STOP,
        State.SHUTDOWN,
    }
)


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


HEALTH_SNAPSHOT_PATH = Path("logs") / "health_snapshot.json"


class HealthStatus:
    """Trạng thái quan sát được cho ``tools/health.py`` (§18 task.md).

    Dùng ``time.time()`` (wall-clock), không phải ``time.monotonic()``, vì các
    giá trị này được ghi ra ``logs/health_snapshot.json`` để một process khác
    (``tools/health.py``) đọc lại — monotonic clock không có ý nghĩa giữa hai
    process khác nhau.
    """

    def __init__(self) -> None:
        self.last_frame_at: float | None = None
        self.last_inference_at: float | None = None
        self.last_pick_at: float | None = None
        self.last_error: str | None = None
        self.state: str | None = None

    def as_dict(self) -> dict:
        return {
            "last_frame_at": self.last_frame_at,
            "last_inference_at": self.last_inference_at,
            "last_pick_at": self.last_pick_at,
            "last_error": self.last_error,
            "state": self.state,
            "updated_at": time.time(),
        }

    def write_snapshot(self, path: Path = HEALTH_SNAPSHOT_PATH) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.as_dict()), encoding="utf-8")
        except OSError:  # noqa: BLE001 - health snapshot không được làm crash pipeline
            logger.warning("step.health.snapshot_write_failed")


class AutoSortPi:
    """Điều phối chính dưới dạng finite state machine."""

    def __init__(
        self,
        config_dir: str = "config",
        camera: CameraStream | None = None,
        detector: WasteDetector | None = None,
        transformer: CoordinateTransformer | None = None,
        comm: SerialCommunicator | None = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.settings = yaml.safe_load(
            (self.config_dir / "settings.yaml").read_text(encoding="utf-8")
        )
        self.mapping = load_json(str(self.config_dir / "class_mapping.json"))
        self.class_names = load_json(str(self.config_dir / "model_classes.json"))

        # Injected components (§14) — nếu None sẽ được construct thật ở INITIALIZING.
        self.camera = camera
        self.detector = detector
        self.transformer = transformer
        self.comm = comm
        self._camera_injected = camera is not None
        self._detector_injected = detector is not None
        self._transformer_injected = transformer is not None
        self._comm_injected = comm is not None

        self.target_validator: TargetValidator | None = None
        self.safety_validator: SafetyValidator | None = None
        self.scheduler: Scheduler | None = None
        self.tracker: ObjectTracker | None = None

        self.pending_targets: list[dict] = []
        self.scheduled: list[dict] = []
        self.current_target: dict | None = None
        self._current_payload: str = ""
        self.uart_attempt: int = 0
        self.ack_deadline: float = 0.0
        self.done_deadline: float = 0.0
        self._pick_started_at: float = 0.0
        self.picked_this_cycle: int = 0

        self.consecutive_model_errors: int = 0
        self.recovery_attempts: int = 0
        self.pending_error: Exception | None = None

        self.health = HealthStatus()
        self.state = State.BOOT

        self._handlers = {
            State.BOOT: self._state_boot,
            State.SELF_CHECK: self._state_self_check,
            State.INITIALIZING: self._state_initializing,
            State.READY: self._state_ready,
            State.SCANNING: self._state_scanning,
            State.SCHEDULING: self._state_scheduling,
            State.VALIDATING_TARGET: self._state_validating_target,
            State.EXECUTING: self._state_executing,
            State.WAITING_ACK: self._state_waiting_ack,
            State.WAITING_DONE: self._state_waiting_done,
            State.VERIFYING_RESULT: self._state_verifying_result,
            State.RECOVERING: self._state_recovering,
            State.ERROR: self._state_error,
            State.SAFE_STOP: self._state_safe_stop,
        }

    # ---------------------------------------------------------- cycle scope

    def _begin_cycle(self) -> None:
        request_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid4().hex[:8]
        )
        structlog.contextvars.bind_contextvars(request_id=request_id)

    def _end_cycle(self) -> None:
        structlog.contextvars.clear_contextvars()

    # --------------------------------------------------------------- BOOT

    def _state_boot(self) -> State:
        logger.info("system.boot")
        return State.SELF_CHECK

    def _state_self_check(self) -> State:
        try:
            assert_valid_settings(self.settings)
            model_path = Path(self.settings["inference"]["model_path"])
            if not self._detector_injected and not model_path.is_file():
                raise ConfigError(f"Model file không tồn tại: {model_path}")
            calib_path = Path(self.settings["calibration"]["file"])
            if not self._transformer_injected and not calib_path.is_file():
                raise ConfigError(f"Calibration file không tồn tại: {calib_path}")
            logger.info("system.self_check.ok")
            return State.INITIALIZING
        except ConfigError as exc:
            self.pending_error = exc
            return State.ERROR

    # --------------------------------------------------------- INITIALIZING

    def _state_initializing(self) -> State:
        camera_cfg = self.settings["camera"]
        infer_cfg = self.settings["inference"]
        calib_cfg = self.settings["calibration"]
        uart_cfg = self.settings["uart"]
        workspace_cfg = self.settings["workspace"]
        robot_safety_cfg = self.settings["robot"]["safety"]
        dedup_cfg = self.settings["dedup"]

        try:
            if self.camera is None:
                self.camera = CameraStream(
                    src=camera_cfg["source"],
                    width=camera_cfg["width"],
                    height=camera_cfg["height"],
                    fps=camera_cfg["fps"],
                    max_reconnect_attempts=camera_cfg["max_reconnect_attempts"],
                    reconnect_backoff_seconds=camera_cfg["reconnect_backoff_seconds"],
                ).start()

            if self.detector is None:
                self.detector = WasteDetector(
                    infer_cfg["model_path"],
                    confidence_threshold=infer_cfg["confidence_threshold"],
                    nms_iou_threshold=infer_cfg["nms_iou_threshold"],
                    num_classes=len(self.class_names),
                )

            if self.transformer is None:
                self.transformer = CoordinateTransformer(
                    calib_cfg["file"],
                    max_reprojection_error_mm=calib_cfg["max_reprojection_error_mm"],
                    allow_identity=False,
                )

            if self.comm is None:
                self.comm = SerialCommunicator(
                    port=uart_cfg["port"],
                    baudrate=uart_cfg["baudrate"],
                    timeout=uart_cfg["timeout"],
                    ack_timeout=uart_cfg["ack_timeout"],
                    action_timeout=uart_cfg["action_timeout"],
                    max_retries=uart_cfg["max_retries"],
                    max_reconnect_attempts=uart_cfg["max_reconnect_attempts"],
                    reconnect_backoff_seconds=uart_cfg["reconnect_backoff_seconds"],
                )

            self.target_validator = TargetValidator(
                workspace=WorkspaceBounds(
                    x_min_mm=workspace_cfg["x_min_mm"],
                    x_max_mm=workspace_cfg["x_max_mm"],
                    y_min_mm=workspace_cfg["y_min_mm"],
                    y_max_mm=workspace_cfg["y_max_mm"],
                ),
                valid_group_ids=frozenset(int(g) for g in set(self.mapping.values())),
                min_confidence=workspace_cfg["min_confidence"],
                max_target_age_seconds=workspace_cfg["max_target_age_seconds"],
            )
            self.safety_validator = SafetyValidator(
                SafetyLimits(
                    x_min_mm=robot_safety_cfg["x_min_mm"],
                    x_max_mm=robot_safety_cfg["x_max_mm"],
                    y_min_mm=robot_safety_cfg["y_min_mm"],
                    y_max_mm=robot_safety_cfg["y_max_mm"],
                    min_angle_deg=robot_safety_cfg["min_angle_deg"],
                    max_angle_deg=robot_safety_cfg["max_angle_deg"],
                )
            )
            self.scheduler = Scheduler(SchedulerConfig.from_dict(self.settings["scheduler"]))
            self.tracker = ObjectTracker(
                suppression_radius_mm=dedup_cfg["suppression_radius_mm"],
                cooldown_seconds=dedup_cfg["cooldown_seconds"],
                disappearance_confirm_scans=dedup_cfg["disappearance_confirm_scans"],
            )

            prune_old_request_logs(self.settings["logging"].get("retention_days", 14))
            logger.info("system.ready", classes=len(self.class_names))
            return State.READY
        except (CameraError, ModelError, CalibrationError) as exc:
            self.pending_error = exc
            return State.ERROR

    def _state_ready(self) -> State:
        print("Auto Sort Pi: vision pipeline ready.")
        logger.info("system.state.ready")
        self.recovery_attempts = 0
        return State.SCANNING

    # --------------------------------------------------------------- SCANNING

    def _state_scanning(self) -> State:
        self._begin_cycle()
        loop_delay = self.settings["runtime"]["loop_delay_seconds"]
        camera_cfg = self.settings["camera"]

        frame = self.camera.read(max_age=camera_cfg["max_frame_age_seconds"])
        if frame is None:
            self._end_cycle()
            time.sleep(loop_delay)
            return State.SCANNING

        self.health.last_frame_at = time.time()

        try:
            detections = self.detector.infer(frame.image)
        except (ModelInferenceError, ModelDecodingError) as exc:
            self.consecutive_model_errors += 1
            max_errors = self.settings["inference"]["max_consecutive_errors"]
            logger.error(
                "step.infer.failed",
                error=str(exc),
                error_code=type(exc).__name__,
                consecutive=self.consecutive_model_errors,
            )
            self.health.last_error = f"{type(exc).__name__}: {exc}"
            self._end_cycle()
            if self.consecutive_model_errors >= max_errors:
                self.pending_error = exc
                return State.SAFE_STOP
            time.sleep(loop_delay)
            return State.SCANNING

        self.consecutive_model_errors = 0
        self.health.last_inference_at = time.time()

        targets = self._build_targets(detections, frame)
        targets = self.target_validator.filter_valid(targets)
        targets = self.tracker.filter_new_targets(targets)

        logger.info(
            "step.scan.done",
            frame_id=frame.frame_id,
            n_detections=len(detections),
            n_targets=len(targets),
        )
        if not targets:
            self._end_cycle()
            time.sleep(loop_delay)
            return State.SCANNING

        self.pending_targets = targets
        return State.SCHEDULING

    def _build_targets(self, detections: list[dict], frame) -> list[dict]:
        """Detection Validation + Class Mapping + Pixel→World (§2 pipeline)."""
        targets: list[dict] = []
        for detection in detections:
            box_w, box_h = detection["box"][2], detection["box"][3]
            if box_w <= 0 or box_h <= 0:
                continue  # Detection Validation: bbox suy biến bị loại ngay.

            class_id = int(detection["class_id"])
            if class_id < 0 or class_id >= len(self.class_names):
                continue
            class_name = self.class_names[class_id]
            group_id = self.mapping.get(class_name)
            if group_id is None:
                continue

            u_center, v_center = detection["center"]
            try:
                x_world, y_world = self.transformer.pixel_to_world(u_center, v_center)
            except CalibrationError as exc:
                logger.warning("step.transform.failed", error=str(exc))
                continue

            targets.append(
                {
                    "target_id": uuid4().hex,
                    "group_id": int(group_id),
                    "x": x_world,
                    "y": y_world,
                    "conf": float(detection["conf"]),
                    "class": class_name,
                    "pixel_x": u_center,
                    "pixel_y": v_center,
                    "frame_id": frame.frame_id,
                    "captured_at": frame.timestamp,
                }
            )
        return targets

    # ------------------------------------------------------------- SCHEDULING

    def _state_scheduling(self) -> State:
        self.scheduled = self.scheduler.schedule(self.pending_targets)
        self.pending_targets = []
        self.picked_this_cycle = 0
        logger.info(
            "step.schedule.done",
            order=[
                {
                    "target_id": t.get("target_id"),
                    "group_id": t["group_id"],
                    "x": t["x"],
                    "y": t["y"],
                    "scheduler_distance": t.get("scheduler_distance"),
                }
                for t in self.scheduled
            ],
        )
        if not self.scheduled:
            self._end_cycle()
            return State.SCANNING
        return State.VALIDATING_TARGET

    # --------------------------------------------------------- VALIDATING_TARGET

    def _state_validating_target(self) -> State:
        while self.scheduled:
            target = self.scheduled.pop(0)
            try:
                self.target_validator.validate(target)
            except Exception as exc:  # noqa: BLE001 - TargetValidationError cụ thể, log rồi bỏ target
                logger.warning(
                    "step.target_validation.rejected_pre_execute",
                    target_id=target.get("target_id"),
                    reason=str(exc),
                )
                continue

            safety_result = self.safety_validator.check(target)
            if not safety_result.ok:
                logger.error(
                    "step.safety.rejected",
                    target_id=target.get("target_id"),
                    group_id=target["group_id"],
                    reason=safety_result.reason,
                )
                continue

            self.current_target = target
            return State.EXECUTING

        logger.info("cycle.success", n_picked=self.picked_this_cycle)
        self._end_cycle()
        return State.SCANNING

    # ------------------------------------------------------------- EXECUTING

    def _state_executing(self) -> State:
        target = self.current_target
        assert target is not None
        self._current_payload = build_pick_payload(
            target["group_id"], target["x"], target["y"], target.get("angle", 0.0)
        )
        self.uart_attempt = 0
        self._pick_started_at = time.monotonic()
        logger.info(
            "step.pick.start",
            target_id=target.get("target_id"),
            group_id=target["group_id"],
            x=target["x"],
            y=target["y"],
            confidence=target.get("conf"),
            pixel_x=target.get("pixel_x"),
            pixel_y=target.get("pixel_y"),
            scheduler_distance=target.get("scheduler_distance"),
        )
        if not self.comm.send_command_async(self._current_payload):
            self.pending_error = UartDisconnectedError("UART mất kết nối khi gửi PICK")
            return State.RECOVERING

        self.ack_deadline = time.monotonic() + self.settings["uart"]["ack_timeout"]
        return State.WAITING_ACK

    # ------------------------------------------------------------ WAITING_ACK

    def _state_waiting_ack(self) -> State:
        resp = self.comm.read_one_response()
        if resp.kind == ResponseKind.ACK:
            logger.info("step.uart.ack", target_id=self.current_target.get("target_id"))
            self.done_deadline = time.monotonic() + self.settings["uart"]["action_timeout"]
            return State.WAITING_DONE
        if resp.kind == ResponseKind.ERR:
            return self._fail_current_target(McuError(resp.code), "ack")

        if time.monotonic() >= self.ack_deadline:
            max_retries = self.settings["uart"]["max_retries"]
            self.uart_attempt += 1
            if self.uart_attempt > max_retries:
                return self._fail_current_target(
                    UartTimeoutError(f"ACK timeout sau {self.uart_attempt} lần thử"), "ack_timeout"
                )
            logger.warning(
                "step.uart.ack_timeout_retry",
                target_id=self.current_target.get("target_id"),
                attempt=self.uart_attempt,
            )
            if not self.comm.send_command_async(self._current_payload):
                self.pending_error = UartDisconnectedError("UART mất kết nối khi retry PICK")
                return State.RECOVERING
            self.ack_deadline = time.monotonic() + self.settings["uart"]["ack_timeout"]
            return State.WAITING_ACK

        time.sleep(self.settings["runtime"]["waiting_poll_seconds"])
        return State.WAITING_ACK

    # ----------------------------------------------------------- WAITING_DONE

    def _state_waiting_done(self) -> State:
        resp = self.comm.read_one_response()
        if resp.kind == ResponseKind.DONE:
            return State.VERIFYING_RESULT
        if resp.kind == ResponseKind.ERR:
            return self._fail_current_target(McuError(resp.code), "done")
        if time.monotonic() >= self.done_deadline:
            return self._fail_current_target(UartTimeoutError("DONE timeout"), "done_timeout")

        time.sleep(self.settings["runtime"]["waiting_poll_seconds"])
        return State.WAITING_DONE

    def _fail_current_target(self, exc: Exception, error_code: str) -> State:
        target = self.current_target
        latency_ms = (time.monotonic() - self._pick_started_at) * 1000.0
        logger.error(
            "step.pick.failed",
            target_id=target.get("target_id") if target else None,
            group_id=target.get("group_id") if target else None,
            uart_attempt=self.uart_attempt,
            uart_status="failed",
            error_code=error_code,
            error=str(exc),
            latency_ms=round(latency_ms, 1),
        )
        self.health.last_error = f"{type(exc).__name__}: {exc}"
        self.current_target = None
        # Lỗi MCU / timeout riêng cho một target không được làm chết cả cycle:
        # bỏ target này, tiếp tục xử lý các target còn lại trong hàng đợi.
        return State.VALIDATING_TARGET

    # -------------------------------------------------------- VERIFYING_RESULT

    def _state_verifying_result(self) -> State:
        target = self.current_target
        assert target is not None
        latency_ms = (time.monotonic() - self._pick_started_at) * 1000.0
        logger.info(
            "step.pick.done",
            target_id=target.get("target_id"),
            group_id=target["group_id"],
            uart_attempt=self.uart_attempt,
            uart_status="done",
            latency_ms=round(latency_ms, 1),
        )
        if target.get("target_id"):
            self.target_validator.mark_processed(target["target_id"])
        self.tracker.mark_done(target)
        self.health.last_pick_at = time.time()
        self.picked_this_cycle += 1
        self.current_target = None
        self.recovery_attempts = 0
        return State.VALIDATING_TARGET

    # ----------------------------------------------------------------- ERROR

    def _state_error(self) -> State:
        exc = self.pending_error
        logger.error("system.error", error_type=type(exc).__name__ if exc else "unknown", error=str(exc))
        self.health.last_error = f"{type(exc).__name__}: {exc}" if exc else "unknown"
        if isinstance(exc, _FATAL_ERROR_TYPES):
            return State.SAFE_STOP
        return State.RECOVERING

    def _state_recovering(self) -> State:
        self.recovery_attempts += 1
        max_attempts = self.settings["runtime"]["max_consecutive_cycle_errors"]
        logger.warning(
            "system.recovering",
            attempt=self.recovery_attempts,
            max_attempts=max_attempts,
            error=str(self.pending_error) if self.pending_error else None,
        )
        if self.recovery_attempts > max_attempts:
            return State.SAFE_STOP

        time.sleep(self.settings["runtime"]["error_backoff_seconds"])
        self.pending_error = None
        self._end_cycle()
        if self.current_target is not None:
            self.current_target = None
            return State.VALIDATING_TARGET
        return State.SCANNING

    def _state_safe_stop(self) -> State:
        reason = str(self.pending_error) if self.pending_error else "unknown"
        logger.error("system.safe_stop", reason=reason)
        print(
            "Auto Sort Pi: SAFE_STOP — cần can thiệp thủ công (xem log để biết chi tiết): "
            + reason
        )
        return State.SHUTDOWN

    # ------------------------------------------------------------------ run

    def step(self) -> State:
        """Thực hiện đúng một chuyển trạng thái FSM. Dùng bởi ``run()`` và test
        (integration test drive từng bước thay vì vòng lặp vô hạn — §15)."""
        self.state = self._handlers[self.state]()
        if self.state in _SNAPSHOT_STATES:
            self.health.state = self.state.name
            self.health.write_snapshot()
        return self.state

    def run(self) -> None:
        try:
            while self.state != State.SHUTDOWN:
                self.step()
        except KeyboardInterrupt:
            logger.info("system.stopping")
            print("Stopping Auto Sort Pi...")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        structlog.contextvars.clear_contextvars()
        if self.camera is not None:
            self.camera.stop()
        if self.comm is not None:
            self.comm.close()


def _handle_sigterm(signum, frame) -> None:  # noqa: ARG001 - chữ ký bắt buộc bởi signal.signal
    """systemd gửi SIGTERM khi ``systemctl stop``/restart — chuyển thành
    KeyboardInterrupt để tái dùng đường graceful-shutdown sẵn có của
    ``AutoSortPi.run()`` (đóng camera + UART trong ``_shutdown()``)."""
    raise KeyboardInterrupt()


def main() -> None:
    configure_logging()
    signal.signal(signal.SIGTERM, _handle_sigterm)
    AutoSortPi().run()


if __name__ == "__main__":
    main()
