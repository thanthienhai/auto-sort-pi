"""Main state machine for the Auto Sort Pi pipeline.

Máy trạng thái tường minh điều phối pipeline thị giác → robot:

    INIT ──► SCAN ──► SCHEDULE ──► EXECUTE ──► SCAN
              ▲          │            │
              │          └──(no target)          │ (pick failed / lỗi)
              └────────────────────────── ERROR ◄┘

Mỗi lần vào SCAN mở một *chu kỳ* mang ``request_id`` (cycle id) để structlog ghi
toàn bộ input/output từng step của chu kỳ đó ra một file riêng.
"""

from __future__ import annotations

import json
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
    optimize_trajectory,
)
from modules.logging_setup import configure_logging, get_logger

logger = get_logger("autosort.main")


class State(Enum):
    INIT = auto()
    SCAN = auto()
    SCHEDULE = auto()
    EXECUTE = auto()
    ERROR = auto()
    SHUTDOWN = auto()


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class AutoSortPi:
    """Điều phối chính dưới dạng finite state machine."""

    def __init__(self, config_dir: str = "config") -> None:
        self.config_dir = Path(config_dir)
        self.settings = yaml.safe_load(
            (self.config_dir / "settings.yaml").read_text(encoding="utf-8")
        )
        self.mapping = load_json(str(self.config_dir / "class_mapping.json"))
        self.class_names = load_json(str(self.config_dir / "model_classes.json"))
        self.loop_delay = float(self.settings["runtime"]["loop_delay_seconds"])

        self.camera: CameraStream | None = None
        self.detector: WasteDetector | None = None
        self.transformer: CoordinateTransformer | None = None
        self.comm: SerialCommunicator | None = None

        self.targets: list[dict] = []
        self.scheduled: list[dict] = []
        self.failed_target: dict | None = None
        self.state = State.INIT

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

    # --------------------------------------------------------------- states

    def _state_init(self) -> State:
        camera_cfg = self.settings["camera"]
        infer_cfg = self.settings["inference"]
        uart_cfg = self.settings["uart"]

        self.camera = CameraStream(
            src=camera_cfg["source"],
            width=camera_cfg["width"],
            height=camera_cfg["height"],
        ).start()
        self.detector = WasteDetector(
            infer_cfg["model_path"],
            confidence_threshold=infer_cfg["confidence_threshold"],
            nms_iou_threshold=infer_cfg["nms_iou_threshold"],
            num_classes=len(self.class_names),
        )
        self.transformer = CoordinateTransformer(
            str(self.config_dir / "calibration_matrix.json")
        )
        self.comm = SerialCommunicator(
            port=uart_cfg["port"],
            baudrate=uart_cfg["baudrate"],
            timeout=uart_cfg["timeout"],
            ack_timeout=uart_cfg["ack_timeout"],
            action_timeout=uart_cfg["action_timeout"],
            max_retries=uart_cfg["max_retries"],
        )
        logger.info("system.ready", classes=len(self.class_names))
        print("Auto Sort Pi: vision pipeline ready.")
        return State.SCAN

    def _state_scan(self) -> State:
        self._begin_cycle()
        try:
            frame = self.camera.read()
            if frame is None:
                self._end_cycle()
                time.sleep(self.loop_delay)
                return State.SCAN

            detections = self.detector.infer(frame)
            self.targets = self._build_targets(detections)
            logger.info(
                "step.detect.done",
                n_detections=len(detections),
                n_targets=len(self.targets),
            )
            if not self.targets:
                self._end_cycle()
                time.sleep(self.loop_delay)
                return State.SCAN
            return State.SCHEDULE
        except Exception as exc:  # noqa: BLE001 - lỗi 1 khung không được sập vòng lặp
            logger.exception("step.scan.failed", error=str(exc))
            return State.ERROR

    def _build_targets(self, detections: list[dict]) -> list[dict]:
        targets: list[dict] = []
        for detection in detections:
            class_id = int(detection["class_id"])
            if class_id < 0 or class_id >= len(self.class_names):
                continue
            class_name = self.class_names[class_id]
            group_id = self.mapping.get(class_name)
            if group_id is None:
                continue

            u_center, v_center = detection["center"]
            x_world, y_world = self.transformer.pixel_to_world(u_center, v_center)
            targets.append(
                {
                    "group_id": int(group_id),
                    "x": x_world,
                    "y": y_world,
                    "conf": float(detection["conf"]),
                    "class": class_name,
                }
            )
        return targets

    def _state_schedule(self) -> State:
        scheduler_cfg = self.settings["scheduler"]
        self.scheduled = optimize_trajectory(
            self.targets,
            start_position=tuple(scheduler_cfg["start_position_mm"]),
            fragile_group_id=int(scheduler_cfg["fragile_group_id"]),
        )
        logger.info(
            "step.schedule.done",
            order=[
                {"group_id": t["group_id"], "x": t["x"], "y": t["y"]}
                for t in self.scheduled
            ],
        )
        return State.EXECUTE

    def _state_execute(self) -> State:
        for index, target in enumerate(self.scheduled):
            logger.info(
                "step.pick.start",
                index=index,
                group_id=target["group_id"],
                x=target["x"],
                y=target["y"],
            )
            ok = self.comm.send_pick_command(
                group_id=target["group_id"],
                x=target["x"],
                y=target["y"],
            )
            if not ok:
                self.failed_target = target
                logger.error("step.pick.failed", index=index, target=target)
                return State.ERROR
            logger.info("step.pick.done", index=index)

        logger.info("cycle.success", n_picked=len(self.scheduled))
        self._end_cycle()
        return State.SCAN

    def _state_error(self) -> State:
        logger.warning("cycle.aborted", failed_target=self.failed_target)
        self.failed_target = None
        self._end_cycle()
        time.sleep(self.loop_delay)
        return State.SCAN

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        handlers = {
            State.INIT: self._state_init,
            State.SCAN: self._state_scan,
            State.SCHEDULE: self._state_schedule,
            State.EXECUTE: self._state_execute,
            State.ERROR: self._state_error,
        }
        try:
            while self.state != State.SHUTDOWN:
                self.state = handlers[self.state]()
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


def main() -> None:
    configure_logging()
    AutoSortPi().run()


if __name__ == "__main__":
    main()
