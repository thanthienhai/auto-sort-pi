"""Validate cấu hình khi startup (§20 task.md).

Configuration invalid thì hệ thống không được vào ``READY`` — ``main.py`` gọi
``validate_settings`` trong state ``SELF_CHECK`` và chuyển thẳng ``SAFE_STOP``
nếu có lỗi, kèm danh sách lý do cụ thể (không phải một exception mơ hồ).
"""

from __future__ import annotations

from typing import Any

from .exceptions import ConfigError

REQUIRED_SECTIONS = (
    "camera",
    "inference",
    "calibration",
    "workspace",
    "scheduler",
    "robot",
    "dedup",
    "uart",
    "runtime",
    "logging",
)


def _check_range(errors: list[str], name: str, value: Any, lo: float, hi: float) -> None:
    try:
        if not (lo <= float(value) <= hi):
            errors.append(f"{name}={value!r} phải trong [{lo}, {hi}]")
    except (TypeError, ValueError):
        errors.append(f"{name}={value!r} không phải số hợp lệ")


def _check_positive(errors: list[str], name: str, value: Any) -> None:
    try:
        if float(value) <= 0:
            errors.append(f"{name}={value!r} phải > 0")
    except (TypeError, ValueError):
        errors.append(f"{name}={value!r} không phải số hợp lệ")


def validate_settings(settings: dict[str, Any]) -> list[str]:
    """Trả về danh sách lỗi (rỗng nếu hợp lệ). Không raise — để caller quyết định."""
    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in settings:
            errors.append(f"Thiếu section bắt buộc: '{section}'")
    if errors:
        return errors  # không kiểm tra sâu hơn nếu thiếu cả section

    camera = settings["camera"]
    _check_positive(errors, "camera.width", camera.get("width"))
    _check_positive(errors, "camera.height", camera.get("height"))
    _check_positive(errors, "camera.fps", camera.get("fps"))
    _check_positive(errors, "camera.max_frame_age_seconds", camera.get("max_frame_age_seconds"))

    inference = settings["inference"]
    if not inference.get("model_path"):
        errors.append("inference.model_path không được rỗng")
    _check_range(errors, "inference.confidence_threshold", inference.get("confidence_threshold"), 0.0, 1.0)
    _check_range(errors, "inference.nms_iou_threshold", inference.get("nms_iou_threshold"), 0.0, 1.0)

    calibration = settings["calibration"]
    if not calibration.get("file"):
        errors.append("calibration.file không được rỗng")
    _check_positive(errors, "calibration.max_reprojection_error_mm", calibration.get("max_reprojection_error_mm"))

    workspace = settings["workspace"]
    for key in ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm"):
        if key not in workspace:
            errors.append(f"workspace.{key} thiếu")
    if "x_min_mm" in workspace and "x_max_mm" in workspace and workspace["x_min_mm"] >= workspace["x_max_mm"]:
        errors.append("workspace.x_min_mm phải nhỏ hơn workspace.x_max_mm")
    if "y_min_mm" in workspace and "y_max_mm" in workspace and workspace["y_min_mm"] >= workspace["y_max_mm"]:
        errors.append("workspace.y_min_mm phải nhỏ hơn workspace.y_max_mm")

    scheduler = settings["scheduler"]
    if "start_position_mm" not in scheduler or len(scheduler["start_position_mm"]) != 2:
        errors.append("scheduler.start_position_mm phải là [x, y]")
    if "drop_positions_mm" not in scheduler or not scheduler["drop_positions_mm"]:
        errors.append("scheduler.drop_positions_mm không được rỗng")
    else:
        for group_id, pos in scheduler["drop_positions_mm"].items():
            if len(pos) != 2:
                errors.append(f"scheduler.drop_positions_mm[{group_id}] phải là [x, y]")

    robot_safety = settings.get("robot", {}).get("safety", {})
    for key in ("x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm"):
        if key not in robot_safety:
            errors.append(f"robot.safety.{key} thiếu")

    uart = settings["uart"]
    if not uart.get("port"):
        errors.append("uart.port không được rỗng")
    _check_positive(errors, "uart.baudrate", uart.get("baudrate"))
    _check_positive(errors, "uart.ack_timeout", uart.get("ack_timeout"))
    _check_positive(errors, "uart.action_timeout", uart.get("action_timeout"))

    dedup = settings["dedup"]
    _check_positive(errors, "dedup.suppression_radius_mm", dedup.get("suppression_radius_mm"))
    _check_positive(errors, "dedup.cooldown_seconds", dedup.get("cooldown_seconds"))

    runtime = settings["runtime"]
    _check_positive(errors, "runtime.loop_delay_seconds", runtime.get("loop_delay_seconds"))

    return errors


def assert_valid_settings(settings: dict[str, Any]) -> None:
    errors = validate_settings(settings)
    if errors:
        raise ConfigError("Configuration invalid:\n  - " + "\n  - ".join(errors))
