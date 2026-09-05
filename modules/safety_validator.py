"""Robot safety layer (§12 task.md).

Gate cuối cùng ngay trước khi gửi lệnh ``PICK`` qua UART. Không có exception ở
đây — trả về ``SafetyCheckResult`` tường minh (ok/reason) để FSM log lý do và
chuyển sang trạng thái phù hợp (bỏ target, không phải crash cycle). AI model /
scheduler không bao giờ được phép gọi thẳng ``SerialCommunicator`` mà bỏ qua
bước này.
"""

from __future__ import annotations

from dataclasses import dataclass

from .logging_setup import get_logger

logger = get_logger("autosort.safety")


@dataclass
class SafetyLimits:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    min_angle_deg: float = -180.0
    max_angle_deg: float = 180.0


@dataclass
class SafetyCheckResult:
    ok: bool
    reason: str = ""


class SafetyValidator:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def check(self, target: dict) -> SafetyCheckResult:
        x, y = target.get("x"), target.get("y")
        angle = target.get("angle", 0.0)

        if x is None or y is None:
            result = SafetyCheckResult(False, "target thiếu toạ độ x/y")
        elif not (self.limits.x_min_mm <= x <= self.limits.x_max_mm):
            result = SafetyCheckResult(
                False,
                f"X={x} ngoài giới hạn an toàn [{self.limits.x_min_mm}, {self.limits.x_max_mm}]",
            )
        elif not (self.limits.y_min_mm <= y <= self.limits.y_max_mm):
            result = SafetyCheckResult(
                False,
                f"Y={y} ngoài giới hạn an toàn [{self.limits.y_min_mm}, {self.limits.y_max_mm}]",
            )
        elif not (self.limits.min_angle_deg <= angle <= self.limits.max_angle_deg):
            result = SafetyCheckResult(
                False,
                f"angle={angle} ngoài giới hạn an toàn "
                f"[{self.limits.min_angle_deg}, {self.limits.max_angle_deg}]",
            )
        else:
            result = SafetyCheckResult(True)

        if not result.ok:
            logger.error("step.safety.rejected", target=target, reason=result.reason)
        return result
