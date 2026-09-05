"""Exception hierarchy dùng chung cho toàn bộ pipeline.

FSM (``main.py``) bắt các exception theo nhóm để chọn recovery policy phù hợp
(§3 task.md): lỗi camera, lỗi model, lỗi calibration, lỗi target, lỗi UART,
lỗi MCU, timeout, lỗi cấu hình đều là các subclass riêng biệt của
``AutoSortError`` để không bị gộp chung thành "Exception" mơ hồ.
"""

from __future__ import annotations


class AutoSortError(Exception):
    """Base cho mọi lỗi có thể dự đoán được trong pipeline."""


# --------------------------------------------------------------- config
class ConfigError(AutoSortError):
    """Configuration file thiếu field, sai kiểu, hoặc giá trị không hợp lệ."""


# --------------------------------------------------------------- camera
class CameraError(AutoSortError):
    """Lỗi chung liên quan tới camera."""


class CameraDisconnectedError(CameraError):
    """Camera bị rút / mất kết nối trong lúc đang chạy."""


class CameraTimeoutError(CameraError):
    """Không lấy được frame mới trong thời gian quy định."""


# --------------------------------------------------------------- model
class ModelError(AutoSortError):
    """Lỗi chung liên quan tới model TFLite."""


class ModelLoadError(ModelError):
    """Không load được model, hoặc model không tương thích (shape/dtype/class)."""


class ModelInferenceError(ModelError):
    """Lỗi xảy ra khi gọi ``interpreter.invoke()``."""


class ModelDecodingError(ModelError):
    """Output tensor của model không đúng layout mong đợi, không decode được."""


# --------------------------------------------------------------- calibration
class CalibrationError(AutoSortError):
    """Calibration file thiếu, sai định dạng, hoặc là placeholder (identity)."""


# --------------------------------------------------------------- target
class TargetValidationError(AutoSortError):
    """Một target cụ thể không hợp lệ để đưa vào scheduler/UART."""


class SafetyViolationError(AutoSortError):
    """Target vi phạm giới hạn an toàn của workspace robot."""


# --------------------------------------------------------------- uart / mcu
class UartError(AutoSortError):
    """Lỗi chung liên quan tới UART transport (không phải lỗi MCU logic)."""


class UartDisconnectedError(UartError):
    """Cổng serial bị rút / không mở được."""


class UartTimeoutError(UartError):
    """Timeout khi chờ ACK hoặc DONE (nghi mất gói)."""


class McuError(AutoSortError):
    """MCU trả về ``!ERR,CODE`` — lỗi phần cứng/thực thi, không phải mất gói."""

    def __init__(self, code: str | None, message: str = "") -> None:
        self.code = code
        super().__init__(message or f"MCU reported error code={code}")
