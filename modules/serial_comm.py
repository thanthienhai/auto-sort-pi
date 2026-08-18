"""UART command framing and blocking ACK/DONE handshake.

Frame gửi đi: ``$<payload>*<CHECKSUM>\n`` với ``CHECKSUM`` là XOR của toàn bộ
``payload`` (từ sau ``$`` đến trước ``*``), Hex 2 ký tự.

Chính sách theo README:
- Chờ ``!ACK`` sau khi gửi; nếu **timeout** (nghi mất gói) thì retry tối đa
  ``max_retries`` lần.
- Nhận ``!ACK`` rồi chờ ``!DONE`` để xác nhận hoàn tất.
- Nhận ``!ERR,CODE`` (lỗi phần cứng / thực thi) hoặc timeout khi chờ ``!DONE`` thì
  **dừng** target hiện tại — không retry, vì đây không phải lỗi mất gói.
"""

from __future__ import annotations

import time
from typing import Any

from .logging_setup import get_logger

logger = get_logger("autosort.uart")

# Kết quả một pha chờ phản hồi.
_ACK = "ack"
_DONE = "done"
_ERR = "err"
_TIMEOUT = "timeout"


class SerialCommunicator:
    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        ack_timeout: float = 2.0,
        action_timeout: float = 15.0,
        max_retries: int = 3,
        ser: Any | None = None,
        boot_delay: float = 2.0,
    ) -> None:
        self.ack_timeout = ack_timeout
        self.action_timeout = action_timeout
        self.max_retries = max_retries
        if ser is not None:
            # Cho phép inject một serial-like object (test / adapter khác).
            self.ser = ser
            return

        import serial  # lazy import: chỉ cần pyserial khi chạy thật trên phần cứng.

        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        if boot_delay:
            time.sleep(boot_delay)

    # ------------------------------------------------------------- framing

    @staticmethod
    def calc_checksum(payload: str) -> str:
        checksum = 0
        for char in payload:
            checksum ^= ord(char)
        return f"{checksum:02X}"

    @classmethod
    def build_frame(cls, payload: str) -> str:
        return f"${payload}*{cls.calc_checksum(payload)}\n"

    @classmethod
    def verify_frame(cls, frame: str) -> bool:
        """Kiểm tra checksum của một frame ``$payload*XX``."""
        frame = frame.strip()
        if not frame.startswith("$") or "*" not in frame:
            return False
        payload, _, checksum = frame[1:].partition("*")
        return checksum.upper() == cls.calc_checksum(payload)

    # ------------------------------------------------------------- commands

    def send_pick_command(
        self,
        group_id: int,
        x: float,
        y: float,
        angle: float = 0.0,
    ) -> bool:
        payload = f"PICK,{group_id},{x:.1f},{y:.1f},{angle:.1f}"
        return self._send_with_handshake(payload)

    def scan(self) -> bool:
        return self._send_with_handshake("SCAN")

    def purge(self) -> bool:
        return self._send_with_handshake("PURGE")

    # ------------------------------------------------------------ handshake

    def _send_with_handshake(self, payload: str) -> bool:
        frame = self.build_frame(payload).encode("ascii")
        for attempt in range(self.max_retries + 1):
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            self.ser.flush()
            logger.info("step.uart.tx", payload=payload, attempt=attempt)

            status, code = self._wait_for(self.ack_timeout, expect=_ACK)
            if status == _TIMEOUT:
                # Nghi mất gói -> retry.
                logger.warning("step.uart.ack_timeout", payload=payload, attempt=attempt)
                continue
            if status == _ERR:
                logger.error("step.uart.err", phase="ack", code=code, payload=payload)
                return False

            logger.info("step.uart.ack", payload=payload)
            done_status, done_code = self._wait_for(self.action_timeout, expect=_DONE)
            if done_status == _DONE:
                logger.info("step.uart.done", payload=payload)
                return True
            if done_status == _ERR:
                logger.error("step.uart.err", phase="done", code=done_code, payload=payload)
            else:
                logger.error("step.uart.done_timeout", payload=payload)
            # ERR / timeout sau khi đã ACK: dừng target hiện tại, không retry.
            return False

        logger.error("step.uart.failed", payload=payload, retries=self.max_retries)
        return False

    def _wait_for(self, timeout: float, expect: str) -> tuple[str, str | None]:
        """Chờ đến khi nhận ``expect`` (``!ACK`` / ``!DONE``), ``!ERR`` hoặc timeout.

        Trả về ``(status, error_code)`` với status ∈ {ack, done, err, timeout}.
        """
        deadline = time.monotonic() + timeout
        target_line = "!ACK" if expect == _ACK else "!DONE"
        while time.monotonic() < deadline:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line == target_line:
                return (expect, None)
            if line.startswith("!ERR"):
                _, _, code = line.partition(",")
                return (_ERR, code or None)
            # Dòng lạ (vd !ACK khi đang chờ !DONE, hoặc nhiễu) -> tiếp tục chờ.
        return (_TIMEOUT, None)

    def close(self) -> None:
        if getattr(self.ser, "is_open", False):
            self.ser.close()
