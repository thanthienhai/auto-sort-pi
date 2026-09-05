"""UART command framing and blocking ACK/DONE handshake.

Framing/parsing được tách sang ``modules.uart_protocol`` (§13 task.md) — module
này chỉ còn lo transport: mở serial port, gửi frame, đọc response, retry,
reconnect khi mất kết nối, và chính sách handshake theo README:

- Chờ ``!ACK`` sau khi gửi; nếu **timeout** (nghi mất gói) thì retry tối đa
  ``max_retries`` lần.
- Nhận ``!ACK`` rồi chờ ``!DONE`` để xác nhận hoàn tất.
- Nhận ``!ERR,CODE`` (lỗi phần cứng / thực thi) hoặc timeout khi chờ ``!DONE``
  thì **dừng** target hiện tại — không retry, vì đây không phải lỗi mất gói.
- Response không nhận dạng được (malformed / nhiễu) bị bỏ qua và tiếp tục chờ
  cho tới hết timeout, không coi là lỗi tức thì.

Business logic (``main.py``) không bao giờ gọi thẳng ``serial.Serial`` — luôn
qua ``SerialCommunicator`` để có thể inject ``FakeSerial`` khi test (§14).
"""

from __future__ import annotations

import time
from typing import Any

from .exceptions import McuError, UartDisconnectedError, UartTimeoutError
from .logging_setup import get_logger
from .uart_protocol import (
    ParsedResponse,
    ResponseKind,
    build_frame,
    build_pick_payload,
    calc_checksum,
    parse_response_line,
    verify_frame,
)

logger = get_logger("autosort.uart")


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
        max_reconnect_attempts: int = 3,
        reconnect_backoff_seconds: float = 1.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ack_timeout = ack_timeout
        self.action_timeout = action_timeout
        self.max_retries = max_retries
        self.boot_delay = boot_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self._injected = ser is not None
        self.last_error: McuError | UartTimeoutError | None = None

        if ser is not None:
            # Cho phép inject một serial-like object (test / adapter khác).
            self.ser = ser
            return

        self.ser = self._open_serial()

    # ------------------------------------------------------------- transport

    def _open_serial(self):
        import serial  # lazy import: chỉ cần pyserial khi chạy thật trên phần cứng.

        ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
        )
        if self.boot_delay:
            time.sleep(self.boot_delay)
        return ser

    def _reconnect(self) -> bool:
        if self._injected:
            # FakeSerial (test) không hỗ trợ reconnect thật — coi như thất bại ngay.
            return False
        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.warning(
                "step.uart.reconnect_attempt", attempt=attempt, max_attempts=self.max_reconnect_attempts
            )
            try:
                self.close()
                self.ser = self._open_serial()
                logger.info("step.uart.reconnected", attempt=attempt)
                return True
            except Exception as exc:  # noqa: BLE001 - lỗi driver/OS khi mở lại cổng
                logger.warning("step.uart.reconnect_failed", attempt=attempt, error=str(exc))
                time.sleep(self.reconnect_backoff_seconds)
        logger.error("step.uart.disconnected", attempts=self.max_reconnect_attempts)
        return False

    # ------------------------------------------------------------- framing
    # Alias tiện dùng từ bên ngoài (giữ API cũ) — nguồn thật ở uart_protocol.

    calc_checksum = staticmethod(calc_checksum)
    build_frame = staticmethod(build_frame)
    verify_frame = staticmethod(verify_frame)

    # ------------------------------------------------------------- commands

    def send_pick_command(
        self,
        group_id: int,
        x: float,
        y: float,
        angle: float = 0.0,
    ) -> bool:
        payload = build_pick_payload(group_id, x, y, angle)
        return self._send_with_handshake(payload)

    def scan(self) -> bool:
        return self._send_with_handshake("SCAN")

    def purge(self) -> bool:
        return self._send_with_handshake("PURGE")

    # ------------------------------------------------------------ handshake

    def _write_frame(self, payload: str) -> bool:
        """Ghi một frame ra serial; tự reconnect nếu phát hiện mất kết nối."""
        frame = build_frame(payload).encode("ascii")
        try:
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            self.ser.flush()
            return True
        except Exception as exc:  # noqa: BLE001 - pyserial raises SerialException trên disconnect
            logger.warning("step.uart.write_failed", payload=payload, error=str(exc))
            if self._reconnect():
                try:
                    self.ser.reset_input_buffer()
                    self.ser.write(frame)
                    self.ser.flush()
                    return True
                except Exception as exc2:  # noqa: BLE001
                    logger.error("step.uart.write_failed_after_reconnect", error=str(exc2))
            return False

    def _send_with_handshake(self, payload: str) -> bool:
        self.last_error = None
        for attempt in range(self.max_retries + 1):
            if not self._write_frame(payload):
                raise UartDisconnectedError(f"UART mất kết nối, không gửi được: {payload}")

            logger.info("step.uart.tx", payload=payload, attempt=attempt)

            ack = self._wait_for(self.ack_timeout, expect=ResponseKind.ACK)
            if ack.kind == ResponseKind.EMPTY:
                # Nghi mất gói -> retry.
                logger.warning("step.uart.ack_timeout", payload=payload, attempt=attempt)
                self.last_error = UartTimeoutError(f"ACK timeout (attempt {attempt}): {payload}")
                continue
            if ack.kind == ResponseKind.ERR:
                logger.error("step.uart.err", phase="ack", code=ack.code, payload=payload)
                self.last_error = McuError(ack.code)
                return False

            logger.info("step.uart.ack", payload=payload)
            done = self._wait_for(self.action_timeout, expect=ResponseKind.DONE)
            if done.kind == ResponseKind.DONE:
                logger.info("step.uart.done", payload=payload)
                return True
            if done.kind == ResponseKind.ERR:
                logger.error("step.uart.err", phase="done", code=done.code, payload=payload)
                self.last_error = McuError(done.code)
            else:
                logger.error("step.uart.done_timeout", payload=payload)
                self.last_error = UartTimeoutError(f"DONE timeout: {payload}")
            # ERR / timeout sau khi đã ACK: dừng target hiện tại, không retry.
            return False

        logger.error("step.uart.failed", payload=payload, retries=self.max_retries)
        return False

    def _wait_for(self, timeout: float, expect: ResponseKind) -> ParsedResponse:
        """Chờ đến khi nhận ``expect`` (``ACK``/``DONE``), ``ERR`` hoặc hết timeout.

        Dòng ``MALFORMED`` (nhiễu, response không nhận dạng được, hoặc dòng lạ —
        ví dụ ``!ACK`` lặp lại khi đang chờ ``!DONE``) bị bỏ qua và tiếp tục chờ,
        không được coi là lỗi tức thì.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw_line = self.ser.readline()
            line = raw_line.decode("ascii", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            parsed = parse_response_line(line)
            if parsed.kind == ResponseKind.EMPTY:
                continue
            if parsed.kind == expect:
                return parsed
            if parsed.kind == ResponseKind.ERR:
                return parsed
            if parsed.kind == ResponseKind.MALFORMED:
                logger.warning("step.uart.malformed_response", raw=parsed.raw)
            # Dòng nhận dạng được nhưng không phải cái đang chờ (vd ACK lặp lại
            # trong lúc chờ DONE) -> tiếp tục chờ tới deadline.
        return ParsedResponse(ResponseKind.EMPTY)

    # --------------------------------------------------------- FSM-friendly API
    #
    # send_pick_command()/scan()/purge() ở trên chờ (blocking, bounded bởi
    # ack_timeout/action_timeout) trong một lời gọi — tiện cho script/dev tool.
    # main.py (FSM tick-based) dùng cặp API dưới đây thay thế: gửi lệnh không
    # chờ, rồi mỗi tick gọi ``read_one_response()`` (không block quá lâu) và tự
    # quản lý deadline qua các state WAITING_ACK / WAITING_DONE.

    def send_command_async(self, payload: str) -> bool:
        """Gửi frame, không chờ response. Trả về False nếu UART mất kết nối
        hẳn (đã thử reconnect hết ``max_reconnect_attempts``)."""
        return self._write_frame(payload)

    def read_one_response(self) -> ParsedResponse:
        """Đọc (tối đa) một dòng response, không lặp tới deadline như ``_wait_for``.

        Block tối đa ``self.timeout`` giây (timeout đọc của serial port) nếu
        chưa có dữ liệu — FSM gọi hàm này mỗi tick nên nên cấu hình
        ``uart.timeout`` nhỏ (vd 0.1-0.2s) để vòng lặp phản hồi nhanh.
        """
        raw_line = self.ser.readline()
        line = raw_line.decode("ascii", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
        parsed = parse_response_line(line)
        if parsed.kind == ResponseKind.MALFORMED:
            logger.warning("step.uart.malformed_response", raw=parsed.raw)
        return parsed

    def close(self) -> None:
        if getattr(self.ser, "is_open", False):
            self.ser.close()
