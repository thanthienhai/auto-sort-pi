"""UART command framing and blocking ACK/DONE handshake."""

from __future__ import annotations

import time

import serial


class SerialCommunicator:
    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        ack_timeout: float = 2.0,
        action_timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        self.ack_timeout = ack_timeout
        self.action_timeout = action_timeout
        self.max_retries = max_retries
        time.sleep(2.0)

    @staticmethod
    def calc_checksum(payload: str) -> str:
        checksum = 0
        for char in payload:
            checksum ^= ord(char)
        return f"{checksum:02X}"

    @classmethod
    def build_frame(cls, payload: str) -> str:
        return f"${payload}*{cls.calc_checksum(payload)}\n"

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

    def _send_with_handshake(self, payload: str) -> bool:
        frame = self.build_frame(payload).encode("ascii")
        for _ in range(self.max_retries + 1):
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            self.ser.flush()

            if not self._wait_for_ack():
                continue
            return self._wait_for_done_or_error()
        return False

    def _wait_for_ack(self) -> bool:
        deadline = time.monotonic() + self.ack_timeout
        while time.monotonic() < deadline:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line == "!ACK":
                return True
            if line.startswith("!ERR"):
                return False
        return False

    def _wait_for_done_or_error(self) -> bool:
        deadline = time.monotonic() + self.action_timeout
        while time.monotonic() < deadline:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line == "!DONE":
                return True
            if line.startswith("!ERR"):
                return False
        return False

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()
