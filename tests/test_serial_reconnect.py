"""Kiểm thử SerialCommunicator: reconnect, last_error typing, malformed lines (§13/§14)."""

from __future__ import annotations

import pytest

from modules.exceptions import McuError, UartDisconnectedError, UartTimeoutError
from modules.serial_comm import SerialCommunicator


class FakeSerial:
    def __init__(self, batches, fail_writes=0):
        self.batches = [list(b) for b in batches]
        self.queue = []
        self.writes = 0
        self.fail_writes = fail_writes  # số lần write() đầu tiên raise exception
        self.is_open = True

    def reset_input_buffer(self):
        self.queue.clear()

    def write(self, data):
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise OSError("device reports readiness to read but returned no data")
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


def test_malformed_line_ignored_while_waiting_for_ack():
    ser = FakeSerial([[b"\x00garbage\n", b"!ACK\n", b"!DONE\n"]])
    comm = SerialCommunicator(ser=ser, ack_timeout=0.5, action_timeout=0.5)
    assert comm.send_pick_command(1, 10.0, 20.0) is True


def test_last_error_is_mcu_error_on_ack_err():
    ser = FakeSerial([[b"!ERR,42\n"]])
    comm = SerialCommunicator(ser=ser)
    assert comm.send_pick_command(1, 10.0, 20.0) is False
    assert isinstance(comm.last_error, McuError)
    assert comm.last_error.code == "42"


def test_last_error_is_timeout_on_ack_timeout():
    ser = FakeSerial([[], []], )
    comm = SerialCommunicator(ser=ser, ack_timeout=0.01, action_timeout=0.01, max_retries=1)
    assert comm.send_pick_command(1, 10.0, 20.0) is False
    assert isinstance(comm.last_error, UartTimeoutError)


def test_reconnect_succeeds_after_transient_write_failure(monkeypatch):
    bad_ser = FakeSerial([], fail_writes=1)
    comm = SerialCommunicator(ser=bad_ser, ack_timeout=0.5, action_timeout=0.5, max_reconnect_attempts=2)
    comm._injected = False  # cho phép _reconnect() thử mở lại thay vì bỏ cuộc ngay

    good_ser = FakeSerial([[b"!ACK\n", b"!DONE\n"]])
    monkeypatch.setattr(comm, "_open_serial", lambda: good_ser)

    assert comm.send_pick_command(1, 10.0, 20.0) is True
    assert good_ser.writes == 1


def test_disconnected_raises_after_reconnect_attempts_exhausted(monkeypatch):
    bad_ser = FakeSerial([], fail_writes=999)
    comm = SerialCommunicator(
        ser=bad_ser,
        ack_timeout=0.05,
        action_timeout=0.05,
        max_reconnect_attempts=2,
        reconnect_backoff_seconds=0.01,
    )
    comm._injected = False

    def _always_fail_open():
        raise OSError("could not open port")

    monkeypatch.setattr(comm, "_open_serial", _always_fail_open)

    with pytest.raises(UartDisconnectedError):
        comm.send_pick_command(1, 10.0, 20.0)
