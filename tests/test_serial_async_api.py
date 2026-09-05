"""Kiểm thử API non-blocking dùng cho FSM tick-based (send_command_async / read_one_response)."""

from __future__ import annotations

from modules.serial_comm import SerialCommunicator
from modules.uart_protocol import ResponseKind


class FakeSerial:
    def __init__(self):
        self.queue: list[bytes] = []
        self.writes: list[bytes] = []
        self.is_open = True

    def reset_input_buffer(self):
        self.queue.clear()

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def readline(self):
        return self.queue.pop(0) if self.queue else b""

    def close(self):
        self.is_open = False


def test_send_command_async_writes_correct_frame():
    ser = FakeSerial()
    comm = SerialCommunicator(ser=ser)
    assert comm.send_command_async("PICK,1,150.2,320.0,0.0") is True
    assert ser.writes == [b"$PICK,1,150.2,320.0,0.0*09\n"]


def test_read_one_response_returns_empty_when_no_data():
    ser = FakeSerial()
    comm = SerialCommunicator(ser=ser)
    assert comm.read_one_response().kind == ResponseKind.EMPTY


def test_read_one_response_parses_ack_then_done_across_ticks():
    ser = FakeSerial()
    comm = SerialCommunicator(ser=ser)
    ser.queue.extend([b"!ACK\n", b"!DONE\n"])
    assert comm.read_one_response().kind == ResponseKind.ACK
    assert comm.read_one_response().kind == ResponseKind.DONE
    assert comm.read_one_response().kind == ResponseKind.EMPTY


def test_read_one_response_surfaces_err_with_code():
    ser = FakeSerial()
    comm = SerialCommunicator(ser=ser)
    ser.queue.append(b"!ERR,7\n")
    parsed = comm.read_one_response()
    assert parsed.kind == ResponseKind.ERR
    assert parsed.code == "7"
