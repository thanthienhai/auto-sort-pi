"""Kiểm thử đóng frame, checksum và handshake UART (dùng serial giả)."""

from modules.serial_comm import SerialCommunicator


class FakeSerial:
    """Serial-like giả: mỗi lần ``write`` nạp một batch phản hồi tương ứng attempt."""

    def __init__(self, batches):
        self.batches = [list(b) for b in batches]
        self.queue = []
        self.writes = 0
        self.is_open = True

    def reset_input_buffer(self):
        self.queue.clear()

    def write(self, data):
        self.last_write = data
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


def _comm(batches, **kw):
    ser = FakeSerial(batches)
    comm = SerialCommunicator(ser=ser, ack_timeout=0.02, action_timeout=0.02, **kw)
    return comm, ser


# --------------------------------------------------------------- framing

def test_checksum_matches_textual_spec():
    # XOR "từ sau $ đến trước *" của payload ví dụ trong README.
    assert SerialCommunicator.calc_checksum("PICK,1,150.2,320.0,0.0") == "09"


def test_build_and_verify_frame():
    frame = SerialCommunicator.build_frame("PICK,1,150.2,320.0,0.0")
    assert frame == "$PICK,1,150.2,320.0,0.0*09\n"
    assert SerialCommunicator.verify_frame(frame) is True
    assert SerialCommunicator.verify_frame("$PICK,1,150.2,320.0,0.0*FF") is False


# ------------------------------------------------------------ handshake

def test_happy_path_returns_true():
    comm, ser = _comm([[b"!ACK\n", b"!DONE\n"]])
    assert comm.send_pick_command(1, 150.2, 320.0) is True
    assert ser.writes == 1
    assert ser.last_write == b"$PICK,1,150.2,320.0,0.0*09\n"


def test_err_on_ack_aborts_without_retry():
    comm, ser = _comm([[b"!ERR,12\n"]], max_retries=3)
    assert comm.send_pick_command(1, 10.0, 20.0) is False
    assert ser.writes == 1  # lỗi MCU -> không retry


def test_err_after_ack_aborts():
    comm, ser = _comm([[b"!ACK\n", b"!ERR,7\n"]], max_retries=3)
    assert comm.send_pick_command(1, 10.0, 20.0) is False
    assert ser.writes == 1


def test_ack_timeout_retries_then_gives_up():
    comm, ser = _comm([[], [], []], max_retries=2)  # không bao giờ có ACK
    assert comm.send_pick_command(1, 10.0, 20.0) is False
    assert ser.writes == 3  # 1 lần đầu + 2 retry


def test_ack_timeout_then_success_on_retry():
    comm, ser = _comm([[], [b"!ACK\n", b"!DONE\n"]], max_retries=3)
    assert comm.send_pick_command(1, 10.0, 20.0) is True
    assert ser.writes == 2
