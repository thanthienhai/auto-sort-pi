"""UART protocol framing/parsing riêng biệt khỏi transport (§13 task.md).

Tách khỏi ``serial_comm.py`` để business logic (framing, checksum, phân loại
response) không phụ thuộc trực tiếp vào ``serial.Serial`` và test được thuần
bằng string, không cần serial giả.

Frame gửi đi: ``$<payload>*<CHECKSUM>\n`` — checksum XOR (Hex 2 ký tự) của
``payload`` (từ sau ``$`` đến trước ``*``).

Response nhận về: ``!ACK`` / ``!DONE`` / ``!ERR,CODE``. Một dòng không khớp
3 dạng trên (nhiễu, dòng rỗng, response lạ) được phân loại ``MALFORMED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ResponseKind(Enum):
    ACK = auto()
    DONE = auto()
    ERR = auto()
    MALFORMED = auto()
    EMPTY = auto()  # chưa có dữ liệu trong tick đọc này (không phải lỗi)


@dataclass
class ParsedResponse:
    kind: ResponseKind
    code: str | None = None
    raw: str = ""


def calc_checksum(payload: str) -> str:
    checksum = 0
    for char in payload:
        checksum ^= ord(char)
    return f"{checksum:02X}"


def build_frame(payload: str) -> str:
    return f"${payload}*{calc_checksum(payload)}\n"


def verify_frame(frame: str) -> bool:
    """Kiểm tra checksum của một frame ``$payload*XX``."""
    frame = frame.strip()
    if not frame.startswith("$") or "*" not in frame:
        return False
    payload, _, checksum = frame[1:].partition("*")
    return checksum.upper() == calc_checksum(payload)


def build_pick_payload(group_id: int, x: float, y: float, angle: float = 0.0) -> str:
    return f"PICK,{group_id},{x:.1f},{y:.1f},{angle:.1f}"


def parse_response_line(line: str) -> ParsedResponse:
    """Phân loại một dòng response đã đọc từ serial (đã strip)."""
    line = line.strip()
    if not line:
        return ParsedResponse(ResponseKind.EMPTY, raw=line)
    if line == "!ACK":
        return ParsedResponse(ResponseKind.ACK, raw=line)
    if line == "!DONE":
        return ParsedResponse(ResponseKind.DONE, raw=line)
    if line.startswith("!ERR"):
        _, _, code = line.partition(",")
        return ParsedResponse(ResponseKind.ERR, code=code or None, raw=line)
    return ParsedResponse(ResponseKind.MALFORMED, raw=line)
