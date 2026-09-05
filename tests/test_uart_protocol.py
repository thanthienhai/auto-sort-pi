"""Kiểm thử modules.uart_protocol: framing/parsing thuần (không cần serial giả)."""

from __future__ import annotations

from modules.uart_protocol import (
    ResponseKind,
    build_frame,
    build_pick_payload,
    calc_checksum,
    parse_response_line,
    verify_frame,
)


def test_build_pick_payload_matches_readme_example():
    payload = build_pick_payload(1, 150.2, 320.0, 0.0)
    assert payload == "PICK,1,150.2,320.0,0.0"
    assert calc_checksum(payload) == "09"


def test_build_frame_round_trips_verify_frame():
    frame = build_frame("PICK,1,150.2,320.0,0.0")
    assert frame == "$PICK,1,150.2,320.0,0.0*09\n"
    assert verify_frame(frame) is True
    assert verify_frame("$PICK,1,150.2,320.0,0.0*FF") is False


def test_parse_ack_done_err_empty_malformed():
    assert parse_response_line("!ACK").kind == ResponseKind.ACK
    assert parse_response_line("!DONE").kind == ResponseKind.DONE
    err = parse_response_line("!ERR,12")
    assert err.kind == ResponseKind.ERR
    assert err.code == "12"
    assert parse_response_line("").kind == ResponseKind.EMPTY
    assert parse_response_line("garbage\x00data").kind == ResponseKind.MALFORMED


def test_parse_err_without_code():
    parsed = parse_response_line("!ERR")
    assert parsed.kind == ResponseKind.ERR
    assert parsed.code is None
