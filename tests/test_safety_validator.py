"""Kiểm thử SafetyValidator (§12 task.md): gate cuối trước khi gửi UART."""

from __future__ import annotations

from modules.safety_validator import SafetyLimits, SafetyValidator


def _validator():
    limits = SafetyLimits(x_min_mm=0.0, x_max_mm=400.0, y_min_mm=0.0, y_max_mm=300.0)
    return SafetyValidator(limits)


def test_target_within_bounds_is_ok():
    result = _validator().check({"x": 100.0, "y": 100.0})
    assert result.ok is True
    assert result.reason == ""


def test_target_x_below_min_rejected():
    result = _validator().check({"x": -10.0, "y": 100.0})
    assert result.ok is False
    assert "X=" in result.reason


def test_target_x_above_max_rejected():
    result = _validator().check({"x": 500.0, "y": 100.0})
    assert result.ok is False


def test_target_y_out_of_range_rejected():
    result = _validator().check({"x": 100.0, "y": 999.0})
    assert result.ok is False
    assert "Y=" in result.reason


def test_target_missing_coordinates_rejected():
    result = _validator().check({})
    assert result.ok is False
    assert "thiếu" in result.reason


def test_target_angle_out_of_range_rejected():
    limits = SafetyLimits(x_min_mm=0, x_max_mm=400, y_min_mm=0, y_max_mm=300, min_angle_deg=-90, max_angle_deg=90)
    validator = SafetyValidator(limits)
    result = validator.check({"x": 100.0, "y": 100.0, "angle": 120.0})
    assert result.ok is False
    assert "angle" in result.reason


def test_boundary_values_are_inclusive():
    result = _validator().check({"x": 0.0, "y": 300.0})
    assert result.ok is True
