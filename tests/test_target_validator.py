"""Kiểm thử TargetValidator (§11 task.md)."""

from __future__ import annotations

import math

import pytest

from modules.exceptions import TargetValidationError
from modules.target_validator import TargetValidator, WorkspaceBounds


def _validator(**overrides):
    defaults = dict(
        workspace=WorkspaceBounds(x_min_mm=0.0, x_max_mm=500.0, y_min_mm=0.0, y_max_mm=500.0),
        valid_group_ids=frozenset({1, 2, 3, 4}),
        min_confidence=0.3,
        max_target_age_seconds=5.0,
    )
    defaults.update(overrides)
    return TargetValidator(**defaults)


def _target(**overrides):
    defaults = dict(group_id=1, conf=0.9, x=100.0, y=100.0)
    defaults.update(overrides)
    return defaults


def test_valid_target_passes():
    validator = _validator()
    validator.validate(_target())  # không raise


def test_invalid_group_id_rejected():
    validator = _validator()
    with pytest.raises(TargetValidationError, match="group_id"):
        validator.validate(_target(group_id=99))


def test_confidence_out_of_range_rejected():
    validator = _validator()
    with pytest.raises(TargetValidationError, match="confidence"):
        validator.validate(_target(conf=1.5))


def test_confidence_below_minimum_rejected():
    validator = _validator(min_confidence=0.5)
    with pytest.raises(TargetValidationError, match="ngưỡng tối thiểu"):
        validator.validate(_target(conf=0.4))


def test_nan_coordinates_rejected():
    validator = _validator()
    with pytest.raises(TargetValidationError, match="toạ độ"):
        validator.validate(_target(x=math.nan, y=100.0))


def test_infinite_coordinates_rejected():
    validator = _validator()
    with pytest.raises(TargetValidationError, match="toạ độ"):
        validator.validate(_target(x=math.inf, y=100.0))


def test_out_of_workspace_rejected():
    validator = _validator()
    with pytest.raises(TargetValidationError, match="workspace"):
        validator.validate(_target(x=9999.0, y=100.0))


def test_already_processed_target_rejected():
    validator = _validator()
    validator.mark_processed("t1")
    with pytest.raises(TargetValidationError, match="đã được xử lý"):
        validator.validate(_target(target_id="t1"))


def test_stale_target_rejected():
    validator = _validator(max_target_age_seconds=1.0)
    with pytest.raises(TargetValidationError, match="stale"):
        validator.validate(_target(captured_at=0.0), now=10.0)


def test_fresh_target_within_age_passes():
    validator = _validator(max_target_age_seconds=5.0)
    validator.validate(_target(captured_at=8.0), now=10.0)  # không raise


def test_filter_valid_drops_bad_targets_without_raising():
    validator = _validator()
    targets = [_target(), _target(group_id=99), _target(x=math.nan)]
    valid = validator.filter_valid(targets)
    assert len(valid) == 1
