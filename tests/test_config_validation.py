"""Kiểm thử validate configuration khi startup (§20 task.md)."""

from __future__ import annotations

import copy

import pytest
import yaml
from pathlib import Path

from modules.config_validation import assert_valid_settings, validate_settings
from modules.exceptions import ConfigError

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_real_settings() -> dict:
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


def test_shipped_settings_yaml_is_valid():
    settings = _load_real_settings()
    assert validate_settings(settings) == []
    assert_valid_settings(settings)  # không raise


def test_missing_section_reported():
    settings = _load_real_settings()
    del settings["uart"]
    errors = validate_settings(settings)
    assert any("uart" in e for e in errors)


def test_confidence_threshold_out_of_range_reported():
    settings = _load_real_settings()
    settings["inference"]["confidence_threshold"] = 1.5
    errors = validate_settings(settings)
    assert any("confidence_threshold" in e for e in errors)


def test_workspace_min_greater_than_max_reported():
    settings = _load_real_settings()
    settings["workspace"]["x_min_mm"] = 1000.0
    errors = validate_settings(settings)
    assert any("x_min_mm phải nhỏ hơn" in e for e in errors)


def test_empty_drop_positions_reported():
    settings = _load_real_settings()
    settings["scheduler"]["drop_positions_mm"] = {}
    errors = validate_settings(settings)
    assert any("drop_positions_mm" in e for e in errors)


def test_assert_valid_settings_raises_config_error_with_all_reasons():
    settings = _load_real_settings()
    settings["uart"]["port"] = ""
    settings["inference"]["confidence_threshold"] = 5.0
    with pytest.raises(ConfigError) as exc_info:
        assert_valid_settings(settings)
    message = str(exc_info.value)
    assert "uart.port" in message
    assert "confidence_threshold" in message


def test_deep_copy_of_valid_settings_stays_valid():
    settings = copy.deepcopy(_load_real_settings())
    assert validate_settings(settings) == []
