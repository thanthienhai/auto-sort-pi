"""Kiểm thử prune_old_request_logs (§19 task.md — log rotation theo tuổi file)."""

from __future__ import annotations

import os
import time

from modules.logging_setup import prune_old_request_logs


def test_prune_removes_only_old_files(tmp_path):
    old_file = tmp_path / "old.log"
    new_file = tmp_path / "new.log"
    old_file.write_text("x", encoding="utf-8")
    new_file.write_text("x", encoding="utf-8")

    old_time = time.time() - 30 * 86400
    os.utime(old_file, (old_time, old_time))

    removed = prune_old_request_logs(retention_days=14, log_dir=tmp_path)
    assert removed == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_prune_on_missing_directory_is_noop(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert prune_old_request_logs(retention_days=14, log_dir=missing) == 0


def test_prune_with_nonpositive_retention_is_noop(tmp_path):
    f = tmp_path / "a.log"
    f.write_text("x", encoding="utf-8")
    assert prune_old_request_logs(retention_days=0, log_dir=tmp_path) == 0
    assert f.exists()
