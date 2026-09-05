"""Target validation layer (§11 task.md).

Chạy ngay sau Pixel→World và trước Object Deduplication / Scheduler trong
pipeline (§2). Không target nào được vào scheduler hay UART nếu chưa qua đây.

Kiểm tra: ``group_id`` hợp lệ, ``confidence`` hợp lệ, ``x``/``y`` finite (không
NaN/Inf), trong workspace, target chưa được xử lý (``target_id`` chưa nằm
trong tập đã pick), target không stale (quá cũ so với thời điểm capture).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .exceptions import TargetValidationError
from .logging_setup import get_logger

logger = get_logger("autosort.target_validator")


@dataclass
class WorkspaceBounds:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min_mm <= x <= self.x_max_mm and self.y_min_mm <= y <= self.y_max_mm


@dataclass
class TargetValidator:
    workspace: WorkspaceBounds
    valid_group_ids: frozenset[int]
    min_confidence: float = 0.0
    max_target_age_seconds: float = 5.0
    _processed_target_ids: set[str] = field(default_factory=set)

    def mark_processed(self, target_id: str) -> None:
        self._processed_target_ids.add(target_id)

    def is_processed(self, target_id: str) -> bool:
        return target_id in self._processed_target_ids

    def validate(self, target: dict, now: float | None = None) -> None:
        """Raise ``TargetValidationError`` với lý do cụ thể nếu target không hợp lệ."""
        now = now if now is not None else time.monotonic()

        group_id = target.get("group_id")
        if group_id not in self.valid_group_ids:
            raise TargetValidationError(f"group_id không hợp lệ: {group_id!r}")

        conf = target.get("conf")
        if conf is None or not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
            raise TargetValidationError(f"confidence không hợp lệ: {conf!r}")
        if conf < self.min_confidence:
            raise TargetValidationError(
                f"confidence {conf} dưới ngưỡng tối thiểu {self.min_confidence}"
            )

        x, y = target.get("x"), target.get("y")
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            raise TargetValidationError(f"toạ độ không hợp lệ (NaN/Inf/None): x={x!r} y={y!r}")

        if not self.workspace.contains(x, y):
            raise TargetValidationError(
                f"toạ độ ({x},{y}) nằm ngoài workspace "
                f"[{self.workspace.x_min_mm},{self.workspace.x_max_mm}] x "
                f"[{self.workspace.y_min_mm},{self.workspace.y_max_mm}]"
            )

        target_id = target.get("target_id")
        if target_id is not None and self.is_processed(target_id):
            raise TargetValidationError(f"target {target_id} đã được xử lý trước đó")

        captured_at = target.get("captured_at")
        if captured_at is not None:
            age = now - captured_at
            if age > self.max_target_age_seconds:
                raise TargetValidationError(
                    f"target stale: age={age:.2f}s > max={self.max_target_age_seconds}s"
                )

    def filter_valid(self, targets: list[dict], now: float | None = None) -> list[dict]:
        """Trả về danh sách target hợp lệ; log + bỏ qua từng target lỗi (không sập cycle)."""
        valid: list[dict] = []
        for target in targets:
            try:
                self.validate(target, now=now)
                valid.append(target)
            except TargetValidationError as exc:
                logger.warning("step.target_validation.rejected", target=target, reason=str(exc))
        return valid
