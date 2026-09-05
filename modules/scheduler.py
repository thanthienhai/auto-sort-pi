"""Target scheduling using fragile-item priority and nearest-neighbor selection.

§9 task.md: toàn bộ tham số (start position, fragile group, drop positions,
workspace, batch size) phải đến từ ``config/settings.yaml`` — không hard-code
trong Python. ``DEFAULT_DROP_POSITIONS_MM`` dưới đây chỉ là fallback cho
test/dev khi không truyền config.

**Semantics vị trí hiện tại của robot** (để không mơ hồ, theo yêu cầu task.md):

- Trước khi gắp target đầu tiên trong một chu kỳ, vị trí hiện tại của đầu gắp
  là ``start_position_mm`` — đây được coi là một **pick position** kế thừa từ
  cuối chu kỳ trước (nơi robot đang đứng chờ).
- Ngay sau khi chọn một target để gắp, ``optimize_trajectory`` cập nhật vị trí
  hiện tại thành **drop position** của nhóm rác đó (``drop_positions_mm[group_id]``)
  — vì SRS giả định robot luôn thả xong trước khi tính khoảng cách cho target
  kế tiếp. Vị trí dùng để tính khoảng cách Euclid cho lần chọn tiếp theo luôn
  là drop position vừa cập nhật, KHÔNG phải pick position của target vừa chọn.

Tie-breaking: khi hai target có khoảng cách bằng nhau, thứ tự phải xác định
(deterministic) — chọn theo thứ tự xuất hiện gốc trong ``targets`` (ổn định,
không phụ thuộc thứ tự duyệt dict/set của Python).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable

DEFAULT_DROP_POSITIONS_MM: dict[int, tuple[float, float]] = {
    1: (100.0, 0.0),
    2: (200.0, 0.0),
    3: (300.0, 0.0),
    4: (400.0, 0.0),
}


@dataclass
class SchedulerConfig:
    """Toàn bộ tham số scheduler, nạp từ ``config/settings.yaml['scheduler']``."""

    start_position_mm: tuple[float, float] = (0.0, 0.0)
    fragile_group_id: int = 4
    drop_positions_mm: dict[int, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_DROP_POSITIONS_MM)
    )
    max_workspace_mm: tuple[float, float, float, float] | None = None  # (xmin,xmax,ymin,ymax)
    batch_size: int | None = None  # None = xử lý toàn bộ target trong một batch

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchedulerConfig":
        drop_raw = data.get("drop_positions_mm", DEFAULT_DROP_POSITIONS_MM)
        drop_positions = {int(k): tuple(v) for k, v in drop_raw.items()}
        workspace_raw = data.get("max_workspace_mm")
        max_workspace = tuple(workspace_raw) if workspace_raw else None
        return cls(
            start_position_mm=tuple(data.get("start_position_mm", (0.0, 0.0))),
            fragile_group_id=int(data.get("fragile_group_id", 4)),
            drop_positions_mm=drop_positions,
            max_workspace_mm=max_workspace,
            batch_size=data.get("batch_size"),
        )


def _within_workspace(x: float, y: float, bounds: tuple[float, float, float, float] | None) -> bool:
    if bounds is None:
        return True
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def optimize_trajectory(
    targets: Iterable[dict[str, Any]],
    start_position: tuple[float, float] = (0.0, 0.0),
    drop_positions: dict[int, tuple[float, float]] | None = None,
    fragile_group_id: int = 4,
    max_workspace_mm: tuple[float, float, float, float] | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic pick order following the SRS scheduling rules.

    Target ngoài ``max_workspace_mm`` bị loại khỏi lịch (đây chỉ là một lớp
    phòng thủ bổ sung cho scheduler; validation chính thức nằm ở
    ``modules.target_validator``). ``batch_size`` giới hạn số target xử lý
    trong một lần gọi (phần còn lại bị bỏ, để chu kỳ scan kế tiếp xử lý tiếp —
    tránh một chu kỳ quá tải khi có rất nhiều vật thể cùng lúc).
    """
    indexed = list(enumerate(targets))
    remaining = [
        (i, dict(target))
        for i, target in indexed
        if _within_workspace(float(target["x"]), float(target["y"]), max_workspace_mm)
    ]
    drops = drop_positions or DEFAULT_DROP_POSITIONS_MM
    ordered: list[dict[str, Any]] = []
    current = start_position

    def pick_nearest(candidates: list[tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
        # Tie-break xác định: khoảng cách trước, rồi tới thứ tự xuất hiện gốc
        # (candidates[*][0]) để không phụ thuộc thứ tự duyệt không ổn định.
        return min(
            candidates,
            key=lambda item: (
                hypot(float(item[1]["x"]) - current[0], float(item[1]["y"]) - current[1]),
                item[0],
            ),
        )

    while remaining:
        if batch_size is not None and len(ordered) >= batch_size:
            break
        fragile = [
            entry for entry in remaining
            if int(entry[1]["group_id"]) == fragile_group_id
        ]
        candidates = fragile if fragile else remaining
        original_index, item = pick_nearest(candidates)
        item["scheduler_distance"] = round(
            hypot(float(item["x"]) - current[0], float(item["y"]) - current[1]), 2
        )
        remaining.remove((original_index, item))
        ordered.append(item)

        # Vị trí hiện tại sau khi chọn target này là DROP position (§9 docstring
        # trên) — robot được giả định đã thả xong vật trước khi tính khoảng cách
        # cho lần chọn kế tiếp.
        current = drops.get(int(item["group_id"]), current)

    return ordered


class Scheduler:
    """Wrapper OOP mỏng quanh ``optimize_trajectory`` dùng ``SchedulerConfig``."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def schedule(self, targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return optimize_trajectory(
            targets,
            start_position=self.config.start_position_mm,
            drop_positions=self.config.drop_positions_mm,
            fragile_group_id=self.config.fragile_group_id,
            max_workspace_mm=self.config.max_workspace_mm,
            batch_size=self.config.batch_size,
        )
