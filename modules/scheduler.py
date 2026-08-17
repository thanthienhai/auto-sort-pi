"""Target scheduling using fragile-item priority and nearest-neighbor selection."""

from __future__ import annotations

from math import hypot
from typing import Any, Iterable

DEFAULT_DROP_POSITIONS_MM: dict[int, tuple[float, float]] = {
    1: (100.0, 0.0),
    2: (200.0, 0.0),
    3: (300.0, 0.0),
    4: (400.0, 0.0),
}


def optimize_trajectory(
    targets: Iterable[dict[str, Any]],
    start_position: tuple[float, float] = (0.0, 0.0),
    drop_positions: dict[int, tuple[float, float]] | None = None,
    fragile_group_id: int = 4,
) -> list[dict[str, Any]]:
    """Return a deterministic pick order.

    Group 4 is selected first when present. After each pick, the virtual robot
    position becomes the drop position for that target's group, matching the SRS.
    """
    remaining = [dict(target) for target in targets]
    drops = drop_positions or DEFAULT_DROP_POSITIONS_MM
    ordered: list[dict[str, Any]] = []
    current = start_position

    if any(int(item["group_id"]) == fragile_group_id for item in remaining):
        fragile = [item for item in remaining if int(item["group_id"]) == fragile_group_id]
        remaining = [item for item in remaining if int(item["group_id"]) != fragile_group_id]
        for item in _nearest_neighbor(fragile, current):
            ordered.append(item)
            current = drops.get(int(item["group_id"]), current)

    while remaining:
        item = min(
            remaining,
            key=lambda target: hypot(
                float(target["x"]) - current[0],
                float(target["y"]) - current[1],
            ),
        )
        remaining.remove(item)
        ordered.append(item)
        current = drops.get(int(item["group_id"]), current)

    return ordered


def _nearest_neighbor(
    targets: list[dict[str, Any]],
    start: tuple[float, float],
) -> list[dict[str, Any]]:
    current = start
    remaining = targets[:]
    result: list[dict[str, Any]] = []
    while remaining:
        item = min(
            remaining,
            key=lambda target: hypot(
                float(target["x"]) - current[0],
                float(target["y"]) - current[1],
            ),
        )
        remaining.remove(item)
        result.append(item)
        current = (float(item["x"]), float(item["y"]))
    return result
