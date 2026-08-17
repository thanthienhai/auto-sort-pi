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
    """Return a deterministic pick order following the SRS scheduling rules."""
    remaining = [dict(target) for target in targets]
    drops = drop_positions or DEFAULT_DROP_POSITIONS_MM
    ordered: list[dict[str, Any]] = []
    current = start_position

    def pick_nearest(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return min(
            candidates,
            key=lambda target: hypot(
                float(target["x"]) - current[0],
                float(target["y"]) - current[1],
            ),
        )

    while remaining:
        fragile = [
            item for item in remaining
            if int(item["group_id"]) == fragile_group_id
        ]
        candidates = fragile if fragile else remaining
        item = pick_nearest(candidates)
        remaining.remove(item)
        ordered.append(item)

        # The SRS defines the robot's next position as the destination bin.
        current = drops.get(int(item["group_id"]), current)

    return ordered
