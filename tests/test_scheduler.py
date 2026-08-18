"""Kiểm thử scheduler: ưu tiên nhóm dễ vỡ + nearest-neighbor + cập nhật vị trí thùng."""

from modules.scheduler import optimize_trajectory


def test_empty_input_returns_empty():
    assert optimize_trajectory([]) == []


def test_fragile_group_is_served_first():
    targets = [
        {"group_id": 1, "x": 10.0, "y": 10.0},
        {"group_id": 3, "x": 20.0, "y": 20.0},
        {"group_id": 4, "x": 300.0, "y": 300.0},
        {"group_id": 4, "x": 5.0, "y": 5.0},
    ]
    order = optimize_trajectory(targets, start_position=(0.0, 0.0), fragile_group_id=4)
    groups = [t["group_id"] for t in order]
    # Hai nhóm 4 phải được xử lý trước bất kỳ nhóm nào khác.
    assert groups[:2] == [4, 4]
    # Trong nhóm 4, vật gần vị trí bắt đầu (0,0) hơn được chọn trước.
    assert (order[0]["x"], order[0]["y"]) == (5.0, 5.0)


def test_nearest_neighbor_within_same_priority():
    targets = [
        {"group_id": 1, "x": 100.0, "y": 0.0},
        {"group_id": 1, "x": 10.0, "y": 0.0},
        {"group_id": 1, "x": 50.0, "y": 0.0},
    ]
    order = optimize_trajectory(
        targets,
        start_position=(0.0, 0.0),
        drop_positions={1: (0.0, 0.0)},  # thả về gốc để nearest-neighbor thuần tuý
    )
    xs = [t["x"] for t in order]
    assert xs == [10.0, 50.0, 100.0]


def test_all_targets_are_scheduled_exactly_once():
    targets = [{"group_id": (i % 4) + 1, "x": float(i), "y": 0.0} for i in range(12)]
    order = optimize_trajectory(targets)
    assert len(order) == len(targets)
    assert sorted(t["x"] for t in order) == sorted(t["x"] for t in targets)
