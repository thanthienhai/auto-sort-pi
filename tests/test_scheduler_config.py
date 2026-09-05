"""Kiểm thử phần mở rộng scheduler config-driven (§9 task.md):
workspace filtering, batching, tie-break, SchedulerConfig.from_dict, Scheduler."""

from __future__ import annotations

from modules.scheduler import Scheduler, SchedulerConfig, optimize_trajectory


def test_scheduler_config_from_dict_parses_yaml_shaped_data():
    data = {
        "start_position_mm": [0.0, 0.0],
        "fragile_group_id": 4,
        "drop_positions_mm": {1: [100.0, 0.0], 2: [200.0, 0.0], 3: [300.0, 0.0], 4: [400.0, 0.0]},
        "max_workspace_mm": [0.0, 500.0, 0.0, 500.0],
        "batch_size": 5,
    }
    config = SchedulerConfig.from_dict(data)
    assert config.start_position_mm == (0.0, 0.0)
    assert config.drop_positions_mm[1] == (100.0, 0.0)
    assert config.max_workspace_mm == (0.0, 500.0, 0.0, 500.0)
    assert config.batch_size == 5


def test_scheduler_wrapper_delegates_to_optimize_trajectory():
    config = SchedulerConfig(start_position_mm=(0.0, 0.0), drop_positions_mm={1: (0.0, 0.0)})
    targets = [
        {"group_id": 1, "x": 100.0, "y": 0.0},
        {"group_id": 1, "x": 10.0, "y": 0.0},
    ]
    order = Scheduler(config).schedule(targets)
    assert [t["x"] for t in order] == [10.0, 100.0]


def test_targets_outside_workspace_are_dropped():
    targets = [
        {"group_id": 1, "x": 100.0, "y": 100.0},
        {"group_id": 1, "x": 9999.0, "y": 9999.0},  # ngoài workspace
    ]
    order = optimize_trajectory(targets, max_workspace_mm=(0.0, 500.0, 0.0, 500.0))
    assert len(order) == 1
    assert order[0]["x"] == 100.0


def test_batch_size_limits_targets_processed_per_call():
    targets = [{"group_id": 1, "x": float(i), "y": 0.0} for i in range(5)]
    order = optimize_trajectory(targets, drop_positions={1: (0.0, 0.0)}, batch_size=2)
    assert len(order) == 2
    assert [t["x"] for t in order] == [0.0, 1.0]


def test_tie_break_is_deterministic_by_original_order():
    # Hai target cách đều vị trí bắt đầu -> phải luôn chọn theo thứ tự xuất hiện gốc.
    targets = [
        {"group_id": 1, "x": 50.0, "y": 0.0},
        {"group_id": 1, "x": -50.0, "y": 0.0},
    ]
    for _ in range(5):  # chạy nhiều lần để chắc chắn không "trôi" do thứ tự dict/set
        order = optimize_trajectory(targets, start_position=(0.0, 0.0), drop_positions={1: (0.0, 0.0)})
        assert order[0]["x"] == 50.0
        assert order[1]["x"] == -50.0


def test_current_position_after_pick_is_drop_position_not_pick_position():
    targets = [
        {"group_id": 1, "x": 10.0, "y": 0.0},
        {"group_id": 1, "x": 90.0, "y": 0.0},
    ]
    # Drop position của nhóm 1 là (1000, 0): rất xa target thứ 2 (90,0) so với
    # nếu current vẫn là pick position (10,0). Nearest-neighbor kế tiếp phải
    # dùng drop position (1000,0) để tính khoảng cách -> vẫn chọn (90,0) vì đó
    # là target còn lại duy nhất, nhưng ta xác nhận qua current position lộ ra
    # gián tiếp bằng cách kiểm tra parametrized case dưới có 3 targets.
    order = optimize_trajectory(targets, start_position=(0.0, 0.0), drop_positions={1: (1000.0, 0.0)})
    assert len(order) == 2

    # Case rõ ràng hơn: sau khi gắp target A (group 1, gần gốc), current phải
    # nhảy tới drop position nhóm 1 (rất xa), khiến target C (group 2, gần drop
    # position đó) được ưu tiên hơn target B (group 1, gần gốc nhưng xa drop).
    targets2 = [
        {"group_id": 1, "x": 10.0, "y": 0.0},   # A: gần gốc (start)
        {"group_id": 1, "x": 20.0, "y": 0.0},   # B: cũng gần gốc, xa drop-position nhóm1
        {"group_id": 2, "x": 500.0, "y": 0.0},  # C: gần drop-position nhóm 1 (500,0)
    ]
    order2 = optimize_trajectory(
        targets2,
        start_position=(0.0, 0.0),
        drop_positions={1: (500.0, 0.0), 2: (0.0, 0.0)},
    )
    groups = [t["group_id"] for t in order2]
    # A được chọn đầu (gần start nhất). Sau đó current = drop nhóm1 = (500,0).
    # Từ (500,0), C (500,0) gần hơn hẳn B (20,0) -> C phải được chọn trước B.
    assert groups == [1, 2, 1]
    assert order2[0]["x"] == 10.0
    assert order2[1]["x"] == 500.0
    assert order2[2]["x"] == 20.0
