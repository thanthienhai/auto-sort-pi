"""Kiểm thử ObjectTracker: spatial exclusion + cooldown + disappearance confirmation (§10)."""

from __future__ import annotations

from modules.dedup import ObjectTracker


def test_no_cooldown_all_targets_pass_through():
    tracker = ObjectTracker()
    targets = [{"group_id": 1, "x": 10.0, "y": 10.0}]
    assert tracker.filter_new_targets(targets, now=0.0) == targets


def test_target_suppressed_right_after_done_while_still_detected():
    tracker = ObjectTracker(suppression_radius_mm=20.0, cooldown_seconds=5.0)
    done_target = {"group_id": 1, "x": 100.0, "y": 100.0}
    tracker.mark_done(done_target, now=0.0)

    # Scan ngay sau đó: vẫn detect vật ở gần đúng vị trí đó -> phải bị suppress.
    same_spot_detection = {"group_id": 1, "x": 105.0, "y": 100.0}
    filtered = tracker.filter_new_targets([same_spot_detection], now=0.5)
    assert filtered == []


def test_different_group_at_same_spot_not_suppressed():
    tracker = ObjectTracker(suppression_radius_mm=20.0, cooldown_seconds=5.0)
    tracker.mark_done({"group_id": 1, "x": 100.0, "y": 100.0}, now=0.0)
    other_group = {"group_id": 2, "x": 100.0, "y": 100.0}
    filtered = tracker.filter_new_targets([other_group], now=0.5)
    assert filtered == [other_group]


def test_far_away_target_not_suppressed():
    tracker = ObjectTracker(suppression_radius_mm=20.0, cooldown_seconds=5.0)
    tracker.mark_done({"group_id": 1, "x": 100.0, "y": 100.0}, now=0.0)
    far_target = {"group_id": 1, "x": 500.0, "y": 500.0}
    filtered = tracker.filter_new_targets([far_target], now=0.5)
    assert filtered == [far_target]


def test_cooldown_expires_after_timeout():
    tracker = ObjectTracker(suppression_radius_mm=20.0, cooldown_seconds=2.0)
    tracker.mark_done({"group_id": 1, "x": 100.0, "y": 100.0}, now=0.0)
    same_spot = {"group_id": 1, "x": 100.0, "y": 100.0}
    assert tracker.filter_new_targets([same_spot], now=1.0) == []
    # Sau khi cooldown hết hạn, object mới ở đúng vị trí đó lại được coi là mới.
    filtered = tracker.filter_new_targets([same_spot], now=3.0)
    assert filtered == [same_spot]
    assert tracker.active_cooldowns() == 0


def test_disappearance_confirmed_releases_cooldown_early():
    tracker = ObjectTracker(
        suppression_radius_mm=20.0, cooldown_seconds=100.0, disappearance_confirm_scans=2
    )
    tracker.mark_done({"group_id": 1, "x": 100.0, "y": 100.0}, now=0.0)

    # Scan 1: không còn detect gì ở đó -> miss #1 (chưa đủ để confirm).
    tracker.filter_new_targets([], now=1.0)
    assert tracker.active_cooldowns() == 1

    # Scan 2: vẫn không detect -> miss #2, đủ confirm_scans -> cooldown giải phóng sớm
    # dù còn rất lâu mới hết cooldown_seconds=100.
    tracker.filter_new_targets([], now=2.0)
    assert tracker.active_cooldowns() == 0


def test_reappearance_resets_miss_counter():
    tracker = ObjectTracker(
        suppression_radius_mm=20.0, cooldown_seconds=100.0, disappearance_confirm_scans=2
    )
    tracker.mark_done({"group_id": 1, "x": 100.0, "y": 100.0}, now=0.0)
    same_spot = {"group_id": 1, "x": 100.0, "y": 100.0}

    tracker.filter_new_targets([], now=1.0)  # miss #1
    tracker.filter_new_targets([same_spot], now=2.0)  # phát hiện lại -> reset miss về 0
    tracker.filter_new_targets([], now=3.0)  # miss #1 lại (không phải #2)
    assert tracker.active_cooldowns() == 1  # chưa đủ 2 miss liên tiếp để confirm
