"""Object deduplication across scan cycles (§10 task.md).

Tránh tình trạng cùng một object bị detect lại (và pick lại) trong nhiều scan
cycle liên tiếp:

    target A pick -> robot DONE -> scan -> object A vẫn còn (camera lag / robot
    tay gắp chưa rời khỏi khung hình / nhiễu) -> KHÔNG được lập tức pick lại.

Thiết kế nhẹ, phù hợp Raspberry Pi: kết hợp **spatial exclusion + cooldown**
(chặn theo bán kính quanh vị trí vừa DONE trong một khoảng thời gian) với
**disappearance confirmation** (nếu object biến mất khỏi vị trí đó trong vài
scan liên tiếp, huỷ cooldown sớm — không cần chờ hết thời gian — vì đã xác
nhận robot đã lấy nó đi).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import hypot

from .logging_setup import get_logger

logger = get_logger("autosort.dedup")


@dataclass
class _CooldownEntry:
    x: float
    y: float
    group_id: int
    expires_at: float
    consecutive_misses: int = 0


@dataclass
class ObjectTracker:
    suppression_radius_mm: float = 30.0
    cooldown_seconds: float = 5.0
    disappearance_confirm_scans: int = 2
    _entries: list[_CooldownEntry] = field(default_factory=list)

    def mark_done(self, target: dict, now: float | None = None) -> None:
        """Gọi ngay sau khi MCU báo ``!DONE`` cho một target — bắt đầu cooldown."""
        now = now if now is not None else time.monotonic()
        entry = _CooldownEntry(
            x=float(target["x"]),
            y=float(target["y"]),
            group_id=int(target["group_id"]),
            expires_at=now + self.cooldown_seconds,
        )
        self._entries.append(entry)
        logger.info(
            "step.dedup.cooldown_started",
            group_id=entry.group_id,
            x=entry.x,
            y=entry.y,
            cooldown_seconds=self.cooldown_seconds,
        )

    def _matches(self, entry: _CooldownEntry, target: dict) -> bool:
        if int(target["group_id"]) != entry.group_id:
            return False
        return hypot(float(target["x"]) - entry.x, float(target["y"]) - entry.y) <= self.suppression_radius_mm

    def filter_new_targets(self, targets: list[dict], now: float | None = None) -> list[dict]:
        """Cập nhật trạng thái cooldown theo scan hiện tại và trả về target thực sự mới.

        Mỗi cooldown entry được so khớp với các target scan được trong lượt này:
        - còn khớp (object vẫn ở đó)  -> reset miss counter, tiếp tục suppress.
        - không khớp (object biến mất) -> tăng miss counter; đủ
          ``disappearance_confirm_scans`` lần liên tiếp thì coi là đã được xử lý
          xong, xoá cooldown sớm (không cần chờ hết ``cooldown_seconds``).
        - cooldown hết hạn theo thời gian -> xoá luôn, bất kể matched hay chưa.
        """
        now = now if now is not None else time.monotonic()

        remaining_entries: list[_CooldownEntry] = []
        suppressed_target_ids: set[int] = set()

        for entry in self._entries:
            if now >= entry.expires_at:
                logger.info(
                    "step.dedup.cooldown_expired", group_id=entry.group_id, x=entry.x, y=entry.y
                )
                continue

            matched_any = False
            for i, target in enumerate(targets):
                if self._matches(entry, target):
                    matched_any = True
                    suppressed_target_ids.add(id(target))
            if matched_any:
                entry.consecutive_misses = 0
                remaining_entries.append(entry)
            else:
                entry.consecutive_misses += 1
                if entry.consecutive_misses >= self.disappearance_confirm_scans:
                    logger.info(
                        "step.dedup.disappearance_confirmed",
                        group_id=entry.group_id,
                        x=entry.x,
                        y=entry.y,
                        scans=entry.consecutive_misses,
                    )
                else:
                    remaining_entries.append(entry)

        self._entries = remaining_entries
        return [t for t in targets if id(t) not in suppressed_target_ids]

    def active_cooldowns(self) -> int:
        return len(self._entries)
