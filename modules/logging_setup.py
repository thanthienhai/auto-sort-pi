"""Cấu hình structlog: mỗi chu kỳ xử lý ghi input/output từng step ra một file riêng.

Trong pipeline auto-sort-pi, mỗi vòng "capture → detect → transform → schedule →
uart" được coi là một *request* và mang một ``request_id`` (cycle id) có tiền tố
timestamp. Log được ghi song song:

- File ``logs/requests/<request_id>.log`` (JSON lines) — mỗi chu kỳ một file để dễ
  truy vết đúng một lần gắp và sort theo thời gian.
- Console để xem nhanh khi vận hành.

``request_id`` được bind bằng contextvars ở đầu mỗi chu kỳ nên mọi log call trong
toàn bộ pipeline (detector, transformer, scheduler, serial_comm) đều tự đính kèm mà
không phải truyền tay.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import structlog

PROJECT_DIR = Path(__file__).resolve().parent.parent
REQUEST_LOG_DIR = PROJECT_DIR / "logs" / "requests"

_CONFIGURED = False


class PerRequestFileHandler(logging.Handler):
    """Ghi mỗi record vào file riêng theo request_id: ``<log_dir>/<request_id>.log``.

    Record không có request_id (nằm ngoài context của một chu kỳ) được bỏ qua ở đây
    — chúng vẫn hiện ở console. Mỗi lần ghi mở file ở chế độ append rồi đóng ngay nên
    an toàn và luôn được flush; khối lượng log mỗi chu kỳ nhỏ nên chi phí không đáng
    kể.
    """

    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self.log_dir = log_dir

    def emit(self, record: logging.LogRecord) -> None:
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        if not request_id:
            return
        try:
            message = self.format(record)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(request_id))
            log_path = self.log_dir / f"{safe_id}.log"
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        except Exception:
            self.handleError(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Khởi tạo structlog + stdlib logging. An toàn khi gọi nhiều lần."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    REQUEST_LOG_DIR.mkdir(parents=True, exist_ok=True)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    file_handler = PerRequestFileHandler(REQUEST_LOG_DIR)
    file_handler.setFormatter(file_formatter)

    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    _CONFIGURED = True


def prune_old_request_logs(retention_days: float, log_dir: Path = REQUEST_LOG_DIR) -> int:
    """Xoá các file ``logs/requests/*.log`` cũ hơn ``retention_days`` (§19 task.md).

    Mỗi chu kỳ ghi một file riêng (không phải một file lớn append liên tục),
    nên "rotation" ở đây là dọn theo tuổi file thay vì theo kích thước. Trả về
    số file đã xoá; an toàn khi gọi nhiều lần / thư mục chưa tồn tại.
    """
    if not log_dir.is_dir() or retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for path in log_dir.glob("*.log"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def get_logger(name: str = "autosort") -> Any:
    """Trả về structlog logger đã cấu hình."""
    return structlog.get_logger(name)


def preview(text: str | None, limit: int = 1000) -> dict[str, Any]:
    """Tóm tắt một chuỗi dài để log không phình file."""
    if text is None:
        return {"length": 0, "preview": None}
    text = str(text)
    if len(text) <= limit:
        return {"length": len(text), "preview": text}
    return {"length": len(text), "preview": text[:limit], "truncated": True}
