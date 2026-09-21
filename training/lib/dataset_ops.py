"""Dataset-level operations: manifest I/O, group-aware splitting, loading a
normalized dataset directory. Dùng chung bởi các subcommand trong training/cli.py.

- Manifest schema (§5) và review queue record (§11).
- Group/session-aware train/val/test split (§16, §17): không random split đơn
  thuần — mọi ảnh cùng ``group_key`` (source video / scene / sequence / batch)
  PHẢI rơi vào đúng 1 split. Deterministic theo seed — cùng input + cùng seed
  => cùng kết quả (§25), không phụ thuộc thời gian chạy.
- Loading một dataset đã normalize (images/ + labels/ + metadata.jsonl).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MANIFEST_FIELDS = (
    "name",
    "source",
    "url",
    "source_type",
    "license",
    "author",
    "version",
    "download_date",
    "task",
    "image_count",
    "annotation_format",
    "original_classes",
    "target_mapping",
    "notes",
    "quality_score",
    "source_weight",
    "domain",
    "status",
)


@dataclass
class DatasetManifest:
    name: str
    source: str = ""
    url: str = ""
    source_type: str = ""
    license: str = ""
    author: str = ""
    version: str = ""
    download_date: str | None = None
    task: str = "object_detection"
    image_count: int | None = None
    annotation_format: str = ""
    original_classes: list[str] = field(default_factory=list)
    target_mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""
    quality_score: float | None = None
    source_weight: str = "medium"
    domain: str = "unknown"
    status: str = "registered"

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {k: v for k, v in data.items() if k in MANIFEST_FIELDS}
        return cls(**known)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f: getattr(self, f) for f in MANIFEST_FIELDS}
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


@dataclass
class ReviewRecord:
    image: str
    source_dataset: str
    source_class: str
    candidate_targets: list[str]
    reason: str
    status: str = "pending"
    bbox: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "source_dataset": self.source_dataset,
            "source_class": self.source_class,
            "candidate_targets": self.candidate_targets,
            "reason": self.reason,
            "status": self.status,
            "bbox": self.bbox,
        }


def write_review_manifest(records: list[ReviewRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def read_review_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


def _bucket(group_key: str, seed: int) -> float:
    """Hash deterministic group_key+seed -> float in [0, 1)."""

    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_group_splits(
    group_keys: list[str],
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, str]:
    """Gán mỗi group_key duy nhất vào 'train'/'val'/'test'. Deterministic."""

    assignment: dict[str, str] = {}
    for key in sorted(set(group_keys)):
        bucket = _bucket(key, seed)
        if bucket < ratios.train:
            assignment[key] = "train"
        elif bucket < ratios.train + ratios.val:
            assignment[key] = "val"
        else:
            assignment[key] = "test"
    return assignment


def split_items(
    items: list[tuple[str, str]],
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, list[str]]:
    """items = [(item_id, group_key), ...]. Trả về {'train': [...], 'val': [...], 'test': [...]}."""

    group_keys = [g for _, g in items]
    assignment = assign_group_splits(group_keys, ratios, seed)

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for item_id, group_key in items:
        result[assignment[group_key]].append(item_id)
    return result


def verify_no_leakage(split_result: dict[str, list[str]], item_to_group: dict[str, str]) -> list[str]:
    """Trả về list lỗi nếu có group xuất hiện ở >1 split."""

    group_to_splits: dict[str, set[str]] = {}
    for split_name, item_ids in split_result.items():
        for item_id in item_ids:
            group = item_to_group[item_id]
            group_to_splits.setdefault(group, set()).add(split_name)

    errors = []
    for group, splits in group_to_splits.items():
        if len(splits) > 1:
            errors.append(f"group '{group}' leaks across splits: {sorted(splits)}")
    return errors


@dataclass
class NormalizedItem:
    image_id: str
    image_path: Path
    label_path: Path
    metadata: dict


def load_normalized_dataset(dataset_dir: str | Path) -> list[NormalizedItem]:
    """Đọc 1 dataset đã normalize (từ import_dataset): images/ + labels/ + metadata.jsonl."""

    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    metadata_path = dataset_dir / "metadata.jsonl"

    metadata_by_id: dict[str, dict] = {}
    if metadata_path.is_file():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                metadata_by_id[rec["image_id"]] = rec

    items = []
    if not images_dir.is_dir():
        return items
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image_id = image_path.stem
        label_path = labels_dir / f"{image_id}.txt"
        items.append(
            NormalizedItem(
                image_id=image_id,
                image_path=image_path,
                label_path=label_path,
                metadata=metadata_by_id.get(image_id, {"image_id": image_id}),
            )
        )
    return items


def group_key_for(item: NormalizedItem) -> str:
    """Group key dùng cho split (§16): nếu metadata có sequence/scene/batch (vd
    frame trích từ cùng 1 video camera thực tế) thì group theo đó để tránh
    leakage. Ảnh rời rạc từ public dataset (không có metadata sequence) mặc
    định mỗi ảnh là 1 group riêng — an toàn vì không có quan hệ frame-liên-tiếp
    giữa chúng (duplicate/near-duplicate đã được xử lý riêng ở bước dedup)."""

    meta = item.metadata
    for key in ("sequence_id", "scene_id", "group_key", "capture_session"):
        if meta.get(key):
            return f"{meta.get('source_dataset', '')}:{meta[key]}"
    return item.image_id
