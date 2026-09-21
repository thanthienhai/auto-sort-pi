"""Dataset statistics + coverage report (§12, §13, §14)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from training.lib.formats import parse_yolo_line, validate_bbox
from training.lib.taxonomy import Taxonomy, coverage_status


@dataclass
class DatasetStats:
    total_images: int = 0
    total_instances: int = 0
    instances_per_class: Counter = field(default_factory=Counter)
    images_per_class: Counter = field(default_factory=Counter)
    bbox_area_samples: list[float] = field(default_factory=list)
    resolution_counts: Counter = field(default_factory=Counter)
    source_dataset_counts: Counter = field(default_factory=Counter)
    license_counts: Counter = field(default_factory=Counter)
    domain_counts: Counter = field(default_factory=Counter)
    unmapped_classes: set = field(default_factory=set)
    review_required_classes: set = field(default_factory=set)
    ignored_classes: set = field(default_factory=set)
    duplicate_count: int = 0
    corrupted_image_count: int = 0
    negative_image_count: int = 0
    annotation_error_count: int = 0


def _load_metadata(metadata_path: Path | None) -> dict[str, dict]:
    if metadata_path is None or not metadata_path.is_file():
        return {}
    out = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["image_id"]] = rec
    return out


def compute_dataset_stats(
    images_dir: str | Path,
    labels_dir: str | Path,
    taxonomy: Taxonomy,
    metadata_path: str | Path | None = None,
    duplicate_groups_path: str | Path | None = None,
) -> DatasetStats:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    metadata = _load_metadata(Path(metadata_path) if metadata_path else None)

    stats = DatasetStats()
    image_paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    stats.total_images = len(image_paths)

    for image_path in image_paths:
        image_id = image_path.stem
        meta = metadata.get(image_id, {})
        if meta.get("source_dataset"):
            stats.source_dataset_counts[meta["source_dataset"]] += 1
        if meta.get("license"):
            stats.license_counts[meta["license"]] += 1
        if meta.get("domain"):
            stats.domain_counts[meta["domain"]] += 1
        if meta.get("width") and meta.get("height"):
            stats.resolution_counts[f"{meta['width']}x{meta['height']}"] += 1

        label_path = Path(str(image_path).replace(str(images_dir), str(labels_dir), 1)).with_suffix(".txt")
        if not label_path.is_file() or label_path.stat().st_size == 0:
            stats.negative_image_count += 1
            continue

        classes_in_image = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            box, errors = parse_yolo_line(line)
            if errors or box is None:
                stats.annotation_error_count += 1
                continue
            if not taxonomy.is_valid_class_id(box.class_id):
                stats.annotation_error_count += 1
                continue
            box_errors = validate_bbox(box, taxonomy.num_classes)
            if box_errors:
                stats.annotation_error_count += 1
                continue
            class_name = taxonomy.name_for(box.class_id)
            stats.total_instances += 1
            stats.instances_per_class[class_name] += 1
            classes_in_image.add(class_name)
            stats.bbox_area_samples.append(box.w * box.h)

        for class_name in classes_in_image:
            stats.images_per_class[class_name] += 1

    if duplicate_groups_path is not None:
        dup_path = Path(duplicate_groups_path)
        if dup_path.is_file():
            groups = json.loads(dup_path.read_text(encoding="utf-8"))
            stats.duplicate_count = sum(len(g["duplicates"]) for g in groups)

    return stats


def class_coverage_table(stats: DatasetStats, taxonomy: Taxonomy) -> list[dict]:
    total_instances = stats.total_instances or 1
    rows = []
    for class_name in taxonomy.classes:
        instances = stats.instances_per_class.get(class_name, 0)
        rows.append(
            {
                "class_id": taxonomy.id_for(class_name),
                "class": class_name,
                "group": taxonomy.group_name_for(class_name),
                "images": stats.images_per_class.get(class_name, 0),
                "instances": instances,
                "percentage": round(100.0 * instances / total_instances, 2),
                "status": coverage_status(instances, taxonomy.coverage_thresholds),
                "hard_class_priority": class_name in taxonomy.hard_classes,
            }
        )
    return rows


def bbox_area_distribution(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0}
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    return {
        "count": n,
        "min": sorted_samples[0],
        "max": sorted_samples[-1],
        "mean": sum(sorted_samples) / n,
        "median": sorted_samples[n // 2],
        "p10": sorted_samples[int(n * 0.1)],
        "p90": sorted_samples[min(n - 1, int(n * 0.9))],
    }


def build_report_dict(stats: DatasetStats, taxonomy: Taxonomy) -> dict:
    positive_images = stats.total_images - stats.negative_image_count
    return {
        "total_images": stats.total_images,
        "positive_images": positive_images,
        "negative_images": stats.negative_image_count,
        "negative_ratio": round(stats.negative_image_count / stats.total_images, 4) if stats.total_images else 0.0,
        "total_instances": stats.total_instances,
        "annotation_error_count": stats.annotation_error_count,
        "duplicate_count": stats.duplicate_count,
        "bbox_area_distribution": bbox_area_distribution(stats.bbox_area_samples),
        "resolution_distribution": dict(stats.resolution_counts),
        "source_dataset_distribution": dict(stats.source_dataset_counts),
        "license_distribution": dict(stats.license_counts),
        "domain_distribution": dict(stats.domain_counts),
        "class_coverage": class_coverage_table(stats, taxonomy),
        "hard_class_priority": list(taxonomy.hard_classes),
        "unmapped_classes": sorted(stats.unmapped_classes),
        "review_required_classes": sorted(stats.review_required_classes),
        "ignored_classes": sorted(stats.ignored_classes),
    }


def write_report_json(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report_csv(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class", "group", "images", "instances", "percentage", "status", "hard_class_priority"],
        )
        writer.writeheader()
        for row in report["class_coverage"]:
            writer.writerow(row)


def write_report_md(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Report",
        "",
        f"- Total images: {report['total_images']} (positive: {report['positive_images']}, negative: {report['negative_images']}, ratio: {report['negative_ratio']})",
        f"- Total instances: {report['total_instances']}",
        f"- Annotation errors: {report['annotation_error_count']}",
        f"- Duplicate images: {report['duplicate_count']}",
        "",
        "## Class coverage",
        "",
        "| id | class | group | images | instances | % | status | hard |",
        "|----|-------|-------|--------|-----------|---|--------|------|",
    ]
    for row in report["class_coverage"]:
        lines.append(
            f"| {row['class_id']} | {row['class']} | {row['group']} | {row['images']} | "
            f"{row['instances']} | {row['percentage']} | {row['status']} | "
            f"{'YES' if row['hard_class_priority'] else ''} |"
        )
    source_lines = [f"- {k}: {v}" for k, v in report["source_dataset_distribution"].items()] or ["(none)"]
    license_lines = [f"- {k}: {v}" for k, v in report["license_distribution"].items()] or ["(none)"]
    lines += [
        "",
        "## Source dataset distribution",
        "",
        *source_lines,
        "",
        "## License distribution",
        "",
        *license_lines,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
