"""Single CLI entrypoint cho toàn bộ training/dataset pipeline (§34).

    python -m training.cli <subcommand> --flags...

Subcommands (xem training/README.md cho chi tiết từng bước):
    inspect            - inspect 1 dataset nguồn trước khi import
    import              - import 1 dataset nguồn vào pipeline (§7)
    normalize           - re-normalize ảnh trong 1 dataset đã import (§9)
    validate            - validate ảnh + annotation của 1 dataset (§9, §10)
    dedup               - tìm exact/near-duplicate ảnh (§15)
    split               - group-aware train/val/test split (§16, §17)
    build               - build 1 dataset version YOLO-ready (§24-26)
    report              - class coverage + dataset statistics report (§12-14)
    train               - train baseline detector (§27)
    evaluate            - evaluate 1 checkpoint đã train (§28)
    auto-label          - pseudo-label ảnh mới bằng model đã train (§32)
    download-roboflow   - tải 1 dataset version từ Roboflow API (§8, optional)

Mỗi subcommand chỉ import các dependency optional (torch/ultralytics/roboflow)
bên trong hàm xử lý của nó — ``--help`` hay các subcommand khác không cần các
thư viện này được cài.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training.lib.dataset_ops import (
    DatasetManifest,
    SplitRatios,
    group_key_for,
    load_normalized_dataset,
    read_review_manifest,
    split_items,
    verify_no_leakage,
)
from training.lib.formats import (
    detect_format,
    maybe_extract_zip,
    parse_yolo_line,
    read_dataset,
    validate_bbox,
    validate_label_file,
)
from training.lib.imageio_utils import ImageRecord, find_duplicate_groups, inspect_image, normalize_image_to
from training.lib.importer import import_dataset
from training.lib.report import build_report_dict, compute_dataset_stats, write_report_csv, write_report_json, write_report_md
from training.lib.taxonomy import MappingEngine, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "training" / "datasets" / "raw"
SOURCES_DIR = REPO_ROOT / "training" / "datasets" / "sources"
NORMALIZED_DIR = REPO_ROOT / "training" / "datasets" / "normalized"
REVIEW_DIR = REPO_ROOT / "training" / "datasets" / "review"
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "training" / "configs" / "training.yaml"
DEFAULT_DATASET_SOURCES_CONFIG = REPO_ROOT / "training" / "configs" / "dataset_sources.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def configure_utf8_stdout() -> None:
    """Tránh lỗi encode khi in tiếng Việt trên console non-UTF8 (Windows)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def inspect_source_dataset(input_path: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="inspect_") as tmp:
        extracted = maybe_extract_zip(input_path, tmp)
        fmt = detect_format(extracted)
        images = read_dataset(extracted, fmt)

        class_counts: Counter = Counter()
        resolutions: Counter = Counter()
        images_with_boxes = 0
        total_boxes = 0
        for img in images:
            resolutions[f"{img.width}x{img.height}"] += 1
            if img.boxes:
                images_with_boxes += 1
            for box in img.boxes:
                class_counts[box.label] += 1
                total_boxes += 1

        return {
            "input": input_path,
            "detected_format": fmt,
            "image_count": len(images),
            "images_with_annotations": images_with_boxes,
            "images_without_annotations": len(images) - images_with_boxes,
            "total_instances": total_boxes,
            "original_classes": sorted(class_counts.keys()),
            "instances_per_original_class": dict(class_counts.most_common()),
            "resolution_distribution": dict(resolutions.most_common(20)),
        }


def cmd_inspect(args: argparse.Namespace) -> None:
    report = inspect_source_dataset(args.input)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(text)


def _add_inspect_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("inspect", help="Inspect a source dataset (format, classes, image/instance counts)")
    ap.add_argument("--input", required=True, help="path tới thư mục dataset hoặc file .zip")
    ap.add_argument("--output", default=None, help="ghi report ra file JSON (mặc định in stdout)")
    ap.set_defaults(func=cmd_inspect)


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def cmd_import(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest) if args.manifest else SOURCES_DIR / args.name / "manifest.yaml"
    if manifest_path.is_file():
        manifest = DatasetManifest.load(manifest_path)
    else:
        manifest = DatasetManifest(
            name=args.name,
            source=args.source,
            url=args.url,
            source_type=args.source_type,
            license=args.license,
            domain=args.domain,
            source_weight=args.source_weight,
            status="registered",
        )

    taxonomy = load_taxonomy()
    mapping_engine = MappingEngine.load(taxonomy)

    fmt = None if args.source_type == "auto" else args.source_type
    summary = import_dataset(
        source=args.source,
        dataset_name=args.name,
        manifest=manifest,
        taxonomy=taxonomy,
        mapping_engine=mapping_engine,
        normalized_root=NORMALIZED_DIR,
        review_root=REVIEW_DIR,
        source_format=fmt,
        dry_run=args.dry_run,
    )

    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))

    if not args.dry_run:
        manifest.image_count = summary.images_imported
        manifest.original_classes = sorted(summary.original_classes_seen)
        manifest.annotation_format = manifest.annotation_format or (fmt or "auto")
        manifest.status = "imported"
        manifest.save(manifest_path)
        print(f"Manifest updated: {manifest_path}")
    else:
        print("[dry-run] Không ghi file nào.")


def _add_import_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("import", help="Import a source dataset into the normalized pipeline")
    ap.add_argument("--source", required=True, help="path tới thư mục dataset hoặc file .zip")
    ap.add_argument("--name", required=True, help="tên dataset (dùng làm thư mục con)")
    ap.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "yolo", "coco", "voc", "generic_json"],
        help="ép format thay vì auto-detect",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="path manifest.yaml (mặc định training/datasets/sources/<name>/manifest.yaml)",
    )
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--url", default="")
    ap.add_argument("--domain", default="unknown")
    ap.add_argument("--source-weight", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--dry-run", action="store_true", help="không ghi file, chỉ in summary")
    ap.set_defaults(func=cmd_import)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def cmd_normalize(args: argparse.Namespace) -> None:
    root = Path(args.input)
    images_dir = root / "images" if (root / "images").is_dir() else root

    kept, fixed, removed, errors = 0, 0, 0, []
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        info_before = inspect_image(image_path)
        if not info_before.ok:
            removed += 1
            errors.append({"image": str(image_path), "error": info_before.error})
            continue

        if args.dry_run:
            kept += 1
            continue

        target = image_path if image_path.suffix.lower() not in (".webp", ".bmp") else image_path.with_suffix(".jpg")
        info_after = normalize_image_to(image_path, target)
        if not info_after.ok:
            removed += 1
            errors.append({"image": str(image_path), "error": info_after.error})
            continue
        if target != image_path:
            image_path.unlink(missing_ok=True)
            fixed += 1
        kept += 1

    report = {"input": str(root), "kept": kept, "fixed": fixed, "removed": removed, "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[dry-run] Không ghi file nào.")


def _add_normalize_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("normalize", help="Normalize images in-place (EXIF, corruption check)")
    ap.add_argument("--input", required=True, help="thư mục dataset (chứa images/ hoặc ảnh trực tiếp)")
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(func=cmd_normalize)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> None:
    root = Path(args.input)
    images_dir = root / "images"
    labels_dir = root / "labels"
    taxonomy = load_taxonomy()

    image_errors: list[dict] = []
    annotation_errors: list[dict] = []
    valid_images = 0
    negative_images = 0

    for image_path in sorted(images_dir.iterdir()) if images_dir.is_dir() else []:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        info = inspect_image(image_path)
        if not info.ok:
            image_errors.append({"image": str(image_path), "error": info.error})
            continue
        valid_images += 1

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file() or label_path.stat().st_size == 0:
            negative_images += 1
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()
        result = validate_label_file(lines, taxonomy.num_classes)
        if result["errors"]:
            annotation_errors.append({"label_file": str(label_path), "errors": result["errors"]})

    report = {
        "input": str(root),
        "valid_images": valid_images,
        "negative_images": negative_images,
        "image_error_count": len(image_errors),
        "image_errors": image_errors,
        "annotation_error_files": len(annotation_errors),
        "annotation_errors": annotation_errors,
        "ok": not image_errors and not annotation_errors,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(text)

    sys.exit(0 if report["ok"] else 1)


def _add_validate_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("validate", help="Validate images + YOLO annotations of a normalized dataset")
    ap.add_argument("--input", required=True, help="thư mục dataset chứa images/ + labels/")
    ap.add_argument("--output", default=None, help="ghi report JSON ra file (mặc định stdout)")
    ap.set_defaults(func=cmd_validate)


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def cmd_dedup(args: argparse.Namespace) -> None:
    root = Path(args.input)
    images_dir = root / "images" if (root / "images").is_dir() else root

    records = []
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        info = inspect_image(image_path)
        if info.ok:
            records.append(ImageRecord(image_id=str(image_path), sha256=info.sha256, phash=info.phash))

    groups = find_duplicate_groups(records, phash_threshold=args.phash_threshold)
    total_duplicates = sum(len(g["duplicates"]) for g in groups)
    print(f"images scanned: {len(records)} | duplicate groups: {len(groups)} | duplicate images: {total_duplicates}")

    if not args.dry_run:
        output_path = Path(args.output) if args.output else root / "duplicates.json"
        output_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Ghi: {output_path}")
    else:
        print("[dry-run] Không ghi file nào.")
        print(json.dumps(groups, indent=2, ensure_ascii=False))


def _add_dedup_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("dedup", help="Find exact (SHA256) and near-duplicate (phash) images")
    ap.add_argument("--input", required=True, help="thư mục dataset chứa images/")
    ap.add_argument("--phash-threshold", type=int, default=4, help="Hamming distance tối đa cho near-duplicate")
    ap.add_argument("--output", default=None, help="mặc định <input>/duplicates.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(func=cmd_dedup)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def cmd_split(args: argparse.Namespace) -> None:
    train, val, test = (float(x) for x in args.ratios.split(","))
    ratios = SplitRatios(train=train, val=val, test=test)

    items = load_normalized_dataset(args.input)
    if not items:
        print(f"Không tìm thấy ảnh nào trong {args.input}", file=sys.stderr)
        sys.exit(1)

    item_to_group = {item.image_id: group_key_for(item) for item in items}
    pairs = [(item.image_id, item_to_group[item.image_id]) for item in items]
    result = split_items(pairs, ratios=ratios, seed=args.seed)

    leakage_errors = verify_no_leakage(result, item_to_group)
    if leakage_errors:
        print("LEAKAGE DETECTED:", leakage_errors, file=sys.stderr)
        sys.exit(1)

    summary = {
        "input": args.input,
        "seed": args.seed,
        "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "counts": {k: len(v) for k, v in result.items()},
        "num_groups": len(set(item_to_group.values())),
        "leakage_check": "PASS",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.dry_run:
        output_path = Path(args.output) if args.output else Path(args.input) / "split_assignment.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Ghi: {output_path}")
    else:
        print("[dry-run] Không ghi file nào.")


def _add_split_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("split", help="Group-aware train/val/test split (no leakage across splits)")
    ap.add_argument("--input", required=True, help="thư mục dataset chứa images/ + labels/ + metadata.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.7,0.2,0.1", help="train,val,test — mặc định 70/20/10")
    ap.add_argument("--output", default=None, help="mặc định ghi split_assignment.json vào --input")
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(func=cmd_split)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def cmd_build(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_names: list[str] = config.get("datasets", [])
    split_cfg = config.get("split", {})
    seed = args.seed if args.seed is not None else int(split_cfg.get("seed", 42))
    if args.ratios:
        tr, va, te = (float(x) for x in args.ratios.split(","))
    else:
        r = split_cfg.get("ratios", {})
        tr, va, te = float(r.get("train", 0.7)), float(r.get("val", 0.2)), float(r.get("test", 0.1))
    ratios = SplitRatios(train=tr, val=va, test=te)

    taxonomy = load_taxonomy()
    warnings: list[str] = []

    all_items = []
    source_manifests = {}
    for name in dataset_names:
        ds_dir = NORMALIZED_DIR / name
        items = load_normalized_dataset(ds_dir)
        if not items:
            warnings.append(f"dataset '{name}' has no images in {ds_dir} — skipped")
            continue
        all_items.extend(items)
        manifest_path = SOURCES_DIR / name / "manifest.yaml"
        if manifest_path.is_file():
            source_manifests[name] = DatasetManifest.load(manifest_path).__dict__
        else:
            warnings.append(f"dataset '{name}' missing source manifest at {manifest_path}")

    if not all_items:
        print("Không có ảnh nào để build (kiểm tra dataset_sources.yaml và training/datasets/normalized/)")
        sys.exit(1)

    # --- dedup toàn cục qua mọi dataset đã gộp ---
    records = []
    for item in all_items:
        info = inspect_image(item.image_path)
        if info.ok:
            records.append(ImageRecord(image_id=item.image_id, sha256=info.sha256, phash=info.phash))
    dup_groups = find_duplicate_groups(records, phash_threshold=args.phash_threshold)
    duplicate_ids = {d for g in dup_groups for d in g["duplicates"]}

    kept_items = [it for it in all_items if it.image_id not in duplicate_ids]

    # --- annotation validation trước khi build (bỏ label file lỗi, giữ ảnh làm negative) ---
    validation_errors: dict[str, list[str]] = {}
    for item in kept_items:
        if not item.label_path.is_file():
            continue
        lines = item.label_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            box, errs = parse_yolo_line(line)
            if errs or box is None:
                validation_errors.setdefault(item.image_id, []).extend(errs)
                continue
            box_errs = validate_bbox(box, taxonomy.num_classes)
            if box_errs:
                validation_errors.setdefault(item.image_id, []).extend(f"line {i}: {e}" for e in box_errs)

    # --- group-aware split ---
    item_to_group = {it.image_id: group_key_for(it) for it in kept_items}
    pairs = [(it.image_id, item_to_group[it.image_id]) for it in kept_items]
    split_result = split_items(pairs, ratios=ratios, seed=seed)
    leakage_errors = verify_no_leakage(split_result, item_to_group)
    if leakage_errors:
        print("LEAKAGE DETECTED:", leakage_errors, file=sys.stderr)
        sys.exit(1)

    review_records = read_review_manifest(REVIEW_DIR / "review_manifest.jsonl")
    review_for_datasets = [r for r in review_records if r["source_dataset"] in dataset_names]

    plan = {
        "sources": dataset_names,
        "seed": seed,
        "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "total_images_before_dedup": len(all_items),
        "duplicate_images_removed": len(duplicate_ids),
        "total_images_after_dedup": len(kept_items),
        "split_counts": {k: len(v) for k, v in split_result.items()},
        "ambiguous_labels_in_review_queue": len(review_for_datasets),
        "annotation_error_images": len(validation_errors),
        "classes": list(taxonomy.classes),
        "warnings": warnings,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("[dry-run] Không ghi file nào.")
        return

    output_dir = Path(args.output)
    images_by_id = {it.image_id: it for it in kept_items}
    id_to_split = {img_id: split for split, ids in split_result.items() for img_id in ids}

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    combined_metadata_lines = []
    for image_id, item in images_by_id.items():
        split = id_to_split[image_id]
        dest_image = output_dir / "images" / split / item.image_path.name
        shutil.copy2(item.image_path, dest_image)
        dest_label = output_dir / "labels" / split / f"{item.image_id}.txt"
        if item.label_path.is_file():
            shutil.copy2(item.label_path, dest_label)
        else:
            dest_label.write_text("", encoding="utf-8")
        meta = dict(item.metadata)
        meta["split"] = split
        combined_metadata_lines.append(json.dumps(meta, ensure_ascii=False))

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "combined_metadata.jsonl").write_text(
        "\n".join(combined_metadata_lines) + "\n", encoding="utf-8"
    )
    (metadata_dir / "duplicates.json").write_text(json.dumps(dup_groups, indent=2, ensure_ascii=False), encoding="utf-8")
    (metadata_dir / "source_manifest.json").write_text(
        json.dumps(source_manifests, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (metadata_dir / "class_mapping.json").write_text(
        json.dumps(
            {
                "classes": list(taxonomy.classes),
                "class_to_group": taxonomy.class_group,
                "group_names": taxonomy.group_names,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "review.json").write_text(
        json.dumps(review_for_datasets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metadata_dir / "validation_report.json").write_text(
        json.dumps(validation_errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stats = compute_dataset_stats(
        images_dir=output_dir / "images",
        labels_dir=output_dir / "labels",
        taxonomy=taxonomy,
        metadata_path=metadata_dir / "combined_metadata.jsonl",
        duplicate_groups_path=metadata_dir / "duplicates.json",
    )
    stats_report = build_report_dict(stats, taxonomy)
    (metadata_dir / "stats.json").write_text(json.dumps(stats_report, indent=2, ensure_ascii=False), encoding="utf-8")

    build_manifest = {
        "version": output_dir.name,
        "sources": dataset_names,
        "seed": seed,
        "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "num_images": len(kept_items),
        "num_instances": stats.total_instances,
        "duplicate_images_removed": len(duplicate_ids),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(build_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(taxonomy.classes)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Dataset version: {output_dir.name}",
                "",
                f"- Sources: {', '.join(dataset_names)}",
                f"- Seed: {seed}",
                f"- Images: {len(kept_items)} (duplicates removed: {len(duplicate_ids)})",
                f"- Instances: {stats.total_instances}",
                f"- Split: train={len(split_result['train'])} val={len(split_result['val'])} test={len(split_result['test'])}",
                "",
                "Xem `metadata/` cho manifest/stats/mapping/duplicates/review đầy đủ.",
                "Xem `training/README.md` cho quy trình build/train/evaluate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Đã build dataset version tại: {output_dir}")


def _add_build_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("build", help="Build a reproducible YOLO-ready dataset version")
    ap.add_argument("--config", default=str(DEFAULT_DATASET_SOURCES_CONFIG))
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override seed trong config")
    ap.add_argument("--ratios", default=None, help="train,val,test — override config, vd 0.7,0.2,0.1")
    ap.add_argument("--phash-threshold", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(func=cmd_build)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    root = Path(args.input)
    images_dir = root / "images"
    labels_dir = root / "labels"
    metadata_path = root / "metadata" / "combined_metadata.jsonl"
    if not metadata_path.is_file():
        metadata_path = root / "metadata.jsonl"
    duplicates_path = root / "metadata" / "duplicates.json"
    if not duplicates_path.is_file():
        duplicates_path = root / "duplicates.json"

    taxonomy = load_taxonomy()
    stats = compute_dataset_stats(
        images_dir=images_dir,
        labels_dir=labels_dir,
        taxonomy=taxonomy,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        duplicate_groups_path=duplicates_path if duplicates_path.is_file() else None,
    )
    report = build_report_dict(stats, taxonomy)

    output_dir = Path(args.output_dir) if args.output_dir else root
    write_report_json(report, output_dir / "dataset_report.json")
    write_report_csv(report, output_dir / "dataset_report.csv")
    write_report_md(report, output_dir / "dataset_report.md")

    print(f"total_images={report['total_images']} total_instances={report['total_instances']}")
    print(f"Report written to {output_dir}/dataset_report.{{json,csv,md}}")


def _add_report_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("report", help="Generate dataset_report.{json,csv,md}")
    ap.add_argument("--input", required=True, help="thư mục dataset (chứa images/train,val,test hoặc images/ phẳng)")
    ap.add_argument("--output-dir", default=None, help="mặc định = --input")
    ap.set_defaults(func=cmd_report)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_yaml_config(args.config)
    for key in ("model", "data", "imgsz", "epochs", "batch", "device", "workers", "seed", "project", "name"):
        cli_val = getattr(args, key)
        if cli_val is not None:
            cfg[key] = cli_val
    if args.cache is not None:
        cfg["cache"] = False if args.cache == "false" else args.cache
    if args.plots is not None:
        cfg["plots"] = args.plots == "true"
    cfg.setdefault("cache", False)
    cfg.setdefault("plots", True)

    try:
        batch = int(cfg["batch"])
    except (TypeError, ValueError):
        batch = cfg["batch"]  # "-1" auto hoặc "auto" giữ nguyên cho ultralytics xử lý

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics chưa được cài trong .venv. Cài: pip install ultralytics torch\n"
            f"(chi tiết: {exc})"
        )

    data_path = (REPO_ROOT / cfg["data"]) if not Path(cfg["data"]).is_absolute() else Path(cfg["data"])
    if not data_path.is_file():
        raise SystemExit(f"Không tìm thấy data.yaml: {data_path} — chạy `python -m training.cli build` trước.")

    model = YOLO(cfg["model"])
    results = model.train(
        data=str(data_path),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=batch,
        device=cfg["device"],
        workers=cfg["workers"],
        seed=cfg["seed"],
        project=str(REPO_ROOT / cfg["project"]),
        name=cfg["name"],
        cache=cfg["cache"],
        plots=cfg["plots"],
    )
    print(f"Training xong. Kết quả: {results.save_dir}")


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("train", help="Train baseline YOLOv8n/YOLO11n detector (§27)")
    ap.add_argument("--config", default=str(DEFAULT_TRAINING_CONFIG))
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--cache", default=None, choices=["ram", "disk", "false"], help="ultralytics image cache — để trống dùng giá trị trong config (mặc định false, tốn RAM nếu bật)")
    ap.add_argument("--plots", default=None, choices=["true", "false"], help="tắt để giảm RAM/CPU khi máy hạn chế tài nguyên")
    ap.set_defaults(func=cmd_train)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"ultralytics chưa được cài. Cài: pip install ultralytics torch\n({exc})")

    taxonomy = load_taxonomy()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, project=str(REPO_ROOT / "training" / "runs"), name="eval")

    # metrics.box.p/r/ap50/ap chỉ chứa các class THỰC SỰ CÓ trong split (độ dài =
    # len(ap_class_index)), không phải full 16 class theo thứ tự 0..15 — phải map
    # qua ap_class_index, nếu không precision/recall sẽ bị lệch sang class khác.
    class_names = [metrics.names[i] for i in sorted(metrics.names)]
    present_class_ids = [int(i) for i in metrics.box.ap_class_index]
    metrics_by_class_id = {
        class_id: {
            "precision": float(metrics.box.p[pos]),
            "recall": float(metrics.box.r[pos]),
            "ap50": float(metrics.box.ap50[pos]),
            "ap50_95": float(metrics.box.ap[pos]),
        }
        for pos, class_id in enumerate(present_class_ids)
    }

    per_class = []
    for class_id, name in enumerate(class_names):
        values = metrics_by_class_id.get(class_id)
        per_class.append(
            {
                "class": name,
                "precision": values["precision"] if values else None,
                "recall": values["recall"] if values else None,
                "ap50": values["ap50"] if values else None,
                "ap50_95": values["ap50_95"] if values else None,
            }
        )

    low_recall = sorted(
        [c for c in per_class if c["recall"] is not None],
        key=lambda c: c["recall"],
    )[:5]

    group_names = ("METAL", "PAPER", "PLASTIC", "GLASS")
    confusion_note = (
        "Confusion matrix chi tiết theo nhóm cần ma trận confusion đầy đủ từ "
        "metrics.confusion_matrix (ultralytics) — được lưu trong thư mục run "
        "(confusion_matrix.png / .csv). Recall thấp theo class được highlight ở "
        "'low_recall_classes' vì hệ thống sorting cần ưu tiên recall (§28)."
    )

    report = {
        "weights": args.weights,
        "split": args.split,
        "overall": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision_mean": float(metrics.box.mp),
            "recall_mean": float(metrics.box.mr),
            "f1_mean": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr))
            if (metrics.box.mp + metrics.box.mr) > 0
            else 0.0,
        },
        "per_class": per_class,
        "low_recall_classes": low_recall,
        "sorting_groups": list(group_names),
        "confusion_note": confusion_note,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(text)


def _add_evaluate_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("evaluate", help="Evaluate a trained checkpoint (mAP, per-class, group confusion)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output", default=None, help="ghi report JSON (mặc định stdout)")
    ap.set_defaults(func=cmd_evaluate)


# ---------------------------------------------------------------------------
# auto-label
# ---------------------------------------------------------------------------

def cmd_auto_label(args: argparse.Namespace) -> None:
    cfg = _load_yaml_config(args.config)
    auto_cfg = cfg.get("auto_label", {})
    accept_threshold = args.accept_threshold or float(auto_cfg.get("accept_threshold", 0.90))
    review_threshold = args.review_threshold or float(auto_cfg.get("review_threshold", 0.50))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"ultralytics chưa được cài. Cài: pip install ultralytics torch\n({exc})")

    taxonomy = load_taxonomy()
    model = YOLO(args.model)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    buckets = {"auto_labels": 0, "review": 0, "uncertain": 0}

    for image_path in sorted(input_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        result = model.predict(str(image_path), verbose=False)[0]
        lines_by_bucket: dict[str, list[str]] = {"auto_labels": [], "review": [], "uncertain": []}
        metadata = []
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            xywhn = box.xywhn[0].tolist()
            line = f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
            if conf >= accept_threshold:
                bucket = "auto_labels"
            elif conf >= review_threshold:
                bucket = "review"
            else:
                bucket = "uncertain"
            lines_by_bucket[bucket].append(line)
            metadata.append(
                {
                    "confidence": conf,
                    "predicted_class": taxonomy.name_for(cls_id) if taxonomy.is_valid_class_id(cls_id) else str(cls_id),
                    "bbox": xywhn,
                    "image_path": str(image_path),
                    "model_version": args.model,
                }
            )

        for bucket, lines in lines_by_bucket.items():
            if not lines:
                continue
            buckets[bucket] += len(lines)
            if args.dry_run:
                continue
            bucket_dir = output_dir / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            (bucket_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (bucket_dir / f"{image_path.stem}.jpg").write_bytes(image_path.read_bytes())

        if metadata and not args.dry_run:
            meta_dir = output_dir / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            with (meta_dir / "predictions.jsonl").open("a", encoding="utf-8") as f:
                for m in metadata:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(json.dumps({"thresholds": {"accept": accept_threshold, "review": review_threshold}, "counts": buckets}, indent=2))
    if args.dry_run:
        print("[dry-run] Không ghi file nào.")


def _add_auto_label_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("auto-label", help="Pseudo-label incoming images with a trained model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(REPO_ROOT / "training" / "datasets" / "auto_label_output"))
    ap.add_argument("--config", default=str(DEFAULT_TRAINING_CONFIG))
    ap.add_argument("--accept-threshold", type=float, default=None)
    ap.add_argument("--review-threshold", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(func=cmd_auto_label)


# ---------------------------------------------------------------------------
# download-roboflow
# ---------------------------------------------------------------------------

def cmd_download_roboflow(args: argparse.Namespace) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY chưa được set trong environment. Không hard-code API key (§8).\n"
            "Set biến môi trường rồi chạy lại, hoặc export dataset thủ công (ZIP) và dùng `python -m training.cli import` trực tiếp."
        )

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(f"Thư viện 'roboflow' chưa được cài. Cài: pip install roboflow\n({exc})")

    output_dir = RAW_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    print(f"Downloading {args.workspace}/{args.project} v{args.version} "
          f"({version.images} images) format={args.format} -> {output_dir}")
    version.download(args.format, location=str(output_dir), overwrite=True)

    print(f"Đã tải xong: {output_dir}")
    print(
        "Bước tiếp theo:\n"
        f"  python -m training.cli import --source {output_dir} --name {args.name} --source-type yolo"
    )


def _add_download_roboflow_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("download-roboflow", help="Download a public Roboflow dataset version via API")
    ap.add_argument("--workspace", required=True, help="Roboflow workspace slug (từ URL universe.roboflow.com/<workspace>/<project>)")
    ap.add_argument("--project", required=True, help="Roboflow project slug")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--name", required=True, help="tên dataset nội bộ (thư mục output + dùng cho `import` --name)")
    ap.add_argument("--format", default="yolov8", help="export format Roboflow (mặc định yolov8)")
    ap.set_defaults(func=cmd_download_roboflow)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="training.cli", description="auto-sort-pi training/dataset pipeline CLI")
    sub = ap.add_subparsers(dest="command", required=True)
    _add_inspect_parser(sub)
    _add_import_parser(sub)
    _add_normalize_parser(sub)
    _add_validate_parser(sub)
    _add_dedup_parser(sub)
    _add_split_parser(sub)
    _add_build_parser(sub)
    _add_report_parser(sub)
    _add_train_parser(sub)
    _add_evaluate_parser(sub)
    _add_auto_label_parser(sub)
    _add_download_roboflow_parser(sub)
    return ap


def main() -> None:
    configure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
