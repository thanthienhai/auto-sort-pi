"""Import pipeline (§7): source -> detect format -> map classes -> normalize
images -> convert annotations -> split accepted/review.

Ambiguous (REVIEW_REQUIRED) instances KHÔNG được đưa vào normalized dataset —
chúng được ghi vào review queue (§11) kèm ảnh gốc để người dùng review sau.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from training.lib.dataset_ops import DatasetManifest, ReviewRecord, write_review_manifest
from training.lib.formats import maybe_extract_zip, read_dataset
from training.lib.imageio_utils import inspect_image, normalize_image_to
from training.lib.taxonomy import MappingEngine, MappingType, Taxonomy


@dataclass
class ImportSummary:
    dataset_name: str
    images_seen: int = 0
    images_imported: int = 0
    instances_accepted: int = 0
    instances_review: int = 0
    instances_ignored: int = 0
    original_classes_seen: set = field(default_factory=set)
    mapping_breakdown: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["original_classes_seen"] = sorted(self.original_classes_seen)
        return d


def import_dataset(
    source: str | Path,
    dataset_name: str,
    manifest: DatasetManifest,
    taxonomy: Taxonomy,
    mapping_engine: MappingEngine,
    normalized_root: str | Path,
    review_root: str | Path,
    source_format: str | None = None,
    dry_run: bool = False,
) -> ImportSummary:
    normalized_root = Path(normalized_root) / dataset_name
    review_root = Path(review_root)
    summary = ImportSummary(dataset_name=dataset_name)

    with tempfile.TemporaryDirectory(prefix=f"import_{dataset_name}_") as tmp:
        extracted = maybe_extract_zip(source, tmp)
        raw_images = read_dataset(extracted, source_format)
        summary.images_seen = len(raw_images)

        images_out = normalized_root / "images"
        labels_out = normalized_root / "labels"
        metadata_path = normalized_root / "metadata.jsonl"
        review_records: list[ReviewRecord] = []
        metadata_lines: list[str] = []

        if not dry_run:
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)

        for raw_image in raw_images:
            info = inspect_image(raw_image.image_path)
            if not info.ok:
                summary.warnings.append(f"skip corrupted/unreadable image: {raw_image.image_path} ({info.error})")
                continue

            accepted_lines = []
            for box in raw_image.boxes:
                summary.original_classes_seen.add(box.label)
                result = mapping_engine.resolve(box.label, manifest.target_mapping)
                summary.mapping_breakdown.setdefault(result.mapping_type.value, 0)
                summary.mapping_breakdown[result.mapping_type.value] += 1

                if result.mapping_type == MappingType.REVIEW_REQUIRED:
                    summary.instances_review += 1
                    review_records.append(
                        ReviewRecord(
                            image=str(raw_image.image_path),
                            source_dataset=dataset_name,
                            source_class=box.label,
                            candidate_targets=list(result.candidate_targets),
                            reason=result.note or "Ambiguous mapping",
                            bbox=[box.x, box.y, box.w, box.h],
                        )
                    )
                    continue

                if result.target_class is None or result.target_class not in taxonomy.name_to_id:
                    summary.instances_ignored += 1
                    continue

                class_id = taxonomy.id_for(result.target_class)
                accepted_lines.append(f"{class_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}")
                summary.instances_accepted += 1

            image_id = f"{dataset_name}__{info.sha256[:16]}"
            ext = raw_image.image_path.suffix.lower()
            out_ext = ext if ext in {".jpg", ".jpeg", ".png"} else ".jpg"

            if not dry_run:
                normalize_image_to(raw_image.image_path, images_out / f"{image_id}{out_ext}")
                (labels_out / f"{image_id}.txt").write_text(
                    "\n".join(accepted_lines) + ("\n" if accepted_lines else ""), encoding="utf-8"
                )
                metadata_lines.append(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "source_dataset": dataset_name,
                            "license": manifest.license,
                            "domain": manifest.domain,
                            "source_weight": manifest.source_weight,
                            "width": info.width,
                            "height": info.height,
                            "sha256": info.sha256,
                            "phash": info.phash,
                            "original_path": str(raw_image.image_path),
                        },
                        ensure_ascii=False,
                    )
                )
            summary.images_imported += 1

        if not dry_run:
            metadata_path.write_text("\n".join(metadata_lines) + ("\n" if metadata_lines else ""), encoding="utf-8")
            if review_records:
                _append_review_records(review_records, review_root, dataset_name)

    return summary


def _append_review_records(records: list[ReviewRecord], review_root: str | Path, dataset_name: str) -> None:
    review_root = Path(review_root)
    manifest_path = review_root / "review_manifest.jsonl"
    existing = []
    if manifest_path.is_file():
        existing = manifest_path.read_text(encoding="utf-8").splitlines()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    for rec in records:
        bucket = rec.candidate_targets[0] if rec.candidate_targets else "unknown"
        image_src = Path(rec.image)
        bucket_dir = review_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        dest = bucket_dir / f"{dataset_name}__{image_src.name}"
        if image_src.is_file() and not dest.exists():
            dest.write_bytes(image_src.read_bytes())
