import sys

import yaml
from PIL import Image

import training.cli as build_dataset


def _make_normalized_dataset(root, name, num_images=20):
    ds_dir = root / "normalized" / name
    images_dir = ds_dir / "images"
    labels_dir = ds_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    metadata_lines = []
    for i in range(num_images):
        image_id = f"{name}__img{i:03d}"
        Image.new("RGB", (100, 100), (i % 255, 0, 0)).save(images_dir / f"{image_id}.jpg")
        (labels_dir / f"{image_id}.txt").write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        metadata_lines.append(
            f'{{"image_id": "{image_id}", "source_dataset": "{name}", "license": "test", '
            f'"domain": "unknown", "width": 100, "height": 100}}'
        )
    (ds_dir / "metadata.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    return ds_dir


def _run_build(tmp_path, monkeypatch, output_dir, seed=42):
    normalized_root = tmp_path / "normalized"
    sources_root = tmp_path / "sources"
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    sources_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(build_dataset, "NORMALIZED_DIR", normalized_root)
    monkeypatch.setattr(build_dataset, "SOURCES_DIR", sources_root)
    monkeypatch.setattr(build_dataset, "REVIEW_DIR", review_root)

    config_path = tmp_path / "dataset_sources.yaml"
    config_path.write_text(
        yaml.safe_dump({"datasets": ["testds"], "split": {"seed": seed, "ratios": {"train": 0.7, "val": 0.2, "test": 0.1}}}),
        encoding="utf-8",
    )

    argv = ["cli.py", "build", "--config", str(config_path), "--output", str(output_dir), "--seed", str(seed)]
    monkeypatch.setattr(sys, "argv", argv)
    build_dataset.main()


def test_build_dataset_produces_data_yaml_with_16_classes(tmp_path, monkeypatch):
    _make_normalized_dataset(tmp_path, "testds", num_images=20)
    output_dir = tmp_path / "v_test"
    _run_build(tmp_path, monkeypatch, output_dir)

    data_yaml_path = output_dir / "data.yaml"
    assert data_yaml_path.is_file()
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    assert len(data["names"]) == 16
    assert data["names"][9] == "pet_bottle"
    assert data["train"] == "images/train"

    for split in ("train", "val", "test"):
        assert (output_dir / "images" / split).is_dir()
        assert (output_dir / "labels" / split).is_dir()

    for f in ("manifest.json", "stats.json", "source_manifest.json", "class_mapping.json", "duplicates.json", "review.json"):
        assert (output_dir / "metadata" / f).is_file()


def test_build_dataset_is_deterministic(tmp_path, monkeypatch):
    _make_normalized_dataset(tmp_path, "testds", num_images=30)

    out1 = tmp_path / "v1"
    _run_build(tmp_path, monkeypatch, out1, seed=42)

    # second build re-uses the same normalized source dir but writes elsewhere
    out2 = tmp_path / "v2"
    _run_build(tmp_path, monkeypatch, out2, seed=42)

    def split_contents(out_dir):
        return {
            split: sorted(p.name for p in (out_dir / "images" / split).iterdir())
            for split in ("train", "val", "test")
        }

    assert split_contents(out1) == split_contents(out2)

    manifest1 = (out1 / "metadata" / "manifest.json").read_text(encoding="utf-8")
    manifest2 = (out2 / "metadata" / "manifest.json").read_text(encoding="utf-8")
    import json

    m1, m2 = json.loads(manifest1), json.loads(manifest2)
    assert m1["num_images"] == m2["num_images"]
    assert m1["num_instances"] == m2["num_instances"]
