"""Image validation & normalization (§9): decode check, EXIF, sha256, phash.

Cộng thêm deduplication (§15): exact (SHA256) + near-duplicate (perceptual
hash). Không xóa file source — chỉ đánh dấu duplicate. Output theo schema
``duplicate_groups.json`` mô tả trong spec.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import imagehash

    HAVE_IMAGEHASH = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_IMAGEHASH = False


@dataclass
class ImageInfo:
    ok: bool
    width: int = 0
    height: int = 0
    channels: int = 0
    format: str = ""
    sha256: str = ""
    phash: str | None = None
    error: str = ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_file(path: str | Path) -> str | None:
    if not HAVE_IMAGEHASH:
        return None
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if not HAVE_IMAGEHASH:
        raise RuntimeError("imagehash not installed")
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


def inspect_image(path: str | Path) -> ImageInfo:
    """Kiểm tra corruption/zero-byte/decode, trả metadata (§9). Không ghi file."""

    path = Path(path)
    if not path.is_file():
        return ImageInfo(ok=False, error="file not found")
    if path.stat().st_size == 0:
        return ImageInfo(ok=False, error="zero-byte file")

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            channels = len(img.getbands())
            fmt = (img.format or path.suffix.lstrip(".").upper())
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageInfo(ok=False, error=f"cannot decode image: {exc}")

    return ImageInfo(
        ok=True,
        width=width,
        height=height,
        channels=channels,
        format=fmt,
        sha256=sha256_file(path),
        phash=phash_file(path),
    )


def normalize_image_to(src: str | Path, dst: str | Path) -> ImageInfo:
    """Chuẩn hóa EXIF orientation + re-encode JPG/PNG/WebP -> JPG/PNG, giữ nguyên
    kích thước (không resize source chỉ để import, §9)."""

    info = inspect_image(src)
    if not info.ok:
        return info

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        out_format = "PNG" if dst.suffix.lower() == ".png" else "JPEG"
        if out_format == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        save_kwargs = {"quality": 95} if out_format == "JPEG" else {}
        img.save(dst, format=out_format, **save_kwargs)

    return inspect_image(dst)


@dataclass
class ImageRecord:
    image_id: str
    sha256: str
    phash: str | None = None


def _union_find(items: list[str]):
    parent = {item: item for item in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


def find_duplicate_groups(
    records: list[ImageRecord],
    phash_threshold: int = 4,
) -> list[dict]:
    """Trả về list group trùng lặp: [{group_id, members, keep, duplicates}].

    - Exact duplicate: cùng SHA256.
    - Near duplicate: phash Hamming distance <= phash_threshold (nếu có phash).
    - Trong mỗi group, "keep" = image_id nhỏ nhất theo thứ tự alphabet (deterministic).
    """

    ids = [r.image_id for r in records]
    find, union = _union_find(ids)

    by_sha: dict[str, list[str]] = {}
    for r in records:
        by_sha.setdefault(r.sha256, []).append(r.image_id)
    for members in by_sha.values():
        for m in members[1:]:
            union(members[0], m)

    if HAVE_IMAGEHASH:
        hashed = [r for r in records if r.phash]
        for i in range(len(hashed)):
            for j in range(i + 1, len(hashed)):
                a, b = hashed[i], hashed[j]
                if find(a.image_id) == find(b.image_id):
                    continue
                try:
                    dist = hamming_distance(a.phash, b.phash)
                except Exception:
                    continue
                if dist <= phash_threshold:
                    union(a.image_id, b.image_id)

    groups: dict[str, list[str]] = {}
    for image_id in ids:
        root = find(image_id)
        groups.setdefault(root, []).append(image_id)

    result = []
    group_idx = 0
    for members in groups.values():
        if len(members) <= 1:
            continue
        group_idx += 1
        members_sorted = sorted(members)
        result.append(
            {
                "group_id": f"group_{group_idx:03d}",
                "members": members_sorted,
                "keep": members_sorted[0],
                "duplicates": members_sorted[1:],
            }
        )
    return result
