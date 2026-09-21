from training.lib.imageio_utils import ImageRecord, find_duplicate_groups


def test_exact_duplicate_by_sha256():
    records = [
        ImageRecord("a.jpg", sha256="hash1", phash=None),
        ImageRecord("b.jpg", sha256="hash1", phash=None),
        ImageRecord("c.jpg", sha256="hash2", phash=None),
    ]
    groups = find_duplicate_groups(records)
    assert len(groups) == 1
    assert groups[0]["keep"] == "a.jpg"
    assert groups[0]["duplicates"] == ["b.jpg"]


def test_no_duplicates_when_all_unique():
    records = [
        ImageRecord("a.jpg", sha256="h1"),
        ImageRecord("b.jpg", sha256="h2"),
    ]
    assert find_duplicate_groups(records) == []


def test_duplicate_group_deterministic_keep_is_alphabetical():
    records = [
        ImageRecord("z.jpg", sha256="h1"),
        ImageRecord("a.jpg", sha256="h1"),
        ImageRecord("m.jpg", sha256="h1"),
    ]
    groups = find_duplicate_groups(records)
    assert groups[0]["keep"] == "a.jpg"
    assert groups[0]["duplicates"] == ["m.jpg", "z.jpg"]


def test_does_not_mutate_or_delete_anything():
    # find_duplicate_groups is pure — it must not touch the filesystem.
    records = [ImageRecord("a.jpg", sha256="h1"), ImageRecord("b.jpg", sha256="h1")]
    groups = find_duplicate_groups(records)
    assert groups[0]["members"] == ["a.jpg", "b.jpg"]
