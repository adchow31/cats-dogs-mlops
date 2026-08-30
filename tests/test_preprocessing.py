import os
import tempfile

from src.data_preprocessing import list_class_images, split_dataset


def test_list_class_images_filters_valid_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.jpg"), "w").close()
        open(os.path.join(tmpdir, "b.png"), "w").close()
        open(os.path.join(tmpdir, "c.txt"), "w").close()  # should be excluded
        open(os.path.join(tmpdir, "d.jpeg"), "w").close()

        result = list_class_images(tmpdir)

        assert len(result) == 3
        assert all(f.endswith((".jpg", ".png", ".jpeg")) for f in result)


def test_split_dataset_ratios_sum_correctly():
    files = [f"file_{i}.jpg" for i in range(100)]
    splits = split_dataset(files, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 100


def test_split_dataset_is_deterministic_with_same_seed():
    files = [f"file_{i}.jpg" for i in range(50)]
    splits_a = split_dataset(files, seed=42)
    splits_b = split_dataset(files, seed=42)

    assert splits_a["train"] == splits_b["train"]


def test_split_dataset_no_overlap_between_splits():
    files = [f"file_{i}.jpg" for i in range(100)]
    splits = split_dataset(files, seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)