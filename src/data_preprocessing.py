import os
import random
import shutil
from pathlib import Path

IMAGE_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def list_class_images(class_dir: str) -> list:
    """Return sorted list of valid image file paths in a class directory."""
    valid_ext = {".jpg", ".jpeg", ".png"}
    return sorted([
        str(Path(class_dir) / f)
        for f in os.listdir(class_dir)
        if Path(f).suffix.lower() in valid_ext
    ])


def split_dataset(file_list: list, ratios: dict = SPLIT_RATIOS, seed: int = 42) -> dict:
    """Deterministically split a file list into train/val/test given ratios."""
    file_list = list(file_list)  # avoid mutating the caller's list
    random.Random(seed).shuffle(file_list)
    n = len(file_list)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": file_list[:n_train],
        "val": file_list[n_train:n_train + n_val],
        "test": file_list[n_train + n_val:],
    }


def build_split_dirs(source_root: str, dest_root: str, source_classes=("Cat", "Dog"), dest_classes=("cats", "dogs")):
    """Reorganize a flat/class-folder dataset into dest_root/{train,val,test}/{class}/."""
    for split in ["train", "val", "test"]:
        for cls in dest_classes:
            os.makedirs(Path(dest_root) / split / cls, exist_ok=True)

    for src_cls, dst_cls in zip(source_classes, dest_classes):
        class_dir = Path(source_root) / src_cls
        files = list_class_images(str(class_dir))
        splits = split_dataset(files)
        for split_name, split_files in splits.items():
            for f in split_files:
                shutil.copy(f, Path(dest_root) / split_name / dst_cls / Path(f).name)

    return {s: sum(len(os.listdir(Path(dest_root) / s / c)) for c in dest_classes) for s in ["train", "val", "test"]}