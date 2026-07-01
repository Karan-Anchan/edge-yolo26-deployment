"""Carve a small SKU-110K subset for fast pipeline smoke tests.

Run AFTER the full dataset exists in data/SKU-110K (triggered by
ultralytics' auto-download of SKU-110K.yaml).

SKU-110K ships as a FLAT layout: one images/ dir + one labels/ dir,
with train/val/test membership defined by txt list files at the root.
The subset mirrors that layout.

Usage:
    python make_subset.py             # 100 train / 20 val
    python make_subset.py -n 500      # bigger subset
"""

import argparse
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "data" / "SKU-110K"
SUBSET_DIR = REPO_ROOT / "data" / "sku110k-subset"


def copy_split(split: str, n: int, seed: int) -> None:
    entries = (SRC_DIR / f"{split}.txt").read_text().split()
    random.Random(seed).shuffle(entries)
    picked = entries[:n]
    if len(picked) < n:
        print(f"[subset] warning: only {len(picked)} {split} images available")

    kept = []
    for entry in picked:
        img = SRC_DIR / entry  # entries look like ./images/xxx.jpg
        label = SRC_DIR / "labels" / (img.stem + ".txt")
        if not (img.exists() and label.exists()):
            continue
        shutil.copy2(img, SUBSET_DIR / "images" / img.name)
        shutil.copy2(label, SUBSET_DIR / "labels" / label.name)
        kept.append(f"./images/{img.name}")

    (SUBSET_DIR / f"{split}.txt").write_text("\n".join(kept) + "\n")
    print(f"[subset] {split}: {len(kept)} images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-train", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not SRC_DIR.exists():
        raise SystemExit(f"SKU-110K not found at {SRC_DIR}. Download it first.")

    for sub in ("images", "labels"):
        (SUBSET_DIR / sub).mkdir(parents=True, exist_ok=True)

    copy_split("train", args.num_train, args.seed)
    copy_split("val", max(args.num_train // 5, 10), args.seed)

    yaml_path = SUBSET_DIR / "sku110k-subset.yaml"
    yaml_path.write_text(
        f"path: {SUBSET_DIR.as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n  0: object\n"
    )
    print(f"[subset] wrote {yaml_path}")


if __name__ == "__main__":
    main()
