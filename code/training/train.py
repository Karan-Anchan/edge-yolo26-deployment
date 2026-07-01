"""Fine-tune YOLO26 on SKU-110K (dense retail-shelf detection).

Usage:
    python train.py --smoke              # 3-epoch run on the 100-image subset
    python train.py                      # full fine-tune (yolo26s, 640px)
    python train.py --model yolo26n.pt   # smaller/faster variant

The dataset yaml is resolved by Ultralytics: "SKU-110K.yaml" auto-downloads
the full set (~13.6 GB) into the configured datasets_dir; the subset yaml is
produced by scripts/make_subset.py.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]

# SKU-110K averages ~150 boxes/image (peaks far higher), so the default
# max_det=300 clips real objects during val — raise it or recall lies low.
MAX_DET = 600


def load_model(name: str) -> YOLO:
    """Load the requested checkpoint, falling back to YOLOv8 (the baseline)."""
    try:
        return YOLO(name)
    except Exception as err:
        fallback = name.replace("yolo26", "yolov8")
        if fallback == name:
            raise
        print(f"[train] could not load {name} ({err}); falling back to {fallback}")
        return YOLO(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--data", default="SKU-110K.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto (fits VRAM)")
    parser.add_argument("--smoke", action="store_true",
                        help="3 epochs on the 100-image subset (pipeline check)")
    args = parser.parse_args()

    if args.smoke:
        args.model = "yolo26n.pt"
        args.data = str(REPO_ROOT / "data" / "sku110k-subset" / "sku110k-subset.yaml")
        args.epochs = 3

    model = load_model(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        max_det=MAX_DET,
        project=str(REPO_ROOT / "results" / "training"),
        name="smoke" if args.smoke else Path(args.model).stem,
        exist_ok=True,
        # dense single-class scenes: mosaic stays on, but disable it for the
        # final 10 epochs so boxes near tile seams stop hurting late training
        close_mosaic=0 if args.smoke else 10,
    )

    run_name = "smoke" if args.smoke else Path(args.model).stem
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, max_det=MAX_DET,
        project=str(REPO_ROOT / "results" / "training"),
        name=f"{run_name}_val", exist_ok=True,
    )
    print(f"[train] mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
