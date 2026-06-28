"""Build TensorRT engines (FP16, INT8) from the trained YOLO26 model and
report each engine's mAP drop vs the FP32 ONNX baseline (0.5716).

Runs on the RTX 5070 (Blackwell). FP4 is handled separately in build_fp4.py
because it needs an explicit ModelOpt quantization pass, not a build flag.

We build via Ultralytics' engine export rather than raw trtexec because it
wires up the INT8 calibrator against our dataset automatically and keeps the
exact same preprocessing/imgsz/max_det as training and the ONNX export — so
the only variable across rows is *precision*, which is the whole point of the
study.

Precision ladder (accuracy is measured as a drop from FP32 mAP50-95 0.5716;
≤2% loss ⇒ stay above ~0.560):
    FP16 — halve the bits, near-lossless, the "free" speedup
    INT8 — 8-bit, needs calibration, the real accuracy/speed trade-off
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from engine_utils import validate_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "results" / "training" / "yolo26s" / "weights" / "best.pt"
CALIB_YAML = REPO_ROOT / "data" / "sku110k-subset" / "sku110k-subset.yaml"
ENGINE_DIR = REPO_ROOT / "models" / "tensorrt"

IMGSZ = 640
MAX_DET = 600


def build(precision: str) -> Path:
    """Export a TensorRT engine at the requested precision."""
    model = YOLO(str(WEIGHTS))
    kwargs = dict(format="engine", imgsz=IMGSZ, batch=1,
                  max_det=MAX_DET, device=0, workspace=8)
    if precision == "fp16":
        kwargs["half"] = True
    elif precision == "int8":
        kwargs["int8"] = True
        kwargs["data"] = str(CALIB_YAML)  # calibration images
    else:
        raise ValueError(precision)

    t0 = time.time()
    out = Path(model.export(**kwargs))
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    dst = ENGINE_DIR / f"yolo26s_sku110k_{precision}.engine"
    if out.resolve() != dst.resolve():
        dst.write_bytes(out.read_bytes())
        out.unlink()
    print(f"[trt] built {precision} engine in {time.time()-t0:.0f}s -> "
          f"{dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precisions", nargs="+", default=["fp16", "int8"],
                        choices=["fp16", "int8"])
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    for p in args.precisions:
        engine = build(p)
        if not args.no_validate:
            validate_engine(engine, f"trt {p}")


if __name__ == "__main__":
    main()
