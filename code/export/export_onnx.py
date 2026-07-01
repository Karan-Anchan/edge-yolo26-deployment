"""Export the trained YOLO26 checkpoint to a canonical FP32 ONNX graph.

This ONNX file is the *single common intermediate* consumed by all three
deployment runtimes:

    best.pt --(this script)--> best.onnx --> TensorRT engine   (GPU, FP16/INT8)
                                        \--> ONNX Runtime       (CPU, INT8 QDQ)
                                        \--> onnxruntime-web     (browser, WebGPU)

Design choices (see project_notes for the full reasoning):

* FP32 master. FP16/INT8 are produced *per runtime* downstream (TensorRT builds
  reduced-precision engines from this graph; ORT quantizes it). Exporting FP16
  here would throw away the reference the whole study measures drops against.
* Static shape, batch 1 (imgsz 640). The benchmark measures single-frame
  latency, and a fixed graph is the fastest/most-portable path on all three
  runtimes (WebGPU especially dislikes dynamic shapes).
* opset 19 — new enough for modern ops, old enough for onnxruntime-web's WebGPU
  execution provider.
* simplify=True (onnxslim) — folds constants; smaller graph helps every runtime.
* max_det=600 — MUST match training. YOLO26 is NMS-free/end-to-end, so the top-k
  detection cap is baked into the ONNX graph. SKU-110K images hold ~150 (up to
  ~660) boxes; leaving the default 300 would silently clip real detections in
  every downstream runtime, poisoning recall before quantization is even tested.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "results" / "training" / "yolo26s" / "weights" / "best.pt"
EXPORT_DIR = REPO_ROOT / "models" / "onnx"

MAX_DET = 600  # keep in lockstep with code/training/train.py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--validate", action="store_true",
                        help="run mAP val on the exported ONNX to prove export parity")
    parser.add_argument("--data", default="SKU-110K.yaml")
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,     # static shape → fastest, most portable
        simplify=True,     # onnxslim constant folding
        half=False,        # FP32 master
        batch=1,
        max_det=MAX_DET,
        device="cpu",      # export on CPU for a clean, portable graph
    )

    # ultralytics writes next to the .pt; move it into models/onnx/
    src = Path(onnx_path)
    dst = EXPORT_DIR / "yolo26s_sku110k_fp32.onnx"
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
        src.unlink()
    print(f"[export] wrote {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")

    if args.validate:
        print("[export] validating ONNX mAP (should match the .pt FP32 baseline)...")
        metrics = YOLO(str(dst)).val(data=args.data, imgsz=args.imgsz, max_det=MAX_DET)
        print(f"[export] ONNX mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
