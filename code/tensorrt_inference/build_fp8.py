"""Build an FP8 (E4M3) TensorRT engine — the rung between INT8 and FP4.

FP8 IS supported by modelopt's ONNX quantizer (unlike FP4), so this reuses our
canonical FP32 ONNX directly:

    fp32.onnx --(modelopt fp8 PTQ, 120 calib imgs)--> fp8.onnx (QDQ)
              --(TensorRT strongly-typed build)------> fp8.engine

Why FP8 is interesting here: it keeps a floating-point *exponent* (4 bits) plus
3 mantissa bits, so it represents a wide dynamic range at low bit-count — often
recovering much of the localization accuracy that INT8's uniform 256 levels
lose on this dense task, while still being far smaller/faster than FP16. It is
the natural "is the INT8 accuracy cliff a quantization-granularity problem?"
control experiment.
"""

import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from ultralytics import YOLO

from engine_utils import write_engine_with_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
FP32_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp32.onnx"
FP8_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp8.onnx"
ENGINE = REPO_ROOT / "models" / "tensorrt" / "yolo26s_sku110k_fp8.engine"
CALIB_NPY = REPO_ROOT / "models" / "onnx" / "fp4_calib.npy"  # reuse FP4's calib tensor
IMGSZ = 640
MAX_DET = 600
FP32_BASELINE = 0.5716


def quantize_fp8() -> None:
    from modelopt.onnx.quantization import quantize
    print("[fp8] modelopt FP8 PTQ (QDQ insert + calibrate)...")
    quantize(
        onnx_path=str(FP32_ONNX),
        quantize_mode="fp8",
        calibration_data={"images": np.load(CALIB_NPY)},
        output_path=str(FP8_ONNX),
    )
    print(f"[fp8] wrote {FP8_ONNX.name} ({FP8_ONNX.stat().st_size/1e6:.1f} MB)")


def build_engine() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(FP8_ONNX.read_bytes()):
        for i in range(parser.num_errors):
            print("[fp8] parser error:", parser.get_error(i))
        raise SystemExit("FP8 ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("FP8 engine build returned None")
    write_engine_with_metadata(serialized, ENGINE)
    print(f"[fp8] built engine in {time.time()-t0:.0f}s -> {ENGINE.name} "
          f"({ENGINE.stat().st_size/1e6:.1f} MB)")


def validate() -> None:
    r = YOLO(str(ENGINE)).val(data="SKU-110K.yaml", imgsz=IMGSZ, max_det=MAX_DET)
    drop = FP32_BASELINE - r.box.map
    pct = 100 * drop / FP32_BASELINE
    verdict = "PASS" if pct <= 2.0 else "FAIL (>2% budget)"
    print(f"[fp8] fp8: mAP50-95={r.box.map:.4f} mAP50={r.box.map50:.4f} "
          f"| drop {drop:+.4f} ({pct:+.2f}%) -> {verdict}")


if __name__ == "__main__":
    if not CALIB_NPY.exists():
        raise SystemExit(f"calibration tensor missing ({CALIB_NPY}); run build_fp4.py's "
                         "prep_calibration first")
    quantize_fp8()
    build_engine()
    validate()
    print("[fp8] done")
