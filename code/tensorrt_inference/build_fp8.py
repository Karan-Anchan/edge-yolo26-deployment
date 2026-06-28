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

from pathlib import Path

import numpy as np

from engine_utils import build_strongly_typed_engine, validate_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
FP32_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp32.onnx"
FP8_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp8.onnx"
ENGINE = REPO_ROOT / "models" / "tensorrt" / "yolo26s_sku110k_fp8.engine"
CALIB_NPY = REPO_ROOT / "models" / "onnx" / "fp4_calib.npy"  # reuse FP4's calib tensor


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


if __name__ == "__main__":
    if not CALIB_NPY.exists():
        raise SystemExit(f"calibration tensor missing ({CALIB_NPY}); run build_fp4.py's "
                         "prep_calibration first")
    quantize_fp8()
    build_strongly_typed_engine(FP8_ONNX, ENGINE, "fp8")
    validate_engine(ENGINE, "fp8")
    print("[fp8] done")
