"""Attempt a true NVFP4 (4-bit) engine via modelopt's PyTorch quantization.

FP4 is NOT available in modelopt's ONNX quantizer (only int8/fp8 are), so we
quantize the live Ultralytics nn.Module instead:

    best.pt (nn.Module)
      --(mtq.quantize, NVFP4_DEFAULT_CFG, forward-loop over 120 calib imgs)-->
    quantized module --(ultralytics export, keeps the (1,600,6) NMS-free head)-->
    fp4.onnx (QDQ) --(TensorRT strongly-typed build)--> fp4.engine

Bounded attempt: 4 bits = 16 levels. If the detection/regression head hits an
unsupported op on export or accuracy collapses, that is a documented negative
result — the honest answer to "how low can precision go before this model
breaks?" — not a rabbit hole to keep digging.
"""

import os
from pathlib import Path

# modelopt's MX CUDA kernel is JIT-compiled against CUDA 13.0 (nvcc/TensorRT),
# but torch bundles CUDA 12.8 — so the compiled .pyd needs the CUDA 13.0 runtime
# DLLs on Python's DLL search path at load time (PATH alone doesn't cover .pyd
# dependencies on Python 3.8+). Register it before importing torch/modelopt.
_CUDA13_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
if os.path.isdir(_CUDA13_BIN):
    os.add_dll_directory(_CUDA13_BIN)

import numpy as np
import torch
from ultralytics import YOLO

from engine_utils import MAX_DET, build_strongly_typed_engine, validate_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "results" / "training" / "yolo26s" / "weights" / "best.pt"
FP4_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp4.onnx"
ENGINE = REPO_ROOT / "models" / "tensorrt" / "yolo26s_sku110k_fp4.engine"
CALIB_NPY = REPO_ROOT / "models" / "onnx" / "fp4_calib.npy"
IMGSZ = 640


def quantize_model(model: YOLO) -> None:
    """In-place NVFP4 PTQ of the DetectionModel with a calibration forward-loop."""
    import modelopt.torch.quantization as mtq

    net = model.model.eval().cuda()
    calib = np.load(CALIB_NPY)
    print(f"[fp4] NVFP4 PTQ over {len(calib)} calib images...")

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, len(calib), 8):
                batch = torch.from_numpy(calib[i:i + 8]).cuda()
                m(batch)

    mtq.quantize(net, mtq.NVFP4_DEFAULT_CFG, forward_loop)
    print("[fp4] quantization applied")


def export_onnx(model: YOLO) -> Path:
    """Export the quantized model to ONNX, keeping the NMS-free (1,600,6) head."""
    out = Path(model.export(format="onnx", imgsz=IMGSZ, opset=19, dynamic=False,
                            simplify=False, half=False, batch=1,
                            max_det=MAX_DET, device=0))
    if out.resolve() != FP4_ONNX.resolve():
        FP4_ONNX.parent.mkdir(parents=True, exist_ok=True)
        FP4_ONNX.write_bytes(out.read_bytes())
        out.unlink()
    print(f"[fp4] exported {FP4_ONNX.name} ({FP4_ONNX.stat().st_size/1e6:.1f} MB)")
    return FP4_ONNX


if __name__ == "__main__":
    if not CALIB_NPY.exists():
        raise SystemExit(f"calibration tensor missing ({CALIB_NPY})")
    model = YOLO(str(WEIGHTS))
    quantize_model(model)
    export_onnx(model)
    build_strongly_typed_engine(FP4_ONNX, ENGINE, "fp4")
    validate_engine(ENGINE, "fp4")
    print("[fp4] done")
