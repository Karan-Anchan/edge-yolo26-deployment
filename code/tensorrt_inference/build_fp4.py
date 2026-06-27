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
import time
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
import tensorrt as trt
from ultralytics import YOLO

from engine_utils import write_engine_with_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "results" / "training" / "yolo26s" / "weights" / "best.pt"
FP4_ONNX = REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp4.onnx"
ENGINE = REPO_ROOT / "models" / "tensorrt" / "yolo26s_sku110k_fp4.engine"
CALIB_NPY = REPO_ROOT / "models" / "onnx" / "fp4_calib.npy"
IMGSZ = 640
MAX_DET = 600
FP32_BASELINE = 0.5716


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


def build_engine() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(FP4_ONNX.read_bytes()):
        for i in range(parser.num_errors):
            print("[fp4] parser error:", parser.get_error(i))
        raise SystemExit("FP4 ONNX parse failed (likely an unsupported QDQ op)")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("FP4 engine build returned None")
    write_engine_with_metadata(serialized, ENGINE)
    print(f"[fp4] built engine in {time.time()-t0:.0f}s -> {ENGINE.name} "
          f"({ENGINE.stat().st_size/1e6:.1f} MB)")


def validate() -> None:
    r = YOLO(str(ENGINE)).val(data="SKU-110K.yaml", imgsz=IMGSZ, max_det=MAX_DET)
    drop = FP32_BASELINE - r.box.map
    pct = 100 * drop / FP32_BASELINE
    verdict = "PASS" if pct <= 2.0 else "FAIL (>2% budget)"
    print(f"[fp4] fp4: mAP50-95={r.box.map:.4f} mAP50={r.box.map50:.4f} "
          f"| drop {drop:+.4f} ({pct:+.2f}%) -> {verdict}")


if __name__ == "__main__":
    if not CALIB_NPY.exists():
        raise SystemExit(f"calibration tensor missing ({CALIB_NPY})")
    model = YOLO(str(WEIGHTS))
    quantize_model(model)
    export_onnx(model)
    build_engine()
    validate()
    print("[fp4] done")
