"""Shared helpers for the TensorRT precision ladder.

Two jobs live here so the three build scripts (FP16/INT8, FP8, FP4) don't each
carry their own copy:

1. `write_engine_with_metadata` — engines we build ourselves from a
   modelopt-quantized ONNX (FP8, FP4) lack the metadata header Ultralytics
   normally writes, so `YOLO(engine).val()` wouldn't know the task/imgsz/names.
   We prepend the same header, letting every low-bit engine be validated through
   the *identical* `YOLO().val()` path as FP16/INT8 — apples-to-apples.

       Header format (from ultralytics.nn.backends.tensorrt):
       <4-byte little-endian metadata length><utf-8 JSON metadata><serialized engine>

2. `build_strongly_typed_engine` / `validate_engine` — the strongly-typed engine
   build (FP8, FP4) and the mAP-drop report (all rungs) were copy-pasted across
   the build scripts; they live here once and take the per-rung differences (the
   ONNX path, the log tag) as arguments.
"""

import json
import time
from pathlib import Path

IMGSZ = 640
MAX_DET = 600
FP32_BASELINE = 0.5716  # ONNX FP32 mAP50-95; every engine is scored as a drop from this


def write_engine_with_metadata(serialized: bytes, dst: Path,
                               task: str = "detect",
                               names: dict | None = None,
                               batch: int = 1) -> None:
    """Prepend the Ultralytics metadata header and write the engine."""
    meta = {
        "description": "YOLO26s SKU-110K low-bit engine",
        "author": "edge_yolo26_deployment",
        "task": task,
        "batch": batch,
        "imgsz": [IMGSZ, IMGSZ],
        "stride": 32,
        "names": names or {0: "object"},
    }
    blob = json.dumps(meta).encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(len(blob).to_bytes(4, byteorder="little"))
        f.write(blob)
        f.write(serialized)


def build_strongly_typed_engine(onnx_path: Path, dst: Path, tag: str,
                                workspace_gb: int = 8) -> None:
    """Parse a QDQ ONNX and build a strongly-typed TensorRT engine (FP8/FP4 path).

    Strongly-typed lets TensorRT honour the exact precisions baked into the QDQ
    graph rather than picking its own, which is what makes the low-bit result the
    one the ONNX actually encodes.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        for i in range(parser.num_errors):
            print(f"[{tag}] parser error:", parser.get_error(i))
        raise SystemExit(f"{tag} ONNX parse failed (possibly an unsupported QDQ op)")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit(f"{tag} engine build returned None")
    write_engine_with_metadata(serialized, dst)
    print(f"[{tag}] built engine in {time.time()-t0:.0f}s -> {dst.name} "
          f"({dst.stat().st_size/1e6:.1f} MB)")


def validate_engine(engine: Path, tag: str, data: str = "SKU-110K.yaml",
                    baseline: float = FP32_BASELINE) -> float:
    """Val an engine's mAP and report the drop from the FP32 baseline (≤2% = PASS)."""
    from ultralytics import YOLO

    r = YOLO(str(engine)).val(data=data, imgsz=IMGSZ, max_det=MAX_DET)
    drop = baseline - r.box.map
    pct = 100 * drop / baseline
    verdict = "PASS" if pct <= 2.0 else "FAIL (>2% budget)"
    print(f"[{tag}] mAP50-95={r.box.map:.4f} mAP50={r.box.map50:.4f} "
          f"| drop {drop:+.4f} ({pct:+.2f}%) -> {verdict}")
    return r.box.map
