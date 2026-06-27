"""Shared helpers for engines we build ourselves (FP8, FP4) via the raw
TensorRT API rather than Ultralytics.

Ultralytics-built engines carry a metadata header so `YOLO(engine).val()`
knows the task/imgsz/class names. Engines we build from a modelopt-quantized
ONNX lack it, so we prepend the same header — that lets our low-bit engines be
validated through the *identical* `YOLO().val()` path as FP16/INT8, keeping the
whole precision ladder apples-to-apples.

Header format (from ultralytics.nn.backends.tensorrt):
    <4-byte little-endian metadata length><utf-8 JSON metadata><serialized engine>
"""

import json
from pathlib import Path

IMGSZ = 640


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
