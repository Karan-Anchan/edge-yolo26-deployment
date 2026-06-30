"""Measure ONNX Runtime CPU latency for FP32 vs INT8, MLPerf single-stream style.

Reports p50/p95 latency and FPS on the Ryzen 7 7700 — the CPU rung of the
latency-per-watt study. Accuracy (mAP) is measured separately via ultralytics
val; this script is purely about speed on the same fixed input.

Method (copied from MLPerf's single-stream discipline):
  * one fixed 640x640 input (isolate the model, not the data pipeline)
  * warmup runs discarded (let the CPU clock/caches settle)
  * report median (p50) and tail (p95), not just the mean — tail latency is what
    a real-time app actually feels
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "fp32": REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_fp32.onnx",
    "int8": REPO_ROOT / "models" / "onnx" / "yolo26s_sku110k_int8_ort.onnx",
}
INPUT_NAME = "images"


def bench(path: Path, warmup: int, iters: int, threads: int) -> dict:
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), opts,
                                providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {INPUT_NAME: x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {INPUT_NAME: x})
        times.append((time.perf_counter() - t0) * 1000)  # ms
    times = np.array(times)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "fps": 1000.0 / float(np.percentile(times, 50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0, help="0 = ORT default")
    args = parser.parse_args()

    print(f"CPU single-stream latency (warmup={args.warmup}, iters={args.iters}, "
          f"threads={args.threads or 'default'})")
    print(f"{'model':<8}{'p50 (ms)':>12}{'p95 (ms)':>12}{'FPS':>10}")
    for name, path in MODELS.items():
        if not path.exists():
            print(f"{name:<8}  (missing — build it first)")
            continue
        r = bench(path, args.warmup, args.iters, args.threads)
        print(f"{name:<8}{r['p50']:>12.1f}{r['p95']:>12.1f}{r['fps']:>10.1f}")


if __name__ == "__main__":
    main()
