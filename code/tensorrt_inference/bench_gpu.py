"""GPU latency + power for the TensorRT engines vs a PyTorch-FP32 reference,
completing the latency-per-watt table (RTX 5070, Blackwell).

Method (MLPerf single-stream + NVML power):
  * fixed 640x640 input; warmup discarded; per-iter time with cuda.synchronize
  * report p50 / p95 latency and FPS
  * a background thread samples board power (NVML) during the timed loop
  * energy-per-frame = mean_power x p50_latency; efficiency = FPS / mean_power

The PyTorch-FP32 row is the reference for the project's "TensorRT-INT8 >= 2x
PyTorch-FP32" success criterion.
"""

import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics import YOLO

import pynvml

REPO = Path(__file__).resolve().parents[2]
ENGINES = {
    "FP16": REPO / "models" / "tensorrt" / "yolo26s_sku110k_fp16.engine",
    "FP8":  REPO / "models" / "tensorrt" / "yolo26s_sku110k_fp8.engine",
    "INT8": REPO / "models" / "tensorrt" / "yolo26s_sku110k_int8.engine",
}
PT = REPO / "results" / "training" / "yolo26s" / "weights" / "best.pt"
WARMUP, ITERS = 30, 200
DEV = torch.device("cuda:0")


class PowerSampler(threading.Thread):
    """Poll NVML board power (W) until stopped."""
    def __init__(self, handle):
        super().__init__(daemon=True)
        self.handle, self.samples, self._run = handle, [], True

    def run(self):
        while self._run:
            self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            time.sleep(0.004)

    def stop(self):
        self._run = False
        self.join()


def time_forward(forward, handle) -> dict:
    x = torch.rand(1, 3, 640, 640, device=DEV)
    for _ in range(WARMUP):
        forward(x)
    torch.cuda.synchronize()
    sampler = PowerSampler(handle); sampler.start()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        forward(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    sampler.stop()
    p50 = statistics.median(times)
    p95 = float(np.percentile(times, 95))
    pw = statistics.mean(sampler.samples) if sampler.samples else float("nan")
    return {"p50": p50, "p95": p95, "fps": 1000 / p50,
            "watt": pw, "mj": pw * p50, "fps_per_w": (1000 / p50) / pw}


def main() -> None:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    rows = {}
    # PyTorch FP32 reference
    net = YOLO(str(PT)).model.to(DEV).eval().float()
    with torch.no_grad():
        rows["FP32 (PyTorch)"] = time_forward(lambda x: net(x), handle)

    # TensorRT engines
    for name, path in ENGINES.items():
        if not path.exists():
            print(f"[gpu] skip {name} (missing)"); continue
        backend = AutoBackend(str(path), device=DEV, fp16=(name == "FP16"))
        with torch.no_grad():
            rows[f"{name} (TensorRT)"] = time_forward(lambda x: backend(x), handle)

    ref = rows["FP32 (PyTorch)"]["p50"]
    print(f"\n{'config':<18}{'p50 ms':>9}{'p95 ms':>9}{'FPS':>8}"
          f"{'Watt':>8}{'mJ/frame':>10}{'FPS/W':>8}{'speedup':>9}")
    for name, r in rows.items():
        print(f"{name:<18}{r['p50']:>9.2f}{r['p95']:>9.2f}{r['fps']:>8.0f}"
              f"{r['watt']:>8.1f}{r['mj']:>10.1f}{r['fps_per_w']:>8.2f}"
              f"{ref / r['p50']:>8.2f}x")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
