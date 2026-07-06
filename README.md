<div align="center">

# EDGE // YOLO26

**One trained detector → GPU · CPU · Browser — every export path measured, not assumed.**

![runtimes](https://img.shields.io/badge/runtimes-GPU_%7C_CPU_%7C_Browser-ffab2e?style=flat-square&labelColor=0b0c0a)
![mAP](https://img.shields.io/badge/mAP_50--95-0.572-a3e05f?style=flat-square&labelColor=0b0c0a)
![speedup](https://img.shields.io/badge/GPU-560_FPS_·_7.2x-ffab2e?style=flat-square&labelColor=0b0c0a)
![license](https://img.shields.io/badge/license-MIT-8b9080?style=flat-square&labelColor=0b0c0a)

![Client-side WebGPU detection demo](assets/demo.gif)

*Dense retail-shelf detection running **100% in the browser** on WebGPU — no upload, no server.*

</div>

---

An edge-deployment study: fine-tune an NMS-free detector (**YOLO26-s**) on dense retail shelves
(**SKU-110K**, ~150 objects/image), export **one** ONNX graph, and ship it to three runtimes
across four precisions — measuring the accuracy, latency, and energy cost of each.

## Results

<div align="center">

![Accuracy cost of quantization](assets/benchmark_accuracy.png)

</div>

Baseline **FP32 mAP@50-95 = 0.572**; budget = ≤ 2% drop. Measured on RTX 5070 (GPU) and Ryzen 7
7700 (CPU), MLPerf single-stream + NVML power.

| Runtime · Precision | mAP@50-95 | Δ | Latency p50 | FPS | Power | FPS/W |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU · FP32 *(PyTorch ref)* | 0.572 | — | 12.9 ms | 77 | 66 W | 1.2 |
| **GPU · FP16** | 0.5713 | −0.06% | 1.9 ms | 536 | **58 W** | **9.3** |
| **GPU · FP8** | 0.5652 | −1.12% | **1.8 ms** | **560** | 78 W | 7.2 |
| GPU · INT8 | 0.5393 | −5.65% | 2.7 ms | 374 | 88 W | 4.2 |
| CPU · FP32 | 0.5716 | — | 52.5 ms | 19 | — | — |
| **CPU · INT8** | 0.5675 | −0.72% | 37.7 ms | 27 | — | — |

**Findings**

- **FP16 wins latency-per-watt on GPU** (9.3 FPS/W, near-lossless); FP8 is fastest (560 FPS).
  **INT8 is dominated on Blackwell** — slower *and* hungrier than FP16/FP8, with the worst
  accuracy. Measuring power, not assuming it, flipped the "always quantize to INT8" default.
- **"INT8" isn't one thing:** −5.65% on TensorRT vs −0.72% on ONNX Runtime — an ~8× gap closed by
  per-channel quantization + keeping the detection head in FP32.
- **NMS-free → clean browser deploy:** the graph emits `[1,600,6]` with no NMS op to port;
  verified in-browser on WebGPU (~140 objects/frame).

## Demo

The model runs entirely client-side on WebGPU — the image never leaves the device.

```bash
python -m http.server 8123 --directory code/web_demo   # → http://127.0.0.1:8123
```

## Reproduce

```powershell
conda create -n yolo26 python=3.11 -y; conda activate yolo26
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics onnx onnxslim onnxruntime nvidia-modelopt tensorrt

python code/training/train.py                       # fine-tune (add --smoke for a 100-img test)
python code/export/export_onnx.py --validate        # → FP32 ONNX (parity-checked)
python code/tensorrt_inference/build_engines.py     # FP16 + INT8   │ build_fp8.py for FP8
python code/tensorrt_inference/bench_gpu.py          # GPU latency + power
python code/onnxruntime_inference/quantize_int8.py  # CPU INT8      │ bench_cpu.py for latency
```

<details>
<summary><b>Methodology & setup</b></summary>

- **Controlled variable = precision.** Model weights, `imgsz=640`, `max_det=600` (the default
  300 silently caps recall on dense shelves), and preprocessing are identical across every rung,
  so accuracy deltas are attributable to precision alone.
- **Calibration:** one fixed set of 120 SKU-110K train images, shared by every PTQ path.
- **Accuracy:** Ultralytics `val` on the held-out split (588 images, ~91k boxes), mAP@50-95.
- **Latency:** MLPerf single-stream — warmup discarded, p50/p95 over 100–200 timed runs on a fixed
  640² input; GPU power sampled via NVML during the timed loop.
- **Hardware:** RTX 5070 (Blackwell, 12 GB) · Ryzen 7 7700 · Windows 11.
- **Pinned:** torch 2.11+cu128 · ultralytics 8.4.87 · TensorRT 11.1 · modelopt 0.45.
- **Why the mechanism is clear:** under INT8, mAP@50 barely moved but mAP@50-95 fell — boxes got
  *coarser*, not *missed*. FP8 and per-channel INT8 (two independent controls) confirm the cause is
  quantization **granularity**, not bit-count.

</details>

<details>
<summary><b>Limitations & future work</b></summary>

- **Single training run** — accuracy deltas are point estimates (no variance bars).
- **CPU/WebGPU power not measured** — the GPU per-watt story is complete; CPU RAPL and a controlled
  WebGPU measurement are pending. The demo's on-screen latency is a live per-visitor readout, not a
  benchmark.
- **FP4 (NVFP4) is toolchain-blocked** on this box — modelopt's 4-bit CUDA kernel won't load under
  the CUDA-12.8 (torch) / 13.0 (nvcc) split. A reproducible software-supply-chain gap, not a
  modeling failure; unblocking needs a CUDA-aligned host.
- **No NMS-vs-NMS-free baseline yet** (YOLOv8) — the density-robustness hypothesis is untested.
- **Next:** rescue TensorRT INT8 with per-channel + FP16 head; add CPU/WebGPU power; multi-seed runs.

</details>

## Layout

```
code/{training,export,tensorrt_inference,onnxruntime_inference,web_demo}/  ·  scripts/  ·  assets/
data/ · models/ · results/   # gitignored (regenerable)
```

## References

[YOLO26](https://arxiv.org/html/2509.25164v2) ·
[RT-DETR](https://arxiv.org/abs/2304.08069) ·
[Quantization survey](https://arxiv.org/abs/2103.13630) ·
[MLPerf Inference](https://arxiv.org/abs/1911.02549) ·
[SKU-110K](https://arxiv.org/abs/1904.00853)

MIT — see [`LICENSE`](LICENSE).
