<div align="center">

# EDGE // YOLO26

**One trained detector, deployed to GPU, CPU, and the browser. Each export path is measured separately.**

![runtimes](https://img.shields.io/badge/runtimes-GPU_%7C_CPU_%7C_Browser-ffab2e?style=flat-square&labelColor=0b0c0a)
![mAP](https://img.shields.io/badge/mAP_50--95-0.572-a3e05f?style=flat-square&labelColor=0b0c0a)
![speedup](https://img.shields.io/badge/GPU-560_FPS_·_7.2x-ffab2e?style=flat-square&labelColor=0b0c0a)
![license](https://img.shields.io/badge/license-MIT-8b9080?style=flat-square&labelColor=0b0c0a)

![Client-side WebGPU detection demo](assets/demo.gif)

*Dense retail-shelf detection running **entirely in the browser** with WebGPU. Images stay on the device.*

</div>

---

This project fine-tunes an NMS-free **YOLO26-s** detector on dense retail shelves
(**SKU-110K**, roughly 150 objects per image). The same ONNX graph is deployed to three runtimes
across four precisions, then evaluated for accuracy, latency, and energy use.

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
| Browser · WebGPU | 0.5716 | — | 22.9 ms | 44 | — | — |

**What the measurements showed**

- FP16 has the best latency per watt on the GPU at 9.3 FPS/W, with almost no accuracy loss.
  FP8 is fastest at 560 FPS. On this Blackwell GPU, INT8 is slower, draws more power, and loses
  more accuracy than either option. The power measurements made the usual INT8 default a poor fit
  for this hardware.
- INT8 does not behave the same in every runtime. It costs 5.65% mAP in TensorRT and 0.72% in
  ONNX Runtime, an approximately eightfold difference. The CPU path uses per-channel
  quantization and leaves all 94 detection-head nodes in FP32. TensorRT's calibrator quantizes
  the head as well. Applying the CPU strategy to TensorRT is a next step, not a completed result.
- The NMS-free graph emits `[1,600,6]`, so the browser path does not need a separate NMS
  implementation. It was verified in WebGPU at roughly 140 objects per frame.

## Demo

The model runs client-side with WebGPU, and the image never leaves the device.

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

The controlled variable is precision. Model weights, `imgsz=640`, `max_det=600` (the default
300 silently caps recall on dense shelves), and preprocessing are identical across every rung,
so accuracy deltas are attributable to precision alone.

| | |
| :--- | :--- |
| Calibration | one fixed set of 120 SKU-110K train images, shared by every PTQ path |
| Accuracy | Ultralytics `val` on the held-out split (588 images, ~91k boxes), mAP@50-95 |
| Latency | MLPerf single-stream, warmup discarded, p50/p95 over 100–200 timed runs on a fixed 640² input; GPU power sampled via NVML during the timed loop |
| Hardware | RTX 5070 (Blackwell, 12 GB) · Ryzen 7 7700 · Windows 11 |
| Pinned | torch 2.11+cu128 · ultralytics 8.4.87 · TensorRT 11.1 · modelopt 0.45 |

Under INT8, mAP@50 barely moved while mAP@50-95 fell. This points to less precise boxes rather
than missed detections. The FP8 and per-channel INT8 results support quantization granularity,
not bit count alone, as the cause.

</details>

<details>
<summary><b>Limitations & future work</b></summary>

- Single training run, so the accuracy deltas are point estimates with no variance bars.
- Power is measured for GPU only. WebGPU *latency* is now benchmarked on the RTX 5070 (~23 ms
  p50, ~44 FPS in-browser) but is inherently client-GPU-dependent; CPU RAPL and browser energy
  aren't captured. The demo's on-screen latency is a live per-visitor readout.
- FP4 (NVFP4) is blocked by the local toolchain. Modelopt's 4-bit CUDA kernel will not load
  with the CUDA 12.8 (PyTorch) and CUDA 13.0 (nvcc) split. Testing it requires a host with
  aligned CUDA versions.
- There is no NMS-vs-NMS-free baseline yet (YOLOv8), so the density-robustness hypothesis is
  untested.
- Still to do: rescue TensorRT INT8 with per-channel + FP16 head, add CPU/WebGPU power, and run
  multiple seeds.

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

Released under the [MIT License](LICENSE).
