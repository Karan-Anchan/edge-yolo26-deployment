# WebGPU Browser Demo

Client-side retail-shelf detection — the trained YOLO26 model runs **entirely in the
browser** via `onnxruntime-web` on the **WebGPU** execution provider (WASM fallback on
browsers without WebGPU). No image ever leaves the device.

## Files

| Path | Tracked? | Notes |
| :--- | :--- | :--- |
| `index.html` | ✅ | self-contained page (inline CSS/JS) |
| `samples/*.jpg` | ✅ | 3 lightweight sample shelf images |
| `model/yolo26s_sku110k.onnx` | ❌ (gitignored) | the FP32 ONNX — regenerate, see below |

## Regenerate the model

The model file is gitignored (37 MB). Recreate it from the trained checkpoint:

```bash
python code/export/export_onnx.py                       # -> models/onnx/yolo26s_sku110k_fp32.onnx
cp models/onnx/yolo26s_sku110k_fp32.onnx code/web_demo/model/yolo26s_sku110k.onnx
```

## Run it

WebGPU needs a real origin (not `file://`), so serve the folder:

```bash
python -m http.server 8123 --directory code/web_demo
# open http://127.0.0.1:8123/  in Chrome/Edge (WebGPU) 
```

Pick a shelf (or drop your own image) → **Run detection**, or hit **▶ LIVE VIDEO** for
continuous frame-by-frame detection on a real supermarket-aisle clip (~50 fps on WebGPU). The
runtime badge shows whether you got **WebGPU** or the **WASM** fallback.

`media/walk.mp4` is a muted, 720p, web-optimised clip; like the model it is gitignored — swap in
your own landscape footage at that path.

## Verify (headless)

`python scripts/test_web_demo.py` drives the page in headless Chromium, runs a real
detection, asserts a plausible object count, and writes `results/web_demo_shot.png`.

> **Note on browser latency:** the number shown is a live, per-client readout — it depends
> entirely on the visitor's GPU. It is *not* one of the project's benchmark figures (those are
> measured under controlled conditions in the benchmarking phase).
