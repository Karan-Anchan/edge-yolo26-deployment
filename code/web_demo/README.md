# WebGPU Browser Demo

The trained YOLO26 model runs **entirely in the browser** through `onnxruntime-web`.
It uses **WebGPU** when available and falls back to WASM in other browsers. Images never
leave the device.

## Files

| Path | Tracked? | Notes |
| :--- | :--- | :--- |
| `index.html` | ✅ | self-contained page (inline CSS/JS) |
| `samples/*.jpg` | ✅ | 3 lightweight sample shelf images |
| `model/yolo26s_sku110k.onnx` | ❌ (gitignored) | the FP32 ONNX; regenerate it using the steps below |

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

Choose a shelf image or drop in your own, then select **Run detection**. **▶ LIVE VIDEO** runs
continuous detection on a supermarket-aisle clip at roughly 50 FPS on WebGPU. The runtime badge
shows whether the page is using WebGPU or the WASM fallback.

`media/walk.mp4` is a muted, 720p, web-optimized clip. Like the model, it is gitignored. Put your
own landscape footage at that path to replace it.

## Verify (headless)

`python scripts/test_web_demo.py` drives the page in headless Chromium, runs a real
detection, asserts a plausible object count, and writes `results/web_demo_shot.png`.

> **Note on browser latency:** the displayed number is measured on the visitor's device and
> depends on that device's GPU. It is separate from the controlled benchmark results.
