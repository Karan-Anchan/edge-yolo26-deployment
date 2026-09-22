# Research Showcase + WebGPU Browser Demo

The page combines the deployment study's evidence, method, engineering decisions,
limitations, and a permanently visible working demo. The trained YOLO26 model runs
in the browser through `onnxruntime-web`: WebGPU is tried first, with a WASM CPU
fallback. Images never leave the device.

## Files

| Path | Tracked? | Notes |
| :--- | :--- | :--- |
| `index.html` | ✅ | self-contained page (inline CSS/JS) |
| `samples/*.jpg` | ✅ | 3 lightweight sample shelf images |
| `model/yolo26s_sku110k.onnx` | ✅ | FP32 ONNX used by the published Pages demo |
| `media/walk.mp4` | ✅ | short sample video used by continuous detection |

The repository-wide ignore rules cover ONNX and video artifacts, but these two
deployment assets are already force-tracked so GitHub Pages receives a complete,
working artifact.

## Regenerate the model

Recreate the browser model from the trained checkpoint:

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

Choose a shelf image, the sample video, or a local file. The runtime badge reports
WebGPU only after a WebGPU session is created; otherwise it reports the WASM CPU
fallback. Live telemetry times `session.run()` only. It excludes preprocessing,
box decoding, and canvas drawing, so the reciprocal FPS value is not an
end-to-end video-throughput claim.

## Verify (headless)

`python scripts/test_web_demo.py` drives the page in headless Chromium, runs a real
detection, asserts a plausible object count, and writes `results/web_demo_shot.png`.

> **Note on browser latency:** the displayed number is measured on the visitor's
> device. It is separate from the controlled RTX 5070 browser benchmark.
