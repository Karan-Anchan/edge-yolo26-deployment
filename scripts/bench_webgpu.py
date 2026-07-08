"""Controlled WebGPU latency benchmark for the demo model, in a real browser
engine. Loads the page (so onnxruntime-web is available), creates a WebGPU
session, warms up, times many session.run() calls, and reports p50/p95 plus
the actual GPU adapter used (so we know hardware vs software).

Run: python scripts/bench_webgpu.py
"""

import http.server
import json
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "code" / "web_demo"
PORT = 8137


def serve():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(ROOT), **k)
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


BENCH_JS = """
async () => {
  const info = {};
  if (!navigator.gpu) return {error: 'navigator.gpu missing'};
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return {error: 'no WebGPU adapter (software/no GPU)'};
  info.adapter = adapter.info || {};
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  const sess = await ort.InferenceSession.create('./model/yolo26s_sku110k.onnx',
    {executionProviders:['webgpu'], graphOptimizationLevel:'all'});  // webgpu only — no silent wasm fallback
  const N = 640*640*3;
  const data = new Float32Array(N);
  for (let i=0;i<N;i++) data[i] = Math.random();
  const feed = () => ({images: new ort.Tensor('float32', data, [1,3,640,640])});
  for (let i=0;i<12;i++) await sess.run(feed());          // warmup
  const t = [];
  for (let i=0;i<40;i++){ const s=performance.now(); await sess.run(feed()); t.push(performance.now()-s); }
  t.sort((a,b)=>a-b);
  const pct = p => t[Math.min(t.length-1, Math.floor(p*t.length))];
  info.p50 = pct(0.50); info.p95 = pct(0.95); info.fps = 1000/pct(0.50); info.n = t.length;
  return info;
}
"""


def main() -> None:
    httpd = serve()
    with sync_playwright() as p:
        # headed so Chromium can reach the real RTX 5070 WebGPU adapter
        # (headless Windows Chromium exposes no hardware WebGPU adapter)
        browser = p.chromium.launch(headless=False, args=[
            "--enable-unsafe-webgpu", "--enable-features=Vulkan",
        ])
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_selector("#run:not([disabled])", timeout=120_000)
        result = page.evaluate(BENCH_JS)
        browser.close()
    httpd.shutdown()
    if result.get("error"):
        print("[webgpu] NOT MEASURED:", result["error"])
        return
    print("[webgpu] adapter:", json.dumps(result.get("adapter") or {}))
    print(f"[webgpu] p50={result['p50']:.1f} ms  p95={result['p95']:.1f} ms  "
          f"fps={result['fps']:.1f}  (n={result['n']})")


if __name__ == "__main__":
    main()
