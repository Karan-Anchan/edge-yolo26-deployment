"""Capture a SMOOTH animated GIF of the WebGPU demo for the README:
input shelf -> analysing (scanline sweep) -> detections + live count-up.

Smoothness strategy: use Chrome DevTools **screencast** (Page.startScreencast),
which streams frames at the browser's real paint rate (far more frames than
element screenshots can manage). Each frame carries a timestamp, so GIF frame
durations mirror real time. Frames are cropped to the .console region.

Run: python scripts/make_demo_gif.py
"""

import base64
import http.server
import io
import socketserver
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "code" / "web_demo"
ASSETS = Path(__file__).resolve().parents[1] / "assets"
PORT = 8125
GIF_W = 860


def serve():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(ROOT), **k)
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    httpd = serve()
    frames = []  # (PIL.Image, timestamp)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-webgpu",
                                          "--enable-features=Vulkan"])
        page = browser.new_page(viewport={"width": 1180, "height": 920},
                                device_scale_factor=1)
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_selector("#run:not([disabled])", timeout=120_000)
        page.wait_for_timeout(700)

        box = page.locator(".console").bounding_box()
        crop = (int(box["x"]), int(box["y"]),
                int(box["x"] + box["width"]), int(box["y"] + box["height"]))

        client = page.context.new_cdp_session(page)

        def on_frame(ev):
            img = Image.open(io.BytesIO(base64.b64decode(ev["data"]))).convert("RGB")
            frames.append((img.crop(crop), ev["metadata"]["timestamp"]))
            try:
                client.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception:
                pass

        client.on("Page.screencastFrame", on_frame)
        client.send("Page.startScreencast",
                    {"format": "jpeg", "quality": 90, "everyNthFrame": 1,
                     "maxWidth": 1180, "maxHeight": 920})

        page.wait_for_timeout(700)        # idle frames
        page.click("#run")
        page.wait_for_function(
            "document.querySelector('#toast').textContent.includes('detected')",
            timeout=120_000)
        page.wait_for_timeout(1100)       # count-up + hold on result
        client.send("Page.stopScreencast")
        page.wait_for_timeout(150)
        browser.close()
    httpd.shutdown()

    if not frames:
        raise SystemExit("no screencast frames captured")

    # durations from real inter-frame timestamps (ms), clamped
    raw_dur = []
    for i in range(len(frames)):
        if i + 1 < len(frames):
            dt = (frames[i + 1][1] - frames[i][1]) * 1000
            raw_dur.append(max(30, min(250, dt)))
        else:
            raw_dur.append(1700)          # hold final frame
    raw_dur[0] = max(raw_dur[0], 900)     # hold input frame

    # collapse only TRULY-static consecutive frames (idle / held result). Use
    # the fraction of *meaningfully changed* pixels, not mean diff — a thin
    # scanline or a tiny count digit barely moves the global mean but is real
    # motion we must keep.
    import numpy as np
    def moved(a, b):
        d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=2)
        return float((d > 24).mean())            # fraction of changed pixels
    keep_imgs, keep_dur = [frames[0][0]], [raw_dur[0]]
    for i in range(1, len(frames)):
        if moved(frames[i][0], frames[i - 1][0]) < 0.0012:   # static → merge
            keep_dur[-1] += raw_dur[i]
        else:
            keep_imgs.append(frames[i][0]); keep_dur.append(raw_dur[i])
    keep_dur = [int(d) for d in keep_dur]

    out_imgs = []
    for img in keep_imgs:
        h = round(img.height * GIF_W / img.width)
        out_imgs.append(img.resize((GIF_W, h), Image.LANCZOS)
                        .quantize(colors=96, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))

    out = ASSETS / "demo.gif"
    out_imgs[0].save(out, save_all=True, append_images=out_imgs[1:],
                     duration=keep_dur, loop=0, optimize=True)
    durations = keep_dur
    total = sum(durations)
    print(f"[gif] wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(out_imgs)} frames, "
          f"{total/1000:.1f}s loop, ~{1000*len(out_imgs)/total:.0f} fps)")


if __name__ == "__main__":
    main()
