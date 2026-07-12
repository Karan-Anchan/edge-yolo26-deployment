"""Capture a smooth GIF of the demo's LIVE VIDEO detection mode (boxes tracking
products as the camera walks a supermarket aisle) for the portfolio card.

Runs headed on real WebGPU so detection is ~50 fps (headless WASM would be a
choppy ~2 fps). Uses CDP screencast, crops to the .screen viewport, and sets
GIF frame durations from real timestamps. Output: assets/demo_video.gif
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
PORT = 8188
GIF_W = 560
CAPTURE_S = 5.0


def serve():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(ROOT), **k)
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    httpd = serve()
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=[
            "--enable-unsafe-webgpu", "--enable-features=Vulkan",
            "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1180, "height": 940}, device_scale_factor=1)
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_selector("#run:not([disabled])", timeout=120_000)

        box = page.locator(".screen").bounding_box()  # CSS px

        page.click("#vidchip")
        page.wait_for_function(
            "()=>{const c=document.querySelector('#count').textContent;return c!=='—'&&+c>10;}",
            timeout=120_000)
        page.wait_for_timeout(400)  # let the video paint before capturing

        client = page.context.new_cdp_session(page)

        def on_frame(ev):
            img = Image.open(io.BytesIO(base64.b64decode(ev["data"]))).convert("RGB")
            frames.append((img, ev["metadata"]["timestamp"]))  # store full, crop later
            try:
                client.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception:
                pass

        client.on("Page.screencastFrame", on_frame)
        client.send("Page.startScreencast",
                    {"format": "jpeg", "quality": 88, "everyNthFrame": 1,
                     "maxWidth": 1180, "maxHeight": 940})
        page.wait_for_timeout(int(CAPTURE_S * 1000))
        client.send("Page.stopScreencast")
        page.wait_for_timeout(150)
        browser.close()
    httpd.shutdown()

    if len(frames) < 5:
        raise SystemExit(f"only {len(frames)} frames captured")

    # the screencast image may be scaled vs the CSS viewport — derive the scale
    # from the actual frame size (1180 CSS wide) and crop the .screen box by it,
    # trimming 1px off the bottom so no controls sliver bleeds in.
    fw, fh = frames[0][0].size
    sx, sy = fw / 1180, fh / 940
    crop = (round(box["x"] * sx), round(box["y"] * sy),
            round((box["x"] + box["width"]) * sx), round((box["y"] + box["height"]) * sy - 2))
    frames = [(f[0].crop(crop), f[1]) for f in frames]

    # subsample to ~16 fps by real timestamps (the camera is always moving, so
    # every frame has motion — we just want a real-time-paced, sane frame count).
    TARGET_FPS = 13
    kept = [frames[0]]
    for f in frames[1:]:
        if f[1] - kept[-1][1] >= 1.0 / TARGET_FPS:
            kept.append(f)

    durations = []
    for i in range(len(kept)):
        if i + 1 < len(kept):
            dt = (kept[i + 1][1] - kept[i][1]) * 1000
            durations.append(int(max(40, min(140, dt))))
        else:
            durations.append(int(1000 / TARGET_FPS))

    out_imgs = []
    for img, _ in kept:
        h = round(img.height * GIF_W / img.width)
        out_imgs.append(img.resize((GIF_W, h), Image.LANCZOS)
                        .quantize(colors=40, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))

    out = ASSETS / "demo_video.gif"
    out_imgs[0].save(out, save_all=True, append_images=out_imgs[1:],
                     duration=durations, loop=0, optimize=True)
    total = sum(durations)
    print(f"[vgif] wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(out_imgs)} frames, "
          f"{total/1000:.1f}s, ~{1000*len(out_imgs)/total:.0f} fps)")


if __name__ == "__main__":
    main()
