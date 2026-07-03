"""Drive the WebGPU demo in a headless browser to verify it actually works
end-to-end (model load -> preprocess -> inference -> box draw) and screenshot it.

Headless Chromium usually has no WebGPU, so this exercises the WASM fallback —
the *identical* JS pipeline, which is what we're verifying. Run:

    python scripts/test_web_demo.py
"""

import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "code" / "web_demo"
OUT = Path(__file__).resolve().parents[1] / "results" / "web_demo_shot.png"
PORT = 8123


def serve():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(ROOT), **k)
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    httpd = serve()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-webgpu",
                                          "--enable-features=Vulkan"])
        page = browser.new_page(viewport={"width": 1280, "height": 1400},
                                device_scale_factor=2)
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{PORT}/index.html")

        # wait for model load (Run button enables)
        page.wait_for_selector("#run:not([disabled])", timeout=120_000)
        badge = page.inner_text("#ep-badge")
        print("[test] runtime:", badge)

        page.click("#run")
        # the telemetry numbers count-up (animate), so reading them mid-tween is
        # wrong. The completion toast carries the AUTHORITATIVE final values:
        # "<N> products detected in <ms> ms · on-device".
        page.wait_for_function(
            "document.querySelector('#toast').textContent.includes('detected')",
            timeout=120_000)
        toast_txt = page.inner_text("#toast")
        import re
        m = re.search(r"(\d+) products detected in (\d+) ms", toast_txt)
        assert m, f"unexpected toast: {toast_txt!r}"
        count, ms = int(m.group(1)), int(m.group(2))
        print(f"[test] detections={count}  latency={ms}ms")
        assert count > 10, f"expected many detections on a shelf, got {count}"

        page.wait_for_timeout(800)  # let count-up animation settle for the shot
        OUT.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT))
        print(f"[test] screenshot -> {OUT}")
        browser.close()
    httpd.shutdown()
    if errors:
        print("[test] console errors:", errors[:5])
    print("[test] PASS — demo runs end-to-end in-browser")


if __name__ == "__main__":
    main()
