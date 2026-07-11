"""Generate the README benchmark chart(s) from the measured results.

Plots ACCURACY LOSS % (not raw mAP): zero-based so bars are honest, small
differences stay visible on a 0-6% range, and the 2% budget is a clean line.
Dark, mono, amber — matches the WebGPU demo's instrument aesthetic.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# palette (matches the demo)
BG = "#0b0c0a"; PANEL = "#141613"; INK = "#e9e7dc"; DIM = "#8b9080"
AMBER = "#ffab2e"; RED = "#ff5b45"; LINE = "#282c22"

# measured results: (label, engine, accuracy-loss %, passes budget)
ROWS = [
    ("FP16", "TensorRT · GPU", 0.06, True),
    ("INT8", "ORT · CPU",      0.72, True),
    ("FP8",  "TensorRT · GPU", 1.12, True),
    ("INT8", "TensorRT · GPU", 5.65, False),
]
BUDGET = 2.0

for f in ("JetBrains Mono", "DejaVu Sans Mono"):
    if any(f in fp.name for fp in font_manager.fontManager.ttflist) or f == "DejaVu Sans Mono":
        plt.rcParams["font.family"] = "monospace"
        break


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    x = range(len(ROWS))
    losses = [r[2] for r in ROWS]
    colors = [AMBER if r[3] else RED for r in ROWS]
    bars = ax.bar(x, losses, width=0.62, color=colors, zorder=3,
                  edgecolor=BG, linewidth=2)

    # 2% budget threshold
    ax.axhline(BUDGET, color=DIM, ls=(0, (5, 4)), lw=1.4, zorder=2)
    ax.text(-0.55, BUDGET + 0.14, "2% ACCURACY BUDGET", color=DIM,
            fontsize=9, ha="left", va="bottom", family="monospace",
            fontweight="bold")

    # per-bar labels: loss %, verdict, engine
    for i, (bar, (prec, eng, loss, ok)) in enumerate(zip(bars, ROWS)):
        ax.text(i, loss + 0.13, f"{loss:.2f}%", ha="center", va="bottom",
                color=INK, fontsize=12, fontweight="bold", family="monospace")
        ax.text(i, -0.32, prec, ha="center", va="top", color=INK, fontsize=12,
                fontweight="bold", family="monospace")
        ax.text(i, -0.72, eng, ha="center", va="top", color=DIM, fontsize=8.5,
                family="monospace")
        tag = "PASS" if ok else "OVER BUDGET"
        ax.text(i, loss / 2, tag, ha="center", va="center", rotation=90,
                color=BG, fontsize=8.5, fontweight="bold", family="monospace")

    ax.set_title("ACCURACY COST OF QUANTIZATION",
                 color=INK, fontsize=14, fontweight="bold", loc="left",
                 family="monospace", pad=16)
    ax.text(0, 1.02, "mAP@50-95 loss vs FP32 baseline (0.5716) · YOLO26-s · SKU-110K",
            transform=ax.transAxes, color=DIM, fontsize=9.5, family="monospace")

    ax.set_ylim(0, 6.4); ax.set_xlim(-0.6, len(ROWS) - 0.4)
    ax.set_ylabel("accuracy loss (%)", color=DIM, fontsize=10, family="monospace")
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.margins(y=0.2)

    fig.subplots_adjust(bottom=0.16, top=0.82, left=0.09, right=0.97)
    out = ASSETS / "benchmark_accuracy.png"
    fig.savefig(out, facecolor=BG)
    print(f"[charts] wrote {out}")

    # card-fit variant: taller aspect (~16:10) + generous margins so a portfolio
    # card can object-cover it edge-to-edge, cropping only padding, not data.
    fig.set_size_inches(9, 5.6)
    fig.subplots_adjust(bottom=0.20, top=0.80, left=0.11, right=0.95)
    card = ASSETS / "benchmark_card.png"
    fig.savefig(card, facecolor=BG)
    print(f"[charts] wrote {card}")


if __name__ == "__main__":
    main()
