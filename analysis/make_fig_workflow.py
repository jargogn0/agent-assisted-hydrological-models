#!/usr/bin/env python3
"""Figure 2: the methodological workflow.

Deliberately spare. Each element carries a short label only; the detail lives in
Sect. 2.4 and in Tables 1 and 2. No numerical results appear here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper4_pipeline/src"))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from paper4.style import apply_style, COL_DOUBLE

FIG = ROOT / "paper4_pipeline/outputs/hess_100train_50test_final/figures"

ACCENT = "#1B4F72"     # loop
GREY = "#5B6B75"       # fixed context and outcomes
LINE = "#3D4C55"
SOFT = "#DCE4EA"
PAPER = "#FFFFFF"


def node(ax, cx, cy, w, h, label, num=None, accent=True, fill=PAPER, fs=7.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                facecolor=fill, edgecolor=ACCENT if accent else GREY,
                                lw=1.0, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=LINE, zorder=4, linespacing=1.35)
    if num is not None:
        ax.add_patch(Circle((cx - w / 2 + 0.16, cy + h / 2 - 0.16), 0.135,
                            facecolor=ACCENT, edgecolor="none", zorder=5))
        ax.text(cx - w / 2 + 0.16, cy + h / 2 - 0.163, str(num), ha="center",
                va="center", fontsize=5.4, color="white", zorder=6, fontweight="bold")


def arrow(ax, p0, p1, rad=0.0, color=LINE, ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=color, ls=ls, zorder=2, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.15))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 8.6)

    def stage(y, text):
        ax.text(0.0, y, text, fontsize=6.4, color=GREY, fontweight="bold",
                letterspacing=1.2 if hasattr(ax, "letterspacing") else None)

    # ---------- fixed context -------------------------------------------------
    ax.text(0.0, 8.25, "FIXED BEFORE THE CAMPAIGN", fontsize=6.2, color=GREY,
            fontweight="bold")
    chips = ["approved\ndecision space", "per-experiment\nbudget", "anchored\nexpert baseline",
             "hydrological\nscorecard", "promotion rule\nand safeguards"]
    cw, cgap = 2.10, 0.36
    x0 = 0.0
    for i, c in enumerate(chips):
        x = x0 + i * (cw + cgap)
        ax.add_patch(FancyBboxPatch((x, 7.10), cw, 0.82,
                                    boxstyle="round,pad=0.0,rounding_size=0.08",
                                    facecolor=SOFT, edgecolor="none", zorder=3))
        ax.text(x + cw / 2, 7.51, c, ha="center", va="center", fontsize=6.3,
                color=GREY, zorder=4, linespacing=1.3)
    ax.plot([0, 12], [6.72, 6.72], color=SOFT, lw=0.9, zorder=1)

    # ---------- the loop ------------------------------------------------------
    ax.text(0.0, 6.30, "ONE LOGGED ITERATION", fontsize=6.2, color=ACCENT,
            fontweight="bold")
    w, h = 3.00, 1.02
    top_y, bot_y = 5.10, 3.22
    cxs = [1.55, 6.00, 10.45]
    node(ax, cxs[0], top_y, w, h, "Agent proposal", 1)
    node(ax, cxs[1], top_y, w, h, "Validity gate", 2)
    node(ax, cxs[2], top_y, w, h, "Fixed-budget\nexperiment", 3)
    node(ax, cxs[2], bot_y, w, h, "Hydrological\nscorecard", 4)
    node(ax, cxs[1], bot_y, w, h, "Promote or reject", 5)
    node(ax, cxs[0], bot_y, w, h, "Audit log", 6)

    arrow(ax, (cxs[0] + w / 2, top_y), (cxs[1] - w / 2, top_y))
    arrow(ax, (cxs[1] + w / 2, top_y), (cxs[2] - w / 2, top_y))
    arrow(ax, (cxs[2], top_y - h / 2), (cxs[2], bot_y + h / 2))
    arrow(ax, (cxs[2] - w / 2, bot_y), (cxs[1] + w / 2, bot_y))
    arrow(ax, (cxs[1] - w / 2, bot_y), (cxs[0] + w / 2, bot_y))
    # return leg, curved so the cycle reads as a cycle
    arrow(ax, (cxs[0], bot_y + h / 2), (cxs[0], top_y - h / 2), rad=0.62,
          color=ACCENT, ls="--")
    ax.text(cxs[0] - 1.02, (top_y + bot_y) / 2, "next\niteration", fontsize=6.0,
            color=ACCENT, ha="center", va="center", linespacing=1.3)

    # ---------- confirmation --------------------------------------------------
    ax.plot([0, 12], [2.32, 2.32], color=SOFT, lw=0.9, zorder=1)
    ax.text(12.0, 1.98, "AFTER THE BUDGET IS EXHAUSTED", fontsize=6.2, color=GREY,
            fontweight="bold", ha="right")
    # the loop is left through the audit log, straight down: no diagonals
    node(ax, cxs[0], 1.02, w, 0.86, "Frozen configuration", accent=False, fill=SOFT, fs=7.2)
    node(ax, 8.05, 1.02, 6.9, 0.86,
         "Temporal and held-out-catchment confirmation", accent=False, fill=SOFT, fs=7.2)
    arrow(ax, (cxs[0], bot_y - h / 2), (cxs[0], 1.02 + 0.86 / 2), color=GREY)
    arrow(ax, (cxs[0] + w / 2, 1.02), (8.05 - 6.9 / 2, 1.02), color=GREY)
    ax.text(8.05, 0.30, "evaluated once; results never re-enter the loop",
            ha="center", fontsize=6.1, color=GREY, style="italic")

    fig.tight_layout(pad=0.15)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig02_workflow_schema.{ext}", dpi=600, bbox_inches="tight")
    print("wrote", FIG / "fig02_workflow_schema.png")


if __name__ == "__main__":
    main()
