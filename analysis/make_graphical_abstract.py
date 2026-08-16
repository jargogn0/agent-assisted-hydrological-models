#!/usr/bin/env python3
"""Graphical abstract matching the confirmed findings: the four-family
matched-budget random-proposal control, the confirmed changes with bootstrap
intervals, and the dependence of selection on the scorecard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper4_pipeline/src"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from paper4.style import apply_style, MODEL_COLORS

FIG = ROOT / "paper4_pipeline/outputs/hess_100train_50test_final/figures"

# Model colours come from paper4.style.MODEL_COLORS (Paul Tol vibrant), the same
# palette used by Figures 3 to 7, so a family keeps one colour across the paper.
BLUE = MODEL_COLORS["xgboost"]        # #0077BB
ORANGE = MODEL_COLORS["transformer"]  # #EE7733
TEAL = MODEL_COLORS["xlstm"]          # #009988
GREY = MODEL_COLORS["random_forest"]  # #BBBBBB
CHANGED = BLUE           # "changed" cells in panel D
UNCHANGED = "#E7ECEF"
ACCENT = TEAL            # workflow schematic accent
INK = "#22333B"
MUTED = "#6B7C85"
HEAD = BLUE

# uniform typography across panels
TITLE_PT, LABEL_PT, TICK_PT, VALUE_PT, NOTE_PT = 8.4, 6.6, 6.4, 6.4, 6.0
GRID_A = 0.22

ci = json.load(open(ROOT / "bootstrap_ci.json"))
rv = json.load(open(ROOT / "random_vs_agent.json"))
LBL = {"xgboost": "XGBoost", "transformer": "Transformer",
       "random_forest": "Random Forest", "xlstm": "xLSTM"}
FAM_C = {k: MODEL_COLORS[k] for k in ("xgboost", "transformer",
                                      "random_forest", "xlstm")}
DOM = [("val", "validation"), ("test", "temporal"), ("spatial_test", "held-out")]
ORDER = ["xgboost", "transformer", "random_forest", "xlstm"]


def panel_loop(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.4)
    steps = ["LLM agent\nproposes one\nbounded change", "validity gate\n(decision space)",
             "fixed-budget\nexperiment", "hydrological\nscorecard", "promote\nor reject"]
    w, gap = 1.72, 0.22
    for i, s in enumerate(steps):
        x = i * (w + gap)
        last = i == len(steps) - 1
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 0.86,
                                    boxstyle="round,pad=0.03,rounding_size=0.06",
                                    facecolor="#E4F1EE" if last else "#EDF2F4",
                                    edgecolor=ACCENT if last else "#9DAFB8", lw=1.0))
        ax.text(x + w / 2, 1.43, s, ha="center", va="center", fontsize=6.6, color=INK)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.43), (x + w + gap - 0.02, 1.43),
                                         arrowstyle="-|>", mutation_scale=7, lw=0.9, color=INK))
    tot = 5 * w + 4 * gap
    xr, xl, yb = tot - w / 2, w / 2, 0.72
    ax.plot([xr, xr, xl], [1.0, yb, yb], color="#9DAFB8", lw=0.9, ls="--", zorder=0)
    ax.add_patch(FancyArrowPatch((xl + 0.001, yb), (xl, 0.99), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.9, color="#9DAFB8", ls="--"))
    ax.text(tot / 2, 0.42, "every proposal, decision and failure logged; "
                           "agent sees validation data only",
            ha="center", fontsize=6.0, color=MUTED)
    ax.text(0, 2.12, "A   Controlled agent loop", fontsize=TITLE_PT, color=HEAD, fontweight="bold")


def panel_control(ax) -> None:
    """Four families, plotted as the margin between the two proposal mechanisms
    so that families with very different score ranges remain comparable. Family
    order matches panels C and D, top to bottom."""
    ypos = list(range(len(ORDER)))[::-1]
    for y, fam in zip(ypos, ORDER):
        v = rv[fam]
        diff = v["r_d"] - v["a_d"]
        col = FAM_C[fam]
        ax.barh(y, diff, 0.42, color=col, edgecolor="#7A8891", lw=0.4)
        off = 0.25 if diff > 0 else -0.25
        ax.text(diff + off, y, f"{diff:+.1f}", va="center",
                ha="left" if diff > 0 else "right", fontsize=VALUE_PT, color=col)
        ax.text(-8.5, y - 0.40,
                f"agent {v['a_d']:+.1f}  /  random {v['r_d']:+.1f}",
                va="center", fontsize=NOTE_PT - 0.3, color=MUTED)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{LBL[f]}\n{rv[f]['a_n']} experiments" for f in ORDER],
                       fontsize=TICK_PT)
    for t, f in zip(ax.get_yticklabels(), ORDER):
        t.set_color(FAM_C[f] if f != "random_forest" else "#8A949B")
    ax.set_xlim(-8.8, 8.2)
    ax.set_ylim(-0.62, len(ORDER) - 0.38)
    ax.set_xlabel("random-proposal control minus agent (composite score units)", fontsize=LABEL_PT)
    ax.grid(True, axis="x", alpha=GRID_A, lw=0.4)
    ax.text(0.985, 0.97, "random proposals better", transform=ax.transAxes,
            ha="right", va="top", fontsize=NOTE_PT, color=MUTED)
    ax.text(0.015, 0.97, "agent better", transform=ax.transAxes, ha="left",
            va="top", fontsize=NOTE_PT, color=MUTED)
    ax.set_title("B   Random proposals win 3 of 4", fontsize=TITLE_PT,
                 color=HEAD, fontweight="bold", loc="left", pad=6)


def panel_confirm(ax) -> None:
    rows, labels, colors = [], [], []
    for fam in ORDER:
        for sp, dname in DOM:
            rows.append(ci[fam][sp])
            labels.append(f"{LBL[fam]}, {dname}")
            colors.append(FAM_C[fam])
    ypos = list(range(len(rows)))[::-1]
    for y, r, c in zip(ypos, rows, colors):
        res = not (r["lo"] <= 0 <= r["hi"])
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=1.6,
                alpha=1.0 if res else 0.30, solid_capstyle="round")
        ax.plot([r["delta"]], [y], "o", color=c, ms=3.8 if res else 3.0,
                alpha=1.0 if res else 0.30,
                markerfacecolor=c if res else "white",
                markeredgecolor=c, markeredgewidth=0.9)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=TICK_PT - 0.5)
    ax.set_xlabel("change in median KGE (95 % CI)", fontsize=LABEL_PT)
    ax.grid(True, axis="x", alpha=GRID_A, lw=0.4)
    ax.text(0.985, 0.04, "solid = interval excludes zero", transform=ax.transAxes,
            ha="right", fontsize=NOTE_PT, color=MUTED)
    ax.set_title("C   Only two families resolve", fontsize=TITLE_PT,
                 color=HEAD, fontweight="bold", loc="left", pad=6)


def panel_sensitivity(ax) -> None:
    s = json.load(open(ROOT / "scorecard_sensitivity.json"))
    fams = ORDER[::-1]
    variants = ["efficiency-weighted", "water-balance-weighted", "robustness-weighted"]
    for i, fam in enumerate(fams):
        for j, v in enumerate(variants):
            changed = not s[fam]["variants"][v]["same_winner"]
            ax.add_patch(plt.Rectangle((j, i), 0.9, 0.9,
                                       facecolor=CHANGED if changed else UNCHANGED,
                                       edgecolor="white", lw=1.2))
            ax.text(j + 0.45, i + 0.45, "changed" if changed else "same",
                    ha="center", va="center", fontsize=NOTE_PT - 0.4,
                    color="white" if changed else MUTED)
    ax.set_xlim(0, 3); ax.set_ylim(0, 4)
    ax.set_xticks([0.45, 1.45, 2.45])
    ax.set_xticklabels(["efficiency", "water\nbalance", "robustness"], fontsize=TICK_PT - 0.2)
    ax.set_yticks([i + 0.45 for i in range(4)])
    ax.set_yticklabels([LBL[f] for f in fams], fontsize=TICK_PT)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("alternative scorecard", fontsize=LABEL_PT)
    ax.set_title("D   Objective drives selection (7 of 12)", fontsize=TITLE_PT,
                 color=HEAD, fontweight="bold", loc="left", pad=6)


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(9.6, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.70, 1.0], hspace=0.44, wspace=0.58,
                          left=0.055, right=0.985, top=0.855, bottom=0.10)
    fig.suptitle("Agent-assisted development of machine-learning hydrological models",
                 fontsize=12.5, y=0.975, color=INK)
    fig.text(0.5, 0.918,
             "150 LamaH-CE catchments   |   71 logged experiments   |   4 model families   |   "
             "matched-budget random-proposal control   |   "
             "independent temporal and spatial confirmation",
             ha="center", fontsize=7.0, color=MUTED)
    panel_loop(fig.add_subplot(gs[0, :]))
    panel_control(fig.add_subplot(gs[1, 0]))
    panel_confirm(fig.add_subplot(gs[1, 1]))
    panel_sensitivity(fig.add_subplot(gs[1, 2]))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"graphical_abstract.{ext}", dpi=600, bbox_inches="tight")
        fig.savefig(ROOT / f"graphical_abstract.{ext}", dpi=600, bbox_inches="tight")
    print("wrote graphical_abstract.png/pdf")


if __name__ == "__main__":
    main()
