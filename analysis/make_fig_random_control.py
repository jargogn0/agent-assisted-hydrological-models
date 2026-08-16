#!/usr/bin/env python3
"""Figure 6: agent-assisted versus matched-budget random-proposal trajectories,
one panel per model family. Each family's control is matched to the budget the
agent actually received (20 iterations for the tabular families, 6 and 5 for the
sequence families), so panel x-ranges differ by design."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper4_pipeline/src"))

import matplotlib.pyplot as plt
from paper4.style import apply_style, COL_DOUBLE, MODEL_COLORS
from trajectory_stats import campaigns

AR = ROOT / "paper4_pipeline/outputs/autoresearch"
FIG = ROOT / "paper4_pipeline/outputs/hess_100train_50test_final/figures"
MARGIN = 0.25
# the two arms are drawn from the same palette used for model families;
# each panel names its family in the title, so colour here encodes the arm
AGENT_C, RAND_C = MODEL_COLORS["xgboost"], MODEL_COLORS["transformer"]
FAMS = [("xgboost", "XGBoost", 0), ("random_forest", "Random Forest", 0),
        ("xlstm", "xLSTM", -1), ("transformer", "Transformer", -1)]


def best_so_far(base: float, scores: list[float | None]) -> list[float]:
    b, out = base, [base]
    for s in scores:
        if s is not None and s > b + MARGIN:
            b = s
        out.append(b)
    return out


def main() -> None:
    apply_style()
    rs: dict[str, list] = {}
    for line in (AR / "random_search_log.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rs.setdefault(r["scope"], []).append(r)

    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 5.2))
    for ax, (fam, label, which) in zip(axes.ravel(), FAMS):
        c = campaigns(fam)[which]
        a_base = c["baseline"]["composite_score"]
        a_raw = [r.get("composite_score") for r in c["iterations"]]
        a_step = best_so_far(a_base, a_raw)

        recs = rs[fam]
        r_base = recs[0]["composite_score"]
        r_raw = [x["composite_score"] for x in recs[1:]]
        r_step = best_so_far(r_base, r_raw)

        n = len(a_step)
        ax.axhline(a_base, color="#666666", ls="--", lw=0.7, zorder=1)
        ax.plot(range(1, len(a_raw) + 1), a_raw, color=AGENT_C, lw=0.5, alpha=0.35, zorder=2)
        ax.plot(range(1, len(r_raw) + 1), r_raw, color=RAND_C, lw=0.5, alpha=0.35, zorder=2)
        ax.step(range(n), a_step, where="post", color=AGENT_C, lw=1.7,
                label=f"agent ({a_step[-1] - a_base:+.1f})", zorder=4)
        ax.step(range(len(r_step)), r_step, where="post", color=RAND_C, lw=1.7,
                label=f"random proposals ({r_step[-1] - r_base:+.1f})", zorder=4)
        ax.set_title(f"{label}   (budget {n - 1} experiments)", fontsize=7.6)
        leg = ax.legend(fontsize=6.2, loc="lower right", frameon=True, framealpha=0.92,
                        edgecolor="none", facecolor="white")
        leg.set_zorder(6)
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.set_xlim(0, n - 1)
        step = 5 if n > 10 else 1
        ax.set_xticks(range(0, n, step))
        ax.annotate("anchored baseline", (0.25, a_base), fontsize=5.4, color="#666666",
                    va="bottom", ha="left")
    for ax in axes[1]:
        ax.set_xlabel("Experiment")
    for ax in axes[:, 0]:
        ax.set_ylabel("Composite validation score")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig07_random_control.{ext}", dpi=600, bbox_inches="tight")
    print("wrote", FIG / "fig07_random_control.png")


if __name__ == "__main__":
    main()
