#!/usr/bin/env python3
"""Paired basin-bootstrap confidence intervals for the baseline-to-selected
contrasts, exactly as specified in Methods Sect. 2.6: catchments resampled with
replacement 2000 times, fixed seed 42, pairing preserved, and only catchments
with valid simulations under both configurations included.

Writes bootstrap_ci.json and a markdown table for the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "paper4_pipeline/outputs/autoresearch/runs"
N_BOOT = 2000
SEED = 42
LBL = {"random_forest": "Random Forest", "xgboost": "XGBoost",
       "xlstm": "xLSTM", "transformer": "Transformer"}
SPLITS = [("val", "Development validation"), ("test", "Temporal confirmation"),
          ("spatial_test", "Held-out catchments")]


def load(run: str, model: str, split: str) -> pd.DataFrame:
    f = R / run / "tables" / "metrics_by_basin.csv"
    d = pd.read_csv(f)
    d = d[(d.model == model) & (d.split == split)]
    return d.set_index("ID")


def paired(fam: str, split: str, col: str = "kge"):
    """Return (n, median_base, median_sel, delta, lo, hi, p_two_sided)."""
    b = load(f"agent_confirm_baseline_{fam}", fam, split)[col]
    s = load(f"agent_confirm_selected_{fam}", fam, split)[col]
    common = b.index.intersection(s.index)
    b, s = b.loc[common], s.loc[common]
    ok = b.notna() & s.notna() & np.isfinite(b) & np.isfinite(s)
    b, s = b[ok].to_numpy(), s[ok].to_numpy()
    n = len(b)
    if n == 0:
        return None
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    # paired resample: the same catchment draw is applied to both configurations
    d_boot = np.median(s[idx], axis=1) - np.median(b[idx], axis=1)
    delta = float(np.median(s) - np.median(b))
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    # two-sided bootstrap p: proportion of resamples on the far side of zero
    p = 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / N_BOOT)))
    return dict(n=int(n), base=round(float(np.median(b)), 3),
                sel=round(float(np.median(s)), 3), delta=round(delta, 3),
                lo=round(float(lo), 3), hi=round(float(hi), 3), p=p)


def main() -> None:
    out: dict = {}
    rows = []
    for fam in LBL:
        out[fam] = {}
        for sp, name in SPLITS:
            r = paired(fam, sp)
            if r is None:
                continue
            out[fam][sp] = r
            star = "" if (r["lo"] <= 0 <= r["hi"]) else "*"
            rows.append(f"| {LBL[fam]} | {name} | {r['base']:.3f} | {r['sel']:.3f} | "
                        f"{r['delta']:+.3f} | ({r['lo']:+.3f}, {r['hi']:+.3f}){star} | {r['n']} |")
            print(f"{LBL[fam]:14s} {name:24s} d={r['delta']:+.3f} "
                  f"CI=({r['lo']:+.3f},{r['hi']:+.3f}) p={r['p']:.3f} n={r['n']}")
    hdr = ("| Family | Domain | Expert baseline KGE | Agent KGE | Change | 95 % CI | n |\n"
           "|---|---|---|---|---|---|---|\n")
    (ROOT / "table4_with_ci.md").write_text(hdr + "\n".join(rows) + "\n")
    (ROOT / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print("\nwrote table4_with_ci.md and bootstrap_ci.json")


if __name__ == "__main__":
    main()
