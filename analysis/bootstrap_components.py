#!/usr/bin/env python3
"""Paired basin-bootstrap CIs for the individual scorecard components, not just
median KGE. Needed because the multi-criteria argument in Sect. 3.3/3.4 rests on
lower-quartile KGE, log-NSE, absolute PBIAS, and failure rate, which were
reported as point estimates only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "paper4_pipeline/outputs/autoresearch/runs"
N_BOOT, SEED = 2000, 42
LBL = {"random_forest": "Random Forest", "xgboost": "XGBoost",
       "xlstm": "xLSTM", "transformer": "Transformer"}
SPLITS = [("val", "Development validation"), ("test", "Temporal confirmation"),
          ("spatial_test", "Held-out catchments")]

# name -> (column, statistic over the catchment sample, higher_is_better)
STATS = {
    "median KGE":   ("kge", lambda a: np.median(a), True),
    "Q25 KGE":      ("kge", lambda a: np.percentile(a, 25), True),
    "median logNSE": ("log_nse", lambda a: np.median(a), True),
    "median |PBIAS|": ("pbias", lambda a: np.median(np.abs(a)), False),
    "failure rate": ("_fail", lambda a: a.mean(), False),
}


def frame(run: str, model: str, split: str) -> pd.DataFrame:
    d = pd.read_csv(R / run / "tables" / "metrics_by_basin.csv")
    d = d[(d.model == model) & (d.split == split)].copy()
    d["_fail"] = ((d.kge < 0) | (d.nse < 0)).astype(float)
    return d.set_index("ID")


def paired(fam: str, split: str, col: str, stat, cols_needed=("kge", "nse")):
    b = frame(f"agent_confirm_baseline_{fam}", fam, split)
    s = frame(f"agent_confirm_selected_{fam}", fam, split)
    common = b.index.intersection(s.index)
    b, s = b.loc[common], s.loc[common]
    ok = np.isfinite(b[col]) & np.isfinite(s[col])
    bv, sv = b.loc[ok, col].to_numpy(), s.loc[ok, col].to_numpy()
    n = len(bv)
    if n == 0:
        return None
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d_boot = np.array([stat(sv[i]) - stat(bv[i]) for i in idx])
    delta = float(stat(sv) - stat(bv))
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    p = 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())
    return dict(n=n, base=float(stat(bv)), sel=float(stat(sv)), delta=delta,
                lo=float(lo), hi=float(hi), p=float(min(1.0, max(p, 1 / N_BOOT))),
                sig=not (lo <= 0 <= hi))


def main() -> None:
    out: dict = {}
    for fam in LBL:
        out[fam] = {}
        print(f"\n=== {LBL[fam]} ===")
        for sp, name in SPLITS:
            out[fam][sp] = {}
            print(f"  {name}")
            for label, (col, stat, hib) in STATS.items():
                r = paired(fam, sp, col, stat)
                if r is None:
                    continue
                out[fam][sp][label] = r
                mark = "SIG" if r["sig"] else "ns "
                print(f"    {label:16s} {r['base']:+.3f} -> {r['sel']:+.3f}  "
                      f"d={r['delta']:+.3f} CI=({r['lo']:+.3f},{r['hi']:+.3f}) "
                      f"p={r['p']:.3f} [{mark}]")
    (ROOT / "bootstrap_components.json").write_text(json.dumps(out, indent=2))
    print("\nwrote bootstrap_components.json")


if __name__ == "__main__":
    main()
