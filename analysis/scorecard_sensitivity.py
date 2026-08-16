#!/usr/bin/env python3
"""Post hoc scorecard sensitivity analysis promised in Methods Sect. 2.6.

Re-ranks every completed development configuration under three alternative
scorecard variants, without retraining anything, and quantifies (a) rank
stability against the operational composite and (b) whether the top-ranked
configuration changes. Reports Spearman rank correlations and the identity of
the winner under each variant, per model family.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
AR = ROOT / "paper4_pipeline/outputs/autoresearch"
LBL = {"random_forest": "Random Forest", "xgboost": "XGBoost",
       "xlstm": "xLSTM", "transformer": "Transformer"}


def norm(x: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Min-max to [0, 100] within the family, matching the operational score's
    within-family interpretation."""
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(50.0, index=x.index)
    z = (x - lo) / (hi - lo)
    return 100 * (z if higher_is_better else 1 - z)


VARIANTS = {
    "efficiency-weighted": lambda d: norm(d.median_kge),
    "water-balance-weighted": lambda d: (0.4 * norm(d.median_kge)
                                         + 0.6 * norm(d.median_abs_pbias, False)),
    "robustness-weighted": lambda d: (0.3 * norm(d.median_kge)
                                      + 0.4 * norm(d.q25_kge)
                                      + 0.3 * norm(d.failure_rate, False)),
}


def main() -> None:
    df = pd.read_csv(AR / "results.tsv", sep="\t")
    df = df[df.status.eq("ok") if "status" in df else slice(None)]
    df = df[df.tier.eq("tier1_proxy")]
    need = ["composite_score", "median_kge", "q25_kge", "median_abs_pbias", "failure_rate"]
    df = df.dropna(subset=need)
    # restrict to the reported campaign: the lineages analysed in trajectory_stats.py,
    # so the sensitivity analysis describes the same configurations as Table 3
    import sys
    sys.path.insert(0, str(ROOT))
    from trajectory_stats import campaigns
    keep: set[str] = set()
    for fam in LBL:
        for c in campaigns(fam)[-2:] if fam == "xgboost" else campaigns(fam)[-1:]:
            keep.add(c["baseline"]["tag"])
            keep.update(r["tag"] for r in c["iterations"])
    df = df[df.tag.isin(keep)]

    out, rows = {}, []
    for fam in LBL:
        d = df[df.model_scope.eq(fam)].drop_duplicates("tag").set_index("tag")
        if len(d) < 4:
            continue
        base_rank = d.composite_score.rank(ascending=False)
        winner = str(d.composite_score.idxmax())
        out[fam] = {"n_configs": int(len(d)), "operational_winner": winner, "variants": {}}
        print(f"\n=== {LBL[fam]} ({len(d)} completed configurations) ===")
        print(f"  operational composite winner: {winner}")
        for name, fn in VARIANTS.items():
            s = fn(d)
            rho = spearmanr(base_rank, s.rank(ascending=False)).statistic
            w = str(s.idxmax())
            same = w == winner
            # where does the operational winner land under this variant?
            pos = int(s.rank(ascending=False).loc[winner])
            out[fam]["variants"][name] = {"spearman": round(float(rho), 3),
                                          "winner": w, "same_winner": same,
                                          "operational_winner_rank": pos}
            rows.append(f"| {LBL[fam]} | {name} | {rho:+.2f} | "
                        f"{'unchanged' if same else 'changed'} | {pos} |")
            print(f"  {name:24s} rho={rho:+.2f}  winner "
                  f"{'unchanged' if same else 'CHANGED'}  (operational winner ranks {pos})")

    hdr = ("| Family | Scorecard variant | Rank correlation with operational score | "
           "Top-ranked configuration | Rank of operational winner |\n|---|---|---|---|---|\n")
    (ROOT / "table_sensitivity.md").write_text(hdr + "\n".join(rows) + "\n")
    (ROOT / "scorecard_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("\nwrote table_sensitivity.md and scorecard_sensitivity.json")


if __name__ == "__main__":
    main()
