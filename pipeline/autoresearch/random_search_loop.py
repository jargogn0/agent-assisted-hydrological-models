#!/usr/bin/env python3
"""Matched-budget random-search control for the agent-assisted loop.

Identical in every respect to agent_loop.py except the source of proposals:
configurations are drawn uniformly at random from the same approved decision
space instead of being proposed by the LLM agent. Same anchored baseline, same
per-experiment budget, same scorecard, same promotion rule, same logging.

This provides the controlled comparison the agent loop cannot make on its own:
whether agent proposals outperform random sampling of the same space under an
equal experimental budget.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

PIPE = Path(__file__).resolve().parents[1]
ROOT = PIPE.parent
AR = PIPE / "outputs" / "autoresearch"
LOG = AR / "random_search_log.jsonl"
GEN = AR / "generated_configs"

# the same decision space the agent is given, expressed as samplers
SPACE = {
    "random_forest": {
        "models.random_forest.n_estimators": lambda r: r.choice([100, 150, 200, 300, 400, 500, 600]),
        "models.random_forest.max_depth": lambda r: r.choice([6, 10, 14, 18, 22, 24, 26, 30]),
        "models.random_forest.min_samples_leaf": lambda r: r.choice([1, 2, 3, 4, 6, 8, 10]),
        "models.random_forest.max_samples": lambda r: round(r.uniform(0.5, 1.0), 2),
        "models.sample_per_basin_train": lambda r: r.choice([100, 150, 200, 300, 400, 500, 600]),
    },
    "xgboost": {
        "models.xgboost.n_estimators": lambda r: r.choice([100, 200, 300, 400, 500, 600, 800]),
        "models.xgboost.max_depth": lambda r: r.choice([3, 4, 5, 6, 7, 8, 10, 12]),
        "models.xgboost.learning_rate": lambda r: round(r.uniform(0.005, 0.2), 3),
        "models.xgboost.subsample": lambda r: round(r.uniform(0.6, 1.0), 2),
        "models.xgboost.colsample_bytree": lambda r: round(r.uniform(0.5, 1.0), 2),
        "models.xgboost.reg_lambda": lambda r: round(r.uniform(0.5, 10), 2),
        "models.sample_per_basin_train": lambda r: r.choice([100, 150, 200, 300, 400, 500, 600]),
    },
    # sequence families share one decision space; attention_heads is transformer-only
    "_sequence": {
        "sequence_models.sequence_length": lambda r: r.choice([90, 120, 180, 210, 270, 330, 365]),
        "sequence_models.hidden_size": lambda r: r.choice([64, 96, 128, 160, 192, 256]),
        "sequence_models.num_layers": lambda r: r.choice([1, 2, 3]),
        "sequence_models.dropout": lambda r: round(r.uniform(0.0, 0.4), 2),
        "sequence_models.learning_rate": lambda r: round(r.uniform(0.00005, 0.002), 5),
        "sequence_models.batch_size": lambda r: r.choice([128, 192, 256, 384, 512]),
        "sequence_models.sample_limit_train": lambda r: r.choice([40000, 60000, 80000, 100000, 120000]),
    },
}
SEQ = {"xlstm", "transformer"}
WINDOWS = [1, 3, 7, 14, 30, 60, 90, 180, 365, 540]


def log(rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sample(scope: str, rng: random.Random) -> dict:
    """Draw one modification: mirrors the agent's 'one bounded change' rule by
    sampling a single key from the approved space, plus an occasional
    rolling-window change to match the agent's feature-structure option."""
    if rng.random() < 0.15:
        k = rng.randint(4, 8)
        return {"features.rolling_windows": sorted(rng.sample(WINDOWS, k))}
    if scope in SEQ:
        keys = dict(SPACE["_sequence"])
        if scope == "transformer":
            keys["sequence_models.attention_heads"] = lambda r: r.choice([2, 4, 8])
        key = rng.choice(list(keys))
        return {key: keys[key](rng)}
    key = rng.choice(list(SPACE[scope]))
    return {key: SPACE[scope][key](rng)}


def apply_mutations(cfg: dict, mutations: dict) -> dict:
    out = copy.deepcopy(cfg)
    for dotted, val in mutations.items():
        node = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def run(cfg_path: Path, tag: str, scope: str, note: str) -> tuple[str, float | None]:
    cmd = [str(PIPE / ".venv_arm64" / "bin" / "python"), "-u",
           str(PIPE / "autoresearch" / "runner.py"),
           "--config", str(cfg_path), "--tag", tag,
           "--hypothesis", note, "--tier", "tier1_proxy",
           "--model-scope", scope, "--selection-splits", "val",
           "--notes", "random-search control", "--steps", "tabular"]
    if scope in SEQ:
        # the tabular step builds the model frame the sequence trainer requires
        cmd += ["--sequence-models", scope, "--sequence-device", "mps"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    df = pd.read_csv(AR / "results.tsv", sep="\t")
    row = df[df["tag"].eq(tag)].tail(1)
    score = None
    if not row.empty and "composite_score" in row:
        try:
            score = float(row["composite_score"].iloc[0])
        except (TypeError, ValueError):
            score = None
    return ("ok" if proc.returncode == 0 else "failed"), score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True,
                    choices=["random_forest", "xgboost", "xlstm", "transformer"])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    best_cfg_path = Path(args.base_config)
    stamp0 = time.strftime("%Y%m%d_%H%M%S")

    # iteration 0: the same anchored baseline
    base_tag = f"rand_{args.scope}_baseline0_{stamp0}"
    cfg0 = yaml.safe_load(best_cfg_path.read_text())
    cfg0["project"]["run_name"] = base_tag
    cfg0["project"]["output_dir"] = str(AR / "runs" / base_tag)
    p0 = GEN / f"{base_tag}.yaml"
    p0.write_text(yaml.safe_dump(cfg0, sort_keys=False))
    st, lineage_best = run(p0, base_tag, args.scope, "Iteration 0: expert baseline, no modification")
    log({"iteration": 0, "scope": args.scope, "tag": base_tag, "outcome": "baseline",
         "composite_score": lineage_best, "lineage_best": lineage_best, "seed": args.seed})
    print(f"[random-search] {base_tag}: baseline={lineage_best}", flush=True)

    for i in range(1, args.iterations + 1):
        cfg = yaml.safe_load(best_cfg_path.read_text())
        mut = sample(args.scope, rng)
        tag = f"rand_{args.scope}_{time.strftime('%Y%m%d_%H%M%S')}_i{i:02d}"
        new = apply_mutations(cfg, mut)
        new["project"]["run_name"] = tag
        new["project"]["output_dir"] = str(AR / "runs" / tag)
        cp = GEN / f"{tag}.yaml"
        cp.write_text(yaml.safe_dump(new, sort_keys=False))
        t0 = time.time()
        st, score = run(cp, tag, args.scope, f"Random-search proposal: {json.dumps(mut)[:180]}")
        if score is not None and st == "ok":
            if lineage_best is None or score > lineage_best + 0.25:
                decision = "promote"
                lineage_best = score if lineage_best is None else max(lineage_best, score)
            else:
                decision = "reject"
        else:
            decision = "failed"
        log({"iteration": i, "scope": args.scope, "tag": tag, "mutations": mut,
             "outcome": decision, "composite_score": score, "lineage_best": lineage_best,
             "exp_wall_s": round(time.time() - t0, 1), "seed": args.seed})
        print(f"[random-search] {tag}: {decision} score={score} best={lineage_best}", flush=True)
        if decision == "promote":
            best_cfg_path = cp

    print(f"[random-search] done: {args.scope}, best config {best_cfg_path}")


if __name__ == "__main__":
    main()
