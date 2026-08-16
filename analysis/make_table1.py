#!/usr/bin/env python3
"""Generate Table 1: the complete configuration of every model family, baseline
against agent-selected, one parameter per row.

The earlier version packed every setting into a single cell per family, which
wrapped unreadably. This reports the same information, plus the shared feature
and sampling setup, with one parameter per row and an explicit column marking
the parameters the agent modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "paper4_pipeline"
AR = PIPE / "outputs" / "autoresearch"
GEN = AR / "generated_configs"

LBL = {"random_forest": "Random Forest", "xgboost": "XGBoost",
       "xlstm": "xLSTM", "transformer": "Transformer"}
SEQ = {"xlstm", "transformer"}
BASE = {False: PIPE / "configs/generated/agent_dev_base.yaml",
        True: PIPE / "configs/generated/agent_dev_base_seq_fast.yaml"}

# ordered (config key, printed name) per family; keys are dotted paths
COMMON = [
    ("features.dynamic", "dynamic predictors"),
    ("features.rolling_windows", "antecedent windows (days)"),
    ("features.static_families", "static attribute groups"),
    ("models.sample_seed", "sampling seed"),
]
PER_FAMILY = {
    "random_forest": [
        ("models.random_forest.n_estimators", "number of trees"),
        ("models.random_forest.max_depth", "maximum depth"),
        ("models.random_forest.min_samples_leaf", "minimum samples per leaf"),
        ("models.random_forest.max_samples", "bootstrap sample fraction"),
        ("models.random_forest.max_features", "features considered per split"),
        ("models.random_forest.max_dynamic_window", "maximum antecedent window (days)"),
        ("models.random_forest.peak_flow_quantile", "peak-flow sampling quantile"),
        ("models.random_forest.peak_flow_extra_per_basin", "extra peak-flow samples per catchment"),
        ("models.random_forest.random_state", "random seed"),
    ],
    "xgboost": [
        ("models.xgboost.n_estimators", "number of trees"),
        ("models.xgboost.max_depth", "maximum depth"),
        ("models.xgboost.learning_rate", "learning rate"),
        ("models.xgboost.subsample", "row subsample"),
        ("models.xgboost.colsample_bytree", "column subsample"),
        ("models.xgboost.reg_lambda", "L2 regularisation"),
        ("models.xgboost.objective", "objective"),
        ("models.xgboost.random_state", "random seed"),
    ],
}
SEQ_KEYS = [
    ("sequence_models.sequence_length", "lookback length (days)"),
    ("sequence_models.hidden_size", "hidden size"),
    ("sequence_models.num_layers", "number of layers"),
    ("sequence_models.attention_heads", "attention heads"),
    ("sequence_models.dropout", "dropout"),
    ("sequence_models.learning_rate", "learning rate"),
    ("sequence_models.batch_size", "batch size"),
    ("sequence_models.loss", "loss function"),
    ("sequence_models.grad_clip", "gradient clipping"),
    ("sequence_models.cosine_schedule", "cosine learning-rate schedule"),
    ("sequence_models.early_stopping_patience", "early-stopping patience (epochs)"),
    ("sequence_models.epochs_by_model.{fam}", "training epochs"),
    ("sequence_models.sample_limit_train", "training sample limit"),
    ("sequence_models.sample_limit_eval", "evaluation sample limit"),
    ("sequence_models.random_seed", "random seed"),
]


def get(cfg: dict, dotted: str):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def last_promoted(scope: str) -> str | None:
    best = None
    for line in (AR / "agent_log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("scope") == scope and r.get("outcome") == "promote":
            best = r["tag"]
    return best


def fmt(v) -> str:
    if v is None:
        return "not applicable"
    if isinstance(v, bool):
        return "enabled" if v else "disabled"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def main() -> None:
    rows: list[str] = []

    # shared block, taken from the tabular baseline (identical in the sequence base)
    b_tab = yaml.safe_load(BASE[False].read_text())
    xgb_sel = yaml.safe_load((GEN / f"{last_promoted('xgboost')}.yaml").read_text())
    first = True
    for key, name in COMMON:
        bv = get(b_tab, key)
        rows.append(f"| {'Shared setup' if first else ''} | {name} | {fmt(bv)} | unchanged | - |")
        first = False

    for fam in ["random_forest", "xgboost", "xlstm", "transformer"]:
        is_seq = fam in SEQ
        b = yaml.safe_load(BASE[is_seq].read_text())
        s = yaml.safe_load((GEN / f"{last_promoted(fam)}.yaml").read_text())
        keys = list(PER_FAMILY[fam]) if not is_seq else [
            (k.format(fam=fam), n) for k, n in SEQ_KEYS]
        if not is_seq:
            keys = keys + [("models.sample_per_basin_train", "training samples per catchment")]
        first = True
        for key, name in keys:
            bv, sv = get(b, key), get(s, key)
            if bv is None and sv is None:
                continue
            if fam == "xlstm" and "attention_heads" in key:
                continue  # attention heads apply to the Transformer only
            mod = "yes" if bv != sv else "-"
            rows.append(f"| {LBL[fam] if first else ''} | {name} | {fmt(bv)} | {fmt(sv)} | {mod} |")
            first = False

    tbl = ("| Family | Parameter | Expert baseline | Agent | Modified |\n"
           "|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (ROOT / "table1_full.md").write_text(tbl)
    n_mod = sum(1 for r in rows if r.rstrip().endswith("| yes |"))
    print(f"{len(rows)} rows, {n_mod} modified parameters")
    print(tbl)


if __name__ == "__main__":
    main()
