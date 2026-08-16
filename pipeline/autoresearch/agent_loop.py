#!/usr/bin/env python3
"""Fully instrumented LLM-agent experiment loop.

Every iteration: build context from the experiment log -> ask the LLM agent
(claude CLI, headless) for ONE bounded modification -> validate it against the
approved decision space -> apply to the current best config -> execute through
runner.py (which scores and decides promote/reject) -> append a complete record
to agent_log.jsonl (LLM identity, prompt, verbatim proposal, category, config,
decision, scores, wall time). Invalid proposals are logged and consume budget
without execution, exactly as described in the manuscript.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

PIPE = Path(__file__).resolve().parents[1]
ROOT = PIPE.parent
AR = PIPE / "outputs" / "autoresearch"
BRIEF = (PIPE / "autoresearch" / "agent_brief.md").read_text(encoding="utf-8")
AGENT_LOG = AR / "agent_log.jsonl"
GEN = AR / "generated_configs"

ALLOWED_PREFIXES = (
    "features.dynamic", "features.rolling_windows",
    "models.random_forest.", "models.xgboost.", "models.sample_per_basin_train",
    "sequence_models.",
)

RANGES = {
    "models.random_forest.n_estimators": (100, 600),
    "models.random_forest.max_depth": (6, 30),
    "models.random_forest.min_samples_leaf": (1, 10),
    "models.random_forest.max_samples": (0.5, 1.0),
    "models.xgboost.n_estimators": (100, 800),
    "models.xgboost.max_depth": (3, 12),
    "models.xgboost.learning_rate": (0.005, 0.2),
    "models.xgboost.subsample": (0.6, 1.0),
    "models.xgboost.colsample_bytree": (0.5, 1.0),
    "models.xgboost.reg_lambda": (0.5, 10),
    "models.sample_per_basin_train": (100, 600),
    "sequence_models.sequence_length": (90, 365),
    "sequence_models.hidden_size": (64, 256),
    "sequence_models.num_layers": (1, 3),
    "sequence_models.dropout": (0.0, 0.4),
    "sequence_models.learning_rate": (0.00005, 0.002),
    "sequence_models.batch_size": (128, 512),
    "sequence_models.sample_limit_train": (40000, 120000),
    "sequence_models.attention_heads": (2, 8),
}


def log(rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with AGENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def results_tail(scope: str, k: int = 12) -> str:
    df = pd.read_csv(AR / "results.tsv", sep="\t")
    df = df[df["model_scope"].eq(scope)].tail(k)
    cols = ["tag", "decision", "status", "median_kge", "median_log_nse",
            "median_abs_pbias", "failure_rate", "composite_score", "hypothesis"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_string(index=False) if not df.empty else "(no prior experiments)"


def ask_agent(scope: str, cfg: dict, model: str) -> tuple[dict | None, dict]:
    prompt = (
        f"{BRIEF}\n\n## Current request\n\nModel family: {scope}\n\n"
        f"## Current best configuration (relevant sections)\n\n"
        f"features: {json.dumps(cfg.get('features', {}), default=str)[:1500]}\n"
        f"models: {json.dumps(cfg.get('models', {}), default=str)[:1200]}\n"
        f"sequence_models: {json.dumps(cfg.get('sequence_models', {}), default=str)[:1200]}\n\n"
        f"## Recent experiments for this family\n\n{results_tail(scope)}\n\n"
        "Reply with the single JSON object now."
    )
    t0 = time.time()
    proc = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json", "--model", model],
        capture_output=True, text=True, timeout=600,
    )
    meta = {"agent_wall_s": round(time.time() - t0, 1), "exit": proc.returncode,
            "prompt_verbatim": prompt, "raw_stdout": proc.stdout[:20000]}
    try:
        envelope = json.loads(proc.stdout)
        meta["llm_model"] = envelope.get("modelUsage") or envelope.get("model") or model
        meta["session_id"] = envelope.get("session_id")
        text = envelope.get("result", "")
    except json.JSONDecodeError:
        text = proc.stdout
        meta["llm_model"] = model
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, {**meta, "raw": text[:800]}
    try:
        return json.loads(text[start:end + 1]), meta
    except json.JSONDecodeError:
        return None, {**meta, "raw": text[:800]}


def validate(proposal: dict) -> str | None:
    if not isinstance(proposal.get("mutations"), dict) or not proposal["mutations"]:
        return "no mutations object"
    if len(proposal["mutations"]) > 6:
        return "too many simultaneous mutations"
    for key, val in proposal["mutations"].items():
        if not any(key == p or key.startswith(p) for p in ALLOWED_PREFIXES):
            return f"key outside decision space: {key}"
        base = key.replace("epochs_by_model.", "epochs_by_model_")
        for rk, (lo, hi) in RANGES.items():
            if base == rk and isinstance(val, (int, float)) and not (lo <= val <= hi):
                return f"value out of range: {key}={val}"
    return None


def apply_mutations(cfg: dict, mutations: dict) -> dict:
    out = copy.deepcopy(cfg)
    for dotted, val in mutations.items():
        node = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True,
                    choices=["random_forest", "xgboost", "xlstm", "transformer"])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--llm-model", default="claude-sonnet-5")
    ap.add_argument("--sequence-device", default="mps")
    args = ap.parse_args()

    best_cfg_path = Path(args.base_config)
    is_seq = args.scope in ("xlstm", "transformer")
    lineage_best = None  # composite score of the accepted lineage (starts at baseline)

    # Iteration 0: score the untouched expert baseline to anchor the lineage.
    # Without this, the first agent proposal would be accepted unconditionally.
    base_tag = f"agent_{args.scope}_baseline0_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg0 = yaml.safe_load(best_cfg_path.read_text())
    cfg0["project"]["run_name"] = base_tag
    cfg0["project"]["output_dir"] = str(AR / "runs" / base_tag)
    cfg0_path = GEN / f"{base_tag}.yaml"
    cfg0_path.write_text(yaml.safe_dump(cfg0, sort_keys=False))
    cmd0 = [str(PIPE / ".venv_arm64" / "bin" / "python"), "-u",
            str(PIPE / "autoresearch" / "runner.py"),
            "--config", str(cfg0_path), "--tag", base_tag,
            "--hypothesis", "Iteration 0: expert-defined baseline, no modification",
            "--tier", "tier1_proxy", "--model-scope", args.scope,
            "--selection-splits", "val", "--notes", "lineage baseline anchor",
            "--steps", "tabular"]
    if is_seq:
        cmd0 += ["--sequence-models", args.scope, "--sequence-device", args.sequence_device]
    t0 = time.time()
    subprocess.run(cmd0, cwd=str(ROOT), capture_output=True, text=True)
    df0 = pd.read_csv(AR / "results.tsv", sep="\t")
    row0 = df0[df0["tag"].eq(base_tag)].tail(1)
    if not row0.empty and "composite_score" in row0:
        try:
            lineage_best = float(row0["composite_score"].iloc[0])
        except (TypeError, ValueError):
            lineage_best = None
    log({"iteration": 0, "scope": args.scope, "tag": base_tag,
         "outcome": "baseline", "composite_score": lineage_best,
         "lineage_best": lineage_best, "exp_wall_s": round(time.time() - t0, 1),
         "base_config": str(best_cfg_path)})
    print(f"[agent-loop] {base_tag}: baseline score={lineage_best}", flush=True)

    for i in range(1, args.iterations + 1):
        cfg = yaml.safe_load(best_cfg_path.read_text())
        proposal, meta = ask_agent(args.scope, cfg, args.llm_model)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = f"agent_{args.scope}_{stamp}_i{i:02d}"
        base_rec = {"iteration": i, "scope": args.scope, "tag": tag,
                    "agent_meta": meta, "proposal": proposal,
                    "base_config": str(best_cfg_path)}

        if proposal is None:
            log({**base_rec, "outcome": "invalid", "reason": "unparseable response"})
            continue
        err = validate(proposal)
        if err:
            log({**base_rec, "outcome": "invalid", "reason": err})
            continue

        new_cfg = apply_mutations(cfg, proposal["mutations"])
        new_cfg["project"]["run_name"] = tag
        new_cfg["project"]["output_dir"] = str(AR / "runs" / tag)
        cfg_path = GEN / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(new_cfg, sort_keys=False))

        cmd = [str(PIPE / ".venv_arm64" / "bin" / "python"), "-u",
               str(PIPE / "autoresearch" / "runner.py"),
               "--config", str(cfg_path), "--tag", tag,
               "--hypothesis", str(proposal.get("hypothesis", ""))[:400],
               "--tier", "tier1_proxy", "--model-scope", args.scope,
               "--selection-splits", "val",
               "--reasoning", str(proposal.get("expected_effect", ""))[:400],
               "--notes", f"category={proposal.get('category', 'unspecified')};agent_iteration={i}",
               "--parent-tag", best_cfg_path.stem]
        if is_seq:
            # tabular step builds the model frame the sequence trainer requires
            # (and trains the cheap tabular models as a by-product)
            cmd += ["--steps", "tabular", "--sequence-models", args.scope,
                    "--sequence-device", args.sequence_device]
        else:
            cmd += ["--steps", "tabular"]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        wall = round(time.time() - t0, 1)

        df = pd.read_csv(AR / "results.tsv", sep="\t")
        row = df[df["tag"].eq(tag)].tail(1)
        runner_decision = row["decision"].iloc[0] if not row.empty else "unlogged"
        score = None
        if not row.empty and "composite_score" in row:
            try:
                score = float(row["composite_score"].iloc[0])
            except (TypeError, ValueError):
                score = None
        # lineage-local promotion: the clean rerun is its own comparison lineage,
        # independent of historical records for the same scope in results.tsv
        if score is not None and proc.returncode == 0:
            if lineage_best is None or score > lineage_best + 0.25:
                decision = "promote"
                lineage_best = score if lineage_best is None else max(lineage_best, score)
            else:
                decision = "reject"
        else:
            decision = "failed"
        log({**base_rec, "outcome": decision, "runner_decision": runner_decision,
             "composite_score": score, "lineage_best": lineage_best,
             "category": proposal.get("category"), "exp_wall_s": wall,
             "runner_exit": proc.returncode})
        print(f"[agent-loop] {tag}: {decision} (runner: {runner_decision}) score={score} best={lineage_best} wall={wall}s", flush=True)

        if decision == "promote":
            best_cfg_path = cfg_path

    print(f"[agent-loop] done: {args.scope}, best config: {best_cfg_path}")


if __name__ == "__main__":
    main()
