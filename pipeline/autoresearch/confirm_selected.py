#!/usr/bin/env python3
"""Stage 2-3 confirmations for the clean agent campaign.

For each model family: take the last promoted configuration from agent_log.jsonl
(falling back to the expert baseline when the loop promoted nothing), freeze it,
and evaluate it once on validation, temporal-confirmation, and held-out-catchment
splits. The expert baselines are confirmed the same way, so every family has a
clean baseline-vs-selected comparison across all three domains.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

PIPE = Path(__file__).resolve().parents[1]
ROOT = PIPE.parent
AR = PIPE / "outputs" / "autoresearch"
GEN = AR / "generated_configs"
PY = str(PIPE / ".venv_arm64" / "bin" / "python")

SCOPES = ["random_forest", "xgboost", "xlstm", "transformer"]
SEQ = {"xlstm", "transformer"}
BASES = {False: PIPE / "configs" / "generated" / "agent_dev_base.yaml",
         True: PIPE / "configs" / "generated" / "agent_dev_base_seq_fast.yaml"}


def last_promoted(scope: str) -> Path | None:
    best = None
    for line in (AR / "agent_log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("scope") == scope and r.get("outcome") == "promote" and "lineage_best" in r:
            best = GEN / f"{r['tag']}.yaml"
    return best if best and best.exists() else None


def confirm(name: str, src_cfg: Path, scope: str, is_seq: bool) -> None:
    out = AR / "runs" / name
    if (out / "tables" / "metrics_by_basin.csv").exists():
        print(f"[confirm] {name}: already done, skipping", flush=True)
        return
    cfg = yaml.safe_load(src_cfg.read_text())
    cfg["project"]["run_name"] = name
    cfg["project"]["output_dir"] = str(out)
    cfg.setdefault("evaluation", {})["prediction_splits"] = ["val", "test", "spatial_test"]
    cfg_path = GEN / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = [PY, "-u", str(PIPE / "autoresearch" / "runner.py"),
           "--config", str(cfg_path), "--tag", name,
           "--hypothesis", f"Tier-3 confirmation of {name} on locked temporal and held-out splits",
           "--tier", "tier3_final", "--model-scope", scope,
           "--selection-splits", "test",
           "--notes", "clean-campaign confirmation; no tuning from these results",
           "--steps", "tabular"]
    if is_seq:
        cmd += ["--sequence-models", scope, "--sequence-device", "mps"]
    print(f"[confirm] running {name} from {src_cfg.name}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[confirm] {name}: exit {rc}", flush=True)


def main() -> None:
    only = sys.argv[1:] or SCOPES
    for scope in only:
        is_seq = scope in SEQ
        confirm(f"agent_confirm_baseline_{scope}", BASES[is_seq], scope, is_seq)
        sel = last_promoted(scope)
        if sel is None:
            print(f"[confirm] {scope}: no promoted configuration, baseline is the selection", flush=True)
            continue
        confirm(f"agent_confirm_selected_{scope}", sel, scope, is_seq)
    print("[confirm] CONFIRMATION STAGE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
