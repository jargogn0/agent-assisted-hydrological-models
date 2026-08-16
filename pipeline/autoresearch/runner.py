#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from results_io import RESULT_FIELDS, ensure_results_schema, read_results_tsv

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "paper4_pipeline"
RESULTS = PIPE / "outputs" / "autoresearch" / "results.tsv"
MEMORY = PIPE / "outputs" / "autoresearch" / "memory.json"


def _preferred_python() -> str:
    for candidate in [
        PIPE / ".venv_arm64" / "bin" / "python",
        PIPE / ".venv" / "bin" / "python",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_results() -> pd.DataFrame:
    ensure_results_schema(RESULTS)
    return read_results_tsv(RESULTS)


def _load_output_dir(config: str) -> Path:
    import yaml

    with Path(config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["project"]["output_dir"])
    return out if out.is_absolute() else ROOT / out


def _run_subprocess(cmd: list[str], log_name: str, timeout_s: int | None = None) -> tuple[int, float, str]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        code = proc.returncode
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        code = 124
        timeout_stdout = exc.stdout or ""
        if isinstance(timeout_stdout, bytes):
            timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
        stdout = timeout_stdout + "\n[autoresearch] candidate timed out\n"
    elapsed = time.time() - start
    log_dir = PIPE / "outputs" / "autoresearch"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / log_name).write_text(stdout, encoding="utf-8")
    return code, elapsed, stdout


def run_tabular_command(config: str, steps: list[str], timeout_s: int | None = None) -> tuple[int, float, str]:
    cmd = [
        _preferred_python(),
        str(PIPE / "scripts" / "run_pipeline.py"),
        "--config",
        config,
        "--steps",
        *steps,
    ]
    return _run_subprocess(cmd, "last_run_tabular.log", timeout_s=timeout_s)


def run_sequence_command(
    config: str,
    model: str,
    *,
    device: str | None = None,
    skip_report: bool = True,
    timeout_s: int | None = None,
) -> tuple[int, float, str]:
    cmd = [
        _preferred_python(),
        str(PIPE / "scripts" / "run_sequence.py"),
        "--config",
        config,
        "--model",
        model,
    ]
    if device:
        cmd.extend(["--device", device])
    if skip_report:
        cmd.append("--skip-report")
    return _run_subprocess(cmd, f"last_run_{model}.log", timeout_s=timeout_s)


def run_iteration(
    config: str,
    *,
    steps: list[str],
    sequence_models: list[str],
    sequence_device: str | None,
    timeout_s: int | None,
) -> tuple[int, float]:
    total_elapsed = 0.0
    combined_logs: list[str] = []

    code, elapsed, stdout = run_tabular_command(config, steps, timeout_s=timeout_s)
    total_elapsed += elapsed
    combined_logs.append(f"[tabular]\n{stdout}")
    if code != 0:
        (PIPE / "outputs" / "autoresearch" / "last_run.log").write_text("\n\n".join(combined_logs), encoding="utf-8")
        return code, total_elapsed

    for model in sequence_models:
        code, elapsed, stdout = run_sequence_command(
            config,
            model,
            device=sequence_device,
            skip_report=True,
            timeout_s=timeout_s,
        )
        total_elapsed += elapsed
        combined_logs.append(f"[sequence:{model}]\n{stdout}")
        if code != 0:
            (PIPE / "outputs" / "autoresearch" / "last_run.log").write_text("\n\n".join(combined_logs), encoding="utf-8")
            return code, total_elapsed

    if sequence_models:
        code, elapsed, stdout = run_tabular_command(config, ["plots", "report"], timeout_s=timeout_s)
        total_elapsed += elapsed
        combined_logs.append(f"[report]\n{stdout}")
        if code != 0:
            combined_logs.append("[autoresearch] report step failed; retaining candidate metrics from completed training/evaluation")

    (PIPE / "outputs" / "autoresearch" / "last_run.log").write_text("\n\n".join(combined_logs), encoding="utf-8")
    return 0, total_elapsed


def _metric_frame(out: Path, selection_splits: list[str], model_scope: str) -> pd.DataFrame:
    metrics = _read_csv(out / "tables" / "metrics_by_basin.csv")
    if metrics.empty:
        return metrics
    metrics = metrics[metrics["split"].isin(selection_splits)].copy()
    if model_scope != "all":
        metrics = metrics[metrics["model"].eq(model_scope)].copy()
    return metrics


def _signature_frame(out: Path, selection_splits: list[str], model_scope: str) -> pd.DataFrame:
    sig = _read_csv(out / "tables" / "signature_errors.csv")
    if sig.empty:
        return sig
    sig = sig[sig["split"].isin(selection_splits)].copy()
    if model_scope != "all":
        sig = sig[sig["model"].eq(model_scope)].copy()
    return sig


def summarize(config: str, selection_splits: list[str], model_scope: str) -> dict[str, Any]:
    out = _load_output_dir(config)
    metrics = _metric_frame(out, selection_splits, model_scope)
    if metrics.empty:
        return {}

    sig = _signature_frame(out, selection_splits, model_scope)

    def _summarize_subset(metrics_sub: pd.DataFrame, sig_sub: pd.DataFrame) -> dict[str, Any]:
        kge = pd.to_numeric(metrics_sub["kge"], errors="coerce")
        nse = pd.to_numeric(metrics_sub["nse"], errors="coerce")
        log_nse = pd.to_numeric(metrics_sub["log_nse"], errors="coerce")
        pbias = pd.to_numeric(metrics_sub["pbias"], errors="coerce")
        failures = (kge < 0) | (nse < 0)

        row: dict[str, Any] = {
            "models_seen": ",".join(sorted(metrics_sub["model"].dropna().astype(str).unique())),
            "splits_seen": ",".join(sorted(metrics_sub["split"].dropna().astype(str).unique())),
            "basins": int(metrics_sub["ID"].nunique()),
            "median_kge": float(kge.median()),
            "q25_kge": float(kge.quantile(0.25)),
            "median_nse": float(nse.median()),
            "median_log_nse": float(log_nse.median()),
            "median_abs_pbias": float(pbias.abs().median()),
            "failure_count": int(failures.sum()),
            "failure_rate": float(failures.mean()),
        }

        if not sig_sub.empty:
            for col in [
                "abs_err_runoff_ratio",
                "abs_err_fdc_slope",
                "abs_err_baseflow_index",
                "abs_err_high_q_dur",
                "abs_err_low_q_dur",
                "abs_err_q95",
                "abs_err_q05",
            ]:
                if col in sig_sub.columns:
                    row[f"median_{col}"] = float(pd.to_numeric(sig_sub[col], errors="coerce").median())

        row["composite_score"] = composite_score(row)
        row["guardrail_fail"] = guardrail_fail(row)
        return row

    if model_scope == "all":
        candidates: list[dict[str, Any]] = []
        for model_name, metrics_sub in metrics.groupby(metrics["model"].astype(str), sort=False):
            sig_sub = sig[sig["model"].astype(str).eq(str(model_name))].copy() if not sig.empty else pd.DataFrame()
            row = _summarize_subset(metrics_sub.copy(), sig_sub)
            row["best_model_in_run"] = str(model_name)
            candidates.append(row)
        if not candidates:
            return {}
        candidates.sort(key=lambda row: float(row.get("composite_score", float("-inf"))), reverse=True)
        return candidates[0]

    return _summarize_subset(metrics, sig)


def composite_score(row: dict[str, Any]) -> float:
    score = 100.0 * float(row.get("median_kge", 0.0))
    score += 25.0 * float(row.get("q25_kge", 0.0))
    score += 25.0 * float(row.get("median_log_nse", 0.0))
    score -= 0.5 * float(row.get("median_abs_pbias", 999.0))
    score -= 20.0 * float(row.get("failure_rate", 1.0))

    rr = row.get("median_abs_err_runoff_ratio")
    if rr is not None and np.isfinite(rr):
        score -= 25.0 * float(rr)

    fdc = row.get("median_abs_err_fdc_slope")
    if fdc is not None and np.isfinite(fdc):
        score -= 2.0 * min(float(fdc), 10.0)

    bfi = row.get("median_abs_err_baseflow_index")
    if bfi is not None and np.isfinite(bfi):
        score -= 20.0 * float(bfi)

    high_q_dur = row.get("median_abs_err_high_q_dur")
    if high_q_dur is not None and np.isfinite(high_q_dur):
        score -= 18.0 * float(high_q_dur)

    q95 = row.get("median_abs_err_q95")
    if q95 is not None and np.isfinite(q95):
        score -= 22.0 * float(q95)

    return float(score)


def guardrail_fail(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("median_abs_pbias", 999.0)) > 15.0
        or float(row.get("failure_rate", 1.0)) > 0.10
        or float(row.get("q25_kge", -999.0)) < 0.20
    )


def _previous_best(tier: str, model_scope: str) -> float | None:
    prev = _read_results()
    if prev.empty or "composite_score" not in prev.columns:
        return None
    needed = {"tier", "model_scope", "decision"}
    if not needed.issubset(prev.columns):
        return None
    prev = prev[
        prev["tier"].astype(str).eq(tier)
        & prev["model_scope"].astype(str).eq(model_scope)
        & prev["decision"].astype(str).isin(["keep", "promote"])
    ].copy()
    if prev.empty:
        return None
    scores = pd.to_numeric(prev["composite_score"], errors="coerce").dropna()
    return float(scores.max()) if not scores.empty else None


def decide(code: int, metrics: dict[str, Any], tier: str, model_scope: str, tolerance: float) -> str:
    if code != 0 or not metrics:
        return "reject"
    best = _previous_best(tier, model_scope)
    score = float(metrics["composite_score"])
    if bool(metrics.get("guardrail_fail", False)):
        if best is None or score > best + tolerance:
            return "quarantine"
        return "reject"
    if best is None or score > best + tolerance:
        if tier == "tier1_proxy" and (best is None or score > best + max(1.0, tolerance * 4.0)):
            return "promote"
        return "keep"
    return "reject"


def append_result(record: dict[str, Any]) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    ensure_results_schema(RESULTS)
    exists = RESULTS.exists()
    with RESULTS.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def update_memory(record: dict[str, Any]) -> None:
    MEMORY.parent.mkdir(parents=True, exist_ok=True)
    if MEMORY.exists():
        try:
            memory = json.loads(MEMORY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            memory = {}
    else:
        memory = {}

    memory.setdefault("last_result", {})
    memory.setdefault("best_by_scope", {})
    memory.setdefault("history", [])
    scope = str(record.get("model_scope") or "")
    memory["last_result"] = {
        "tag": record.get("tag"),
        "scope": record.get("model_scope"),
        "decision": record.get("decision"),
        "status": record.get("status"),
        "composite_score": record.get("composite_score"),
        "median_kge": record.get("median_kge"),
        "median_nse": record.get("median_nse"),
        "median_log_nse": record.get("median_log_nse"),
        "median_abs_pbias": record.get("median_abs_pbias"),
        "notes": record.get("notes"),
        "timestamp": record.get("timestamp"),
    }

    prev_results = _read_results()
    if not prev_results.empty:
        valid = prev_results[prev_results["decision"].astype(str).isin(["keep", "promote", "quarantine"])].copy()
        if not valid.empty:
            valid["composite_score"] = pd.to_numeric(valid["composite_score"], errors="coerce")
            valid = valid.dropna(subset=["composite_score"])
            best_by_scope: dict[str, dict[str, Any]] = {}
            for scope_name, group in valid.groupby(valid["model_scope"].astype(str)):
                best = group.sort_values("composite_score", ascending=False).iloc[0]
                best_by_scope[str(scope_name)] = {
                    "tag": best.get("tag"),
                    "config": best.get("config"),
                    "hypothesis": best.get("hypothesis"),
                    "composite_score": float(best.get("composite_score")),
                    "median_kge": best.get("median_kge"),
                    "median_nse": best.get("median_nse"),
                    "median_log_nse": best.get("median_log_nse"),
                    "median_abs_pbias": best.get("median_abs_pbias"),
                    "notes": best.get("notes"),
                    "timestamp": best.get("timestamp"),
                }
            memory["best_by_scope"] = best_by_scope

    history_entry = {
        "tag": record.get("tag"),
        "scope": scope,
        "decision": record.get("decision"),
        "status": record.get("status"),
        "parent_tag": record.get("parent_tag"),
        "parent_score": record.get("parent_score"),
        "hypothesis": record.get("hypothesis"),
        "reasoning": record.get("reasoning"),
        "composite_score": record.get("composite_score"),
        "median_kge": record.get("median_kge"),
        "median_nse": record.get("median_nse"),
        "notes": record.get("notes"),
        "timestamp": record.get("timestamp"),
    }
    memory["history"] = (memory.get("history") or [])[-24:] + [history_entry]
    MEMORY.write_text(json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--tier", default="tier1_proxy")
    parser.add_argument("--model-scope", default="all")
    parser.add_argument("--selection-splits", nargs="+", default=["val"])
    parser.add_argument("--steps", nargs="+", default=["tabular", "plots", "report"])
    parser.add_argument("--sequence-models", nargs="*", default=[])
    parser.add_argument("--sequence-device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--timeout-s", type=int)
    parser.add_argument("--changed-files", default="")
    parser.add_argument("--parent-tag", default="")
    parser.add_argument("--parent-score", default="")
    parser.add_argument("--reasoning", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--tolerance", type=float, default=0.25)
    args = parser.parse_args()

    if "test" in args.selection_splits and args.tier not in {"tier2_full_validation", "tier3_final"}:
        raise SystemExit("AutoResearch proxy selection may not use the test split.")

    code, elapsed = run_iteration(
        args.config,
        steps=args.steps,
        sequence_models=list(args.sequence_models),
        sequence_device=args.sequence_device,
        timeout_s=args.timeout_s,
    )
    metrics = summarize(args.config, args.selection_splits, args.model_scope) if code == 0 else {}
    decision = decide(code, metrics, args.tier, args.model_scope, args.tolerance)
    status = "ok" if code == 0 else "crash"

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "tier": args.tier,
        "decision": decision,
        "status": status,
        "elapsed_s": round(elapsed, 2),
        "model_scope": args.model_scope,
        "parent_tag": args.parent_tag,
        "parent_score": args.parent_score,
        "selection_splits": ",".join(args.selection_splits),
        "steps": ",".join(args.steps + [f"sequence:{m}" for m in args.sequence_models]),
        "hypothesis": args.hypothesis,
        "reasoning": args.reasoning,
        "config": args.config,
        "changed_files": args.changed_files,
        "notes": args.notes,
        **metrics,
    }
    append_result(record)
    update_memory(record)
    print(record)
    if code != 0:
        log = PIPE / "outputs" / "autoresearch" / "last_run.log"
        print(log.read_text(encoding="utf-8")[-4000:])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
