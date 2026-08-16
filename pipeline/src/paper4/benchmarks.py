from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper4.data import read_lamah_csv
from paper4.metrics import evaluate_predictions


def _cosero_qsim_col(cfg: dict) -> str:
    basin_set = str(cfg.get("data", {}).get("basin_set", "A")).upper()
    if basin_set == "A":
        return "Qsim_A"
    if basin_set in {"B", "C"}:
        return "Qsim_B"
    return "Qsim_A"


def _cosero_file(root: Path, basin_id: int) -> Path:
    return root / "F_hydrol_model" / "2_timeseries" / f"ID_{int(basin_id)}.csv"


def _q_m3s_to_mm_day(q_m3s: pd.Series, area_km2: pd.Series) -> pd.Series:
    q = pd.to_numeric(q_m3s, errors="coerce").replace(-999, np.nan)
    area = pd.to_numeric(area_km2, errors="coerce")
    return q * 86400.0 / (area * 1_000_000.0) * 1000.0


def build_cosero_predictions(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return LamaH COSERO predictions in the same schema as learned models."""
    root = Path(cfg["data"]["root"])
    qsim_col = _cosero_qsim_col(cfg)
    pred_splits = cfg.get("evaluation", {}).get("prediction_splits", ["val", "test", "spatial_test"])
    base_cols = ["ID", "date", "YYYY", "MM", "DD", "DOY", "water_year", "split", "q_mm_day", "prec", "area_km2"]
    base = frame.loc[frame["split"].isin(pred_splits), [c for c in base_cols if c in frame.columns]].copy()
    if base.empty:
        return pd.DataFrame()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    parts = []
    ids = sorted(int(i) for i in base["ID"].dropna().unique())
    for k, basin_id in enumerate(ids, start=1):
        path = _cosero_file(root, basin_id)
        if not path.exists():
            continue
        try:
            c = read_lamah_csv(path, usecols=["YYYY", "MM", "DD", qsim_col])
        except ValueError:
            continue
        c["date"] = pd.to_datetime(dict(year=c["YYYY"], month=c["MM"], day=c["DD"]), errors="coerce")
        c["ID"] = basin_id
        c = c[["ID", "date", qsim_col]]
        sub = base[base["ID"].eq(basin_id)].merge(c, on=["ID", "date"], how="left")
        if sub.empty:
            continue
        sub["model"] = "cosero"
        sub["q_pred_mm_day"] = _q_m3s_to_mm_day(sub[qsim_col], sub["area_km2"])
        sub["q_pred_mm_day"] = sub["q_pred_mm_day"].clip(lower=0)
        parts.append(sub.drop(columns=[qsim_col, "area_km2"], errors="ignore"))
        if k % 100 == 0:
            print(f"[paper4] COSERO benchmark loaded {k}/{len(ids)} basins", flush=True)
    if not parts:
        return pd.DataFrame()
    preds = pd.concat(parts, ignore_index=True)
    return preds.dropna(subset=["date", "q_mm_day", "q_pred_mm_day"])


def import_cosero_benchmark(cfg: dict, out_dir: Path) -> None:
    frame_path = out_dir / "tables" / "model_frame.parquet"
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing model frame: {frame_path}")
    frame = pd.read_parquet(frame_path)
    preds = build_cosero_predictions(frame, cfg)
    if preds.empty:
        print("[paper4] COSERO benchmark skipped: no matching simulations found", flush=True)
        return

    pred_path = out_dir / "tables" / "predictions_cosero.csv"
    preds.to_csv(pred_path, index=False)
    metrics, sigs = evaluate_predictions(preds)

    metrics_path = out_dir / "tables" / "metrics_by_basin.csv"
    sig_path = out_dir / "tables" / "signature_errors.csv"
    old_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    old_sigs = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    if not old_metrics.empty:
        old_metrics = old_metrics[old_metrics["model"].astype(str) != "cosero"]
    if not old_sigs.empty:
        old_sigs = old_sigs[old_sigs["model"].astype(str) != "cosero"]
    pd.concat([old_metrics, metrics], ignore_index=True).to_csv(metrics_path, index=False)
    pd.concat([old_sigs, sigs], ignore_index=True).to_csv(sig_path, index=False)
    print(f"[paper4] COSERO benchmark complete: {len(preds):,} daily predictions", flush=True)
