from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def nse(obs, sim) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom <= EPS:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def kge(obs, sim) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1] if np.std(obs) > EPS and np.std(sim) > EPS else np.nan
    alpha = np.std(sim) / (np.std(obs) + EPS)
    beta = np.mean(sim) / (np.mean(obs) + EPS)
    if not np.isfinite(r):
        return np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(obs, sim) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def mae(obs, sim) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(obs[mask] - sim[mask])))


def pbias(obs, sim) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0 or abs(np.sum(obs[mask])) <= EPS:
        return np.nan
    return float(100 * np.sum(sim[mask] - obs[mask]) / np.sum(obs[mask]))


def hydrograph_metrics(obs, sim) -> dict[str, float]:
    return {
        "nse": nse(obs, sim),
        "kge": kge(obs, sim),
        "rmse": rmse(obs, sim),
        "mae": mae(obs, sim),
        "pbias": pbias(obs, sim),
        "log_nse": nse(np.log1p(obs), np.log1p(np.maximum(sim, 0))),
    }


def half_flow_date(df: pd.DataFrame, q_col: str) -> float:
    vals = []
    for _, g in df.dropna(subset=[q_col]).groupby("water_year"):
        if len(g) < 180 or g[q_col].sum() <= EPS:
            continue
        ordered = g.sort_values("date")
        csum = ordered[q_col].cumsum()
        idx = int((csum >= 0.5 * ordered[q_col].sum()).idxmax())
        vals.append(float(ordered.loc[idx, "DOY"]))
    return float(np.mean(vals)) if vals else np.nan


def fdc_slope(q) -> float:
    q = np.asarray(q, dtype=float)
    q = q[np.isfinite(q) & (q > 0)]
    if len(q) < 30:
        return np.nan
    q33 = np.quantile(q, 0.67)
    q66 = np.quantile(q, 0.34)
    if q33 <= 0 or q66 <= 0:
        return np.nan
    return float((np.log(q33) - np.log(q66)) / (0.66 - 0.33))


def simple_baseflow_index(q) -> float:
    q = np.asarray(q, dtype=float)
    q = q[np.isfinite(q)]
    if len(q) < 30 or np.sum(q) <= EPS:
        return np.nan
    alpha = 0.925
    coeff = (1.0 + alpha) / 2.0

    def _forward(q_in):
        qf = np.zeros_like(q_in)
        for i in range(1, len(q_in)):
            qf[i] = alpha * qf[i - 1] + coeff * (q_in[i] - q_in[i - 1])
            if qf[i] < 0.0:
                qf[i] = 0.0
            if qf[i] > q_in[i]:
                qf[i] = q_in[i]
        return qf

    b1 = q - _forward(q)
    b2 = b1 - _forward(b1[::-1])[::-1]
    base = np.clip(b2 - _forward(b2), 0.0, q)
    return float(np.sum(base) / (np.sum(q) + EPS))


def run_lengths(mask: np.ndarray) -> list[int]:
    lengths = []
    current = 0
    for v in mask:
        if v:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def hydrological_signatures(df: pd.DataFrame, q_col: str, prec_col: str = "prec") -> dict[str, float]:
    g = df.dropna(subset=[q_col]).sort_values("date")
    q = g[q_col].to_numpy(dtype=float)
    if len(q) < 30:
        return {}
    high_thr = np.quantile(q, 0.95)
    low_thr = np.quantile(q, 0.05)
    high_mask = q >= high_thr
    low_mask = q <= low_thr
    high_runs = run_lengths(high_mask)
    low_runs = run_lengths(low_mask)
    p_sum = float(g[prec_col].sum()) if prec_col in g.columns else np.nan
    return {
        "q_mean": float(np.mean(q)),
        "runoff_ratio": float(np.sum(q) / (p_sum + EPS)) if np.isfinite(p_sum) and p_sum > 0 else np.nan,
        "fdc_slope": fdc_slope(q),
        "baseflow_index": simple_baseflow_index(q),
        "hfd_mean": half_flow_date(g, q_col),
        "q05": float(np.quantile(q, 0.05)),
        "q95": float(np.quantile(q, 0.95)),
        "high_q_freq": float(np.mean(high_mask)),
        "high_q_dur": float(np.mean(high_runs)) if high_runs else 0.0,
        "low_q_freq": float(np.mean(low_mask)),
        "low_q_dur": float(np.mean(low_runs)) if low_runs else 0.0,
        "zero_q_freq": float(np.mean(q <= EPS)),
    }


def evaluate_predictions(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    sig_rows = []
    for (model, split, basin_id), g in preds.groupby(["model", "split", "ID"]):
        if split == "unused":
            continue
        metrics = hydrograph_metrics(g["q_mm_day"], g["q_pred_mm_day"])
        metric_rows.append({"model": model, "split": split, "ID": basin_id, **metrics, "n": len(g)})
        obs_sig = hydrological_signatures(g, "q_mm_day")
        sim = g.copy()
        sim["q_sim"] = sim["q_pred_mm_day"]
        sim_sig = hydrological_signatures(sim, "q_sim")
        row = {"model": model, "split": split, "ID": basin_id}
        for key, obs_val in obs_sig.items():
            sim_val = sim_sig.get(key, np.nan)
            row[f"obs_{key}"] = obs_val
            row[f"sim_{key}"] = sim_val
            row[f"err_{key}"] = sim_val - obs_val if np.isfinite(obs_val) and np.isfinite(sim_val) else np.nan
            row[f"abs_err_{key}"] = abs(row[f"err_{key}"]) if np.isfinite(row[f"err_{key}"]) else np.nan
        sig_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(sig_rows)
