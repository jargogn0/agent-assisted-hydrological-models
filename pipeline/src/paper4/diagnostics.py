from __future__ import annotations

from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from paper4.flood_typology import generate_flood_typology


PROCESS_ATTRS = [
    "area_calc",
    "p_mean",
    "et0_mean",
    "arid_1",
    "arid_2",
    "frac_snow",
    "forest_fra",
    "urban_fra",
    "bedrk_dep",
    "root_dep",
    "soil_condu",
    "soil_tawc",
    "geol_perme",
    "geol_poros",
    "h1981_baseflow_index_ladson",
    "h1981_baseflow_index_lfstat",
    "h1981_slope_fdc",
    "h1981_hfd_mean",
    "h1981_high_q_freq",
    "h1981_low_q_freq",
    "has_transfer",
    "is_impacted",
]


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _qclass(s: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    x = _numeric(s)
    if x.notna().sum() < 6 or x.nunique(dropna=True) < 3:
        return pd.Series(["unknown"] * len(s), index=s.index)
    q1, q2 = x.quantile([1 / 3, 2 / 3])
    out = pd.Series(labels[1], index=s.index, dtype="object")
    out[x <= q1] = labels[0]
    out[x >= q2] = labels[2]
    out[x.isna()] = "unknown"
    return out


def _impact_class(row: pd.Series) -> str:
    if int(row.get("has_transfer", 0) or 0) == 1:
        return "transfer"
    deg = str(row.get("degimpact", "-")).strip().lower()
    typ = str(row.get("typimpact", "-")).strip()
    if typ != "-" and typ:
        return f"impact_{deg}" if deg and deg != "-" else "impact_unknown"
    return "near_natural"


def process_attributes(idx: pd.DataFrame) -> pd.DataFrame:
    cols = ["ID", "degimpact", "typimpact", "lon", "lat"] + [c for c in PROCESS_ATTRS if c in idx.columns]
    attrs = idx[[c for c in cols if c in idx.columns]].copy()
    for c in attrs.columns:
        if c not in {"ID", "degimpact", "typimpact"}:
            attrs[c] = _numeric(attrs[c])

    if "h1981_baseflow_index_ladson" in attrs.columns:
        attrs["bfi"] = attrs["h1981_baseflow_index_ladson"]
    elif "h1981_baseflow_index_lfstat" in attrs.columns:
        attrs["bfi"] = attrs["h1981_baseflow_index_lfstat"]
    else:
        attrs["bfi"] = np.nan

    attrs["fdc_slope"] = attrs.get("h1981_slope_fdc", np.nan)
    attrs["aridity"] = attrs.get("arid_1", attrs.get("arid_2", np.nan))
    attrs["impact_class"] = attrs.apply(_impact_class, axis=1)
    attrs["snow_class"] = _qclass(attrs.get("frac_snow", pd.Series(np.nan, index=attrs.index)), ("low_snow", "mixed_snow", "snow_influenced"))
    attrs["aridity_class"] = _qclass(attrs["aridity"], ("humid", "moderate_aridity", "dry"))
    attrs["storage_class"] = _qclass(attrs["bfi"], ("low_storage", "mixed_storage", "high_storage"))
    attrs["geology_class"] = _qclass(attrs.get("geol_perme", pd.Series(np.nan, index=attrs.index)), ("low_perm", "mixed_perm", "high_perm"))
    attrs["size_class"] = _qclass(attrs.get("area_calc", pd.Series(np.nan, index=attrs.index)), ("small", "medium", "large"))
    return attrs


def _wide_metrics(metrics: pd.DataFrame, value: str) -> pd.DataFrame:
    return metrics.pivot_table(index=["split", "ID"], columns="model", values=value, aggfunc="median").reset_index()


def model_deltas(metrics: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    wide_kge = _wide_metrics(metrics, "kge")
    wide_nse = _wide_metrics(metrics, "nse")
    wide_pbias = _wide_metrics(metrics, "pbias")
    out = wide_kge[["split", "ID"]].copy()

    def col(wide: pd.DataFrame, name: str) -> pd.Series:
        return wide[name] if name in wide.columns else pd.Series(np.nan, index=wide.index)

    out["kge_cosero"] = col(wide_kge, "cosero")
    out["kge_rf"] = col(wide_kge, "random_forest")
    out["kge_xgb"] = col(wide_kge, "xgboost")
    out["kge_xlstm"] = col(wide_kge, "xlstm")
    out["kge_transformer"] = col(wide_kge, "transformer")
    out["nse_cosero"] = col(wide_nse, "cosero")
    out["nse_rf"] = col(wide_nse, "random_forest")
    out["nse_xgb"] = col(wide_nse, "xgboost")
    out["nse_xlstm"] = col(wide_nse, "xlstm")
    out["nse_transformer"] = col(wide_nse, "transformer")
    out["pbias_cosero"] = col(wide_pbias, "cosero")
    out["pbias_rf"] = col(wide_pbias, "random_forest")
    out["pbias_xgb"] = col(wide_pbias, "xgboost")
    out["pbias_xlstm"] = col(wide_pbias, "xlstm")
    out["pbias_transformer"] = col(wide_pbias, "transformer")

    out["best_tree_kge"] = out[["kge_rf", "kge_xgb"]].max(axis=1)
    out["best_sequence_kge"] = out[["kge_xlstm", "kge_transformer"]].max(axis=1)
    out["memory_gain_xlstm_vs_rf"] = out["kge_xlstm"] - out["kge_rf"]
    out["transformer_gain_vs_rf"] = out["kge_transformer"] - out["kge_rf"]
    out["sequence_gain_vs_tree"] = out["best_sequence_kge"] - out["best_tree_kge"]
    out["tree_advantage"] = out["best_tree_kge"] - out["best_sequence_kge"]
    out["xlstm_minus_transformer"] = out["kge_xlstm"] - out["kge_transformer"]
    out["xgboost_gain_vs_cosero"] = out["kge_xgb"] - out["kge_cosero"]
    out["xlstm_gain_vs_cosero"] = out["kge_xlstm"] - out["kge_cosero"]
    out["transformer_gain_vs_cosero"] = out["kge_transformer"] - out["kge_cosero"]
    return out.merge(attrs, on="ID", how="left")


def process_zone_summary(metrics: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or attrs.empty:
        return pd.DataFrame()
    merged = metrics.merge(attrs, on="ID", how="left")
    zones = ["snow_class", "aridity_class", "storage_class", "geology_class", "impact_class", "size_class"]
    rows = []
    for zone in zones:
        if zone not in merged.columns:
            continue
        g = (
            merged.groupby(["split", "model", zone])
            .agg(
                basins=("ID", "nunique"),
                median_kge=("kge", "median"),
                median_nse=("nse", "median"),
                median_log_nse=("log_nse", "median"),
                median_pbias=("pbias", "median"),
            )
            .reset_index()
            .rename(columns={zone: "zone_value"})
        )
        g.insert(2, "zone", zone)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def signature_model_ranking(sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame()
    cols = [c for c in sig.columns if c.startswith("abs_err_")]
    rows = []
    for split, g in sig.groupby("split"):
        for col in cols:
            tmp = (
                g.groupby("model")[col]
                .median()
                .sort_values()
                .reset_index()
                .rename(columns={col: "median_abs_error"})
            )
            tmp["rank"] = np.arange(1, len(tmp) + 1)
            tmp["split"] = split
            tmp["signature"] = col.replace("abs_err_", "")
            rows.append(tmp[["split", "signature", "rank", "model", "median_abs_error"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def process_correlations(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    attrs = [
        "area_calc",
        "aridity",
        "frac_snow",
        "bfi",
        "fdc_slope",
        "geol_perme",
        "geol_poros",
        "soil_condu",
        "soil_tawc",
        "forest_fra",
        "urban_fra",
        "has_transfer",
        "is_impacted",
    ]
    targets = [
        "memory_gain_xlstm_vs_rf",
        "transformer_gain_vs_rf",
        "sequence_gain_vs_tree",
        "tree_advantage",
        "xgboost_gain_vs_cosero",
        "xlstm_gain_vs_cosero",
        "transformer_gain_vs_cosero",
        "kge_cosero",
        "kge_rf",
        "kge_xgb",
        "kge_xlstm",
        "kge_transformer",
        "pbias_cosero",
        "pbias_rf",
        "pbias_xgb",
        "pbias_xlstm",
        "pbias_transformer",
    ]
    rows = []
    for split, g in deltas.groupby("split"):
        for attr in attrs:
            if attr not in g.columns:
                continue
            for target in targets:
                if target not in g.columns:
                    continue
                pair = g[[attr, target]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) < 8 or pair[attr].nunique() < 2 or pair[target].nunique() < 2:
                    continue
                rho = pair[attr].corr(pair[target], method="spearman")
                rows.append({"split": split, "attribute": attr, "target": target, "spearman_rho": rho, "n": len(pair)})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False) if rows else pd.DataFrame()


def _bootstrap_median(values: pd.Series, *, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {"n": 0, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    point = float(np.median(x))
    if len(x) == 1:
        return {"n": 1, "median": point, "ci_low": point, "ci_high": point}
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(int(n_boot), len(x)), replace=True)
    boot = np.median(draws, axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(x), "median": point, "ci_low": float(lo), "ci_high": float(hi)}


def _seed(seed: int, *parts: object) -> int:
    label = "|".join(map(str, parts)).encode("utf-8")
    return int(seed + zlib.crc32(label) % 100000)


def metric_uncertainty(metrics: pd.DataFrame, *, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap median metric intervals by model/split across catchments."""
    if metrics.empty:
        return pd.DataFrame()
    metric_cols = [c for c in ["kge", "nse", "log_nse", "pbias", "rmse", "mae"] if c in metrics.columns]
    rows = []
    for (split, model), g in metrics.groupby(["split", "model"], dropna=False):
        for metric in metric_cols:
            ci = _bootstrap_median(g[metric], n_boot=n_boot, seed=_seed(seed, split, model, metric))
            rows.append({
                "split": split,
                "model": model,
                "metric": metric,
                "n_basins": ci["n"],
                "median": ci["median"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "ci_level": 0.95,
                "method": "basin_bootstrap_median",
            })
    return pd.DataFrame(rows)


def signature_uncertainty(sig: pd.DataFrame, *, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap median absolute signature-error intervals by model/split."""
    if sig.empty:
        return pd.DataFrame()
    cols = [c for c in sig.columns if c.startswith("abs_err_")]
    rows = []
    for (split, model), g in sig.groupby(["split", "model"], dropna=False):
        for col in cols:
            ci = _bootstrap_median(g[col], n_boot=n_boot, seed=_seed(seed, split, model, col))
            rows.append({
                "split": split,
                "model": model,
                "signature": col.replace("abs_err_", ""),
                "n_basins": ci["n"],
                "median_abs_error": ci["median"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "ci_level": 0.95,
                "method": "basin_bootstrap_median",
            })
    return pd.DataFrame(rows)


def contrast_uncertainty(deltas: pd.DataFrame, *, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap paired model-contrast intervals using basin-level deltas."""
    if deltas.empty:
        return pd.DataFrame()
    contrasts = [
        ("xgboost_gain_vs_cosero", "XGBoost - COSERO KGE"),
        ("xlstm_gain_vs_cosero", "xLSTM - COSERO KGE"),
        ("transformer_gain_vs_cosero", "Transformer - COSERO KGE"),
        ("memory_gain_xlstm_vs_rf", "xLSTM - Random Forest KGE"),
        ("transformer_gain_vs_rf", "Transformer - Random Forest KGE"),
        ("sequence_gain_vs_tree", "Best sequence - best tree KGE"),
        ("tree_advantage", "Best tree - best sequence KGE"),
        ("xlstm_minus_transformer", "xLSTM - Transformer KGE"),
    ]
    rows = []
    for split, g in deltas.groupby("split", dropna=False):
        for col, label in contrasts:
            if col not in g.columns:
                continue
            x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) == 0:
                continue
            ci = _bootstrap_median(pd.Series(x), n_boot=n_boot, seed=_seed(seed, split, col))
            if len(x) == 1:
                win_low = win_high = win_rate = float(x[0] > 0)
            else:
                rng = np.random.default_rng(_seed(seed, "win", split, col))
                draws = rng.choice(x, size=(int(n_boot), len(x)), replace=True)
                boot_win = (draws > 0).mean(axis=1)
                win_rate = float((x > 0).mean())
                win_low, win_high = np.percentile(boot_win, [2.5, 97.5])
            rows.append({
                "split": split,
                "contrast": col,
                "label": label,
                "n_basins": ci["n"],
                "median_delta": ci["median"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "win_rate": win_rate,
                "win_rate_ci_low": float(win_low),
                "win_rate_ci_high": float(win_high),
                "ci_level": 0.95,
                "method": "paired_basin_bootstrap",
            })
    return pd.DataFrame(rows)


def spatial_fold_summaries(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise per-fold median skill for aggregated spatial-transfer outputs."""
    if metrics.empty or "spatial_fold" not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [c for c in ["kge", "nse", "log_nse", "pbias"] if c in metrics.columns]
    fold_metrics = (
        metrics.groupby(["spatial_fold", "split", "model"], as_index=False)
        .agg(
            basins=("ID", "nunique"),
            **{f"median_{c}": (c, "median") for c in metric_cols},
        )
    )
    rows = []
    for (split, model), g in fold_metrics.groupby(["split", "model"], dropna=False):
        row = {
            "split": split,
            "model": model,
            "n_folds": int(g["spatial_fold"].nunique()),
            "total_heldout_basins": int(g["basins"].sum()),
        }
        for metric in metric_cols:
            col = f"median_{metric}"
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"fold_mean_median_{metric}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"fold_sd_median_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"fold_min_median_{metric}"] = float(vals.min()) if len(vals) else np.nan
            row[f"fold_max_median_{metric}"] = float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return fold_metrics, pd.DataFrame(rows)


def _preferred_eval_split(df: pd.DataFrame) -> str:
    seen = set(df["split"].dropna().astype(str).unique()) if "split" in df.columns else set()
    for split in ("test", "spatial_test", "val"):
        if split in seen:
            return split
    return "test"


def write_interpretation(out_dir: Path, metrics: pd.DataFrame, deltas: pd.DataFrame, zone_summary: pd.DataFrame, signature_rank: pd.DataFrame):
    path = out_dir / "tables" / "hydrologic_findings.md"
    if metrics.empty:
        path.write_text("_No metrics available._\n", encoding="utf-8")
        return

    eval_split = _preferred_eval_split(metrics)
    test = metrics[metrics["split"].eq(eval_split)]
    med = test.groupby("model")[["kge", "nse", "log_nse", "pbias"]].median().sort_values("kge", ascending=False)
    lines = ["# Hydrologic Diagnostic Findings", ""]
    lines.append("## Hydrograph Skill")
    for model, row in med.iterrows():
        lines.append(
            f"- `{model}`: median {eval_split} KGE={row['kge']:.3f}, NSE={row['nse']:.3f}, "
            f"logNSE={row['log_nse']:.3f}, PBIAS={row['pbias']:.1f}%."
        )

    if not deltas.empty:
        dtest = deltas[deltas["split"].eq(eval_split)]
        lines += ["", "## Diagnostic Model Contrasts"]
        if "xgboost_gain_vs_cosero" in dtest.columns:
            lines.append(f"- Median XGBoost gain over COSERO: {dtest['xgboost_gain_vs_cosero'].median():.3f} KGE.")
        lines.append(f"- Median xLSTM memory gain over RF: {dtest['memory_gain_xlstm_vs_rf'].median():.3f} KGE.")
        lines.append(f"- Median Transformer gain over RF: {dtest['transformer_gain_vs_rf'].median():.3f} KGE.")
        lines.append(f"- Median best-sequence gain over best-tree model: {dtest['sequence_gain_vs_tree'].median():.3f} KGE.")

    if not signature_rank.empty:
        best = signature_rank[(signature_rank["split"].eq(eval_split)) & (signature_rank["rank"].eq(1))]
        lines += ["", "## Signature Winners"]
        for _, row in best.sort_values("signature").iterrows():
            lines.append(f"- `{row['signature']}` is best preserved by `{row['model']}` (median abs. error {row['median_abs_error']:.3g}).")

    if not zone_summary.empty:
        lines += ["", "## Process-Zone Tables"]
        lines.append("See `process_zone_summary.csv` for model skill stratified by snow, aridity, storage, geology, human impact, and catchment size.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_diagnostics(out_dir: Path) -> None:
    tables = out_dir / "tables"
    metrics_path = tables / "metrics_by_basin.csv"
    sig_path = tables / "signature_errors.csv"
    idx_path = tables / "catchment_index.csv"
    if not metrics_path.exists() or not idx_path.exists():
        return

    metrics = pd.read_csv(metrics_path)
    idx = pd.read_csv(idx_path)
    sig = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()

    attrs = process_attributes(idx)
    attrs.to_csv(tables / "catchment_process_attributes.csv", index=False)
    deltas = model_deltas(metrics, attrs)
    deltas.to_csv(tables / "model_delta_diagnostics.csv", index=False)
    zones = process_zone_summary(metrics, attrs)
    zones.to_csv(tables / "process_zone_summary.csv", index=False)
    ranks = signature_model_ranking(sig)
    ranks.to_csv(tables / "signature_model_ranking.csv", index=False)
    corr = process_correlations(deltas)
    corr.to_csv(tables / "process_control_correlations.csv", index=False)
    metric_uncertainty(metrics).to_csv(tables / "metric_uncertainty.csv", index=False)
    contrast_uncertainty(deltas).to_csv(tables / "model_contrast_uncertainty.csv", index=False)
    signature_uncertainty(sig).to_csv(tables / "signature_error_uncertainty.csv", index=False)
    fold_metrics, fold_summary = spatial_fold_summaries(metrics)
    if not fold_metrics.empty:
        fold_metrics.to_csv(tables / "spatial_fold_metrics.csv", index=False)
    if not fold_summary.empty:
        fold_summary.to_csv(tables / "spatial_fold_summary.csv", index=False)
    generate_flood_typology(out_dir)
    write_interpretation(out_dir, metrics, deltas, zones, ranks)
