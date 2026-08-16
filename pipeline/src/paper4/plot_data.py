"""Build a compact, plot-ready data layer.

`plot_data/` sits next to `tables/` and `figures/` and is the only artifact
required to re-plot any figure without re-reading the multi-hundred-MB
predictions or the giant model frame. Everything is parquet for speed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper4.style import MODEL_ORDER, ZONE_VALUE_ORDER

# Number of representative basins to materialize hydrographs for.
N_EXAMPLE_BASINS_PER_REGIME = 3
EXAMPLE_REGIME_KEYS = (
    ("snow_class", "snow_influenced"),
    ("snow_class", "low_snow"),
    ("aridity_class", "humid"),
    ("aridity_class", "dry"),
    ("storage_class", "high_storage"),
    ("storage_class", "low_storage"),
    ("impact_class", "near_natural"),
    ("impact_class", "impact_l"),
)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _eval_split(metrics: pd.DataFrame) -> str | None:
    if metrics.empty or "split" not in metrics.columns:
        return None
    seen = set(metrics["split"].dropna().astype(str).unique())
    for split in ("test", "spatial_test", "val"):
        if split in seen:
            return split
    return None


def _select_example_basins(metrics: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Pick representative basins per regime: median KGE catchment in each regime cell."""
    if metrics.empty or attrs.empty:
        return pd.DataFrame()
    split = _eval_split(metrics)
    if split is None:
        return pd.DataFrame()
    test = metrics[metrics["split"].eq(split)]
    if test.empty:
        return pd.DataFrame()
    primary = "xgboost" if "xgboost" in test["model"].unique() else test["model"].iloc[0]
    base = test[test["model"].eq(primary)][["ID", "kge"]].merge(attrs, on="ID", how="left")
    rows = []
    for zone, value in EXAMPLE_REGIME_KEYS:
        if zone not in base.columns:
            continue
        sub = base[base[zone].astype(str).eq(value)].dropna(subset=["kge"])
        if sub.empty:
            continue
        sub = sub.sort_values("kge")
        n = len(sub)
        picks = []
        if n >= 1:
            picks.append(sub.iloc[n // 2])
        if n >= 3:
            picks.append(sub.iloc[max(0, n // 2 - n // 6)])
            picks.append(sub.iloc[min(n - 1, n // 2 + n // 6)])
        for pick in picks[:N_EXAMPLE_BASINS_PER_REGIME]:
            rows.append({
                "ID": int(pick["ID"]),
                "regime_zone": zone,
                "regime_value": value,
                "kge_xgb": float(pick["kge"]),
                "name": str(pick.get("name", "")),
                "river": str(pick.get("river", "")),
                "frac_snow": float(pick.get("frac_snow", np.nan)),
                "aridity": float(pick.get("aridity", np.nan)),
                "bfi": float(pick.get("bfi", np.nan)),
                "area_calc": float(pick.get("area_calc", np.nan)),
                "lon_3035": float(pick.get("lon", np.nan)),
                "lat_3035": float(pick.get("lat", np.nan)),
                "impact_class": str(pick.get("impact_class", "")),
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["ID", "regime_zone", "regime_value"]).reset_index(drop=True)


def _gather_predictions(out: Path, basin_ids: list[int]) -> pd.DataFrame:
    """Pull obs+predicted hydrographs for example basins from per-model prediction CSVs.

    Result is long format with columns [ID, date, split, model, q_obs, q_sim].
    Only the example basins are kept, so the file is tiny.
    """
    tables = out / "tables"
    parts = []
    for fname, model_label in [
        ("predictions_tabular.csv", None),
        ("predictions_cosero.csv", "cosero"),
        ("predictions_xlstm.csv", "xlstm"),
        ("predictions_transformer.csv", "transformer"),
    ]:
        path = tables / fname
        if not path.exists():
            continue
        try:
            dt = pd.read_csv(path, usecols=lambda c: c in {"ID", "date", "split", "q_mm_day", "q_pred_mm_day", "model"})
        except pd.errors.EmptyDataError:
            continue
        dt = dt[dt["ID"].isin(basin_ids)].copy()
        if model_label is not None and "model" not in dt.columns:
            dt["model"] = model_label
        if dt.empty:
            continue
        parts.append(dt.rename(columns={"q_mm_day": "q_obs", "q_pred_mm_day": "q_sim"}))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def _basin_geo(idx: pd.DataFrame) -> pd.DataFrame:
    """Convert ETRS89/LAEA Europe (EPSG:3035) to lon/lat for plotting."""
    if idx.empty:
        return idx
    if not {"lon", "lat"}.issubset(idx.columns):
        return pd.DataFrame()
    try:
        from pyproj import Transformer
    except Exception:
        return pd.DataFrame()
    trans = Transformer.from_crs(3035, 4326, always_xy=True)
    lons, lats = trans.transform(idx["lon"].to_numpy(dtype=float), idx["lat"].to_numpy(dtype=float))
    out = pd.DataFrame({
        "ID": idx["ID"].astype(int).values,
        "x_3035": idx["lon"].astype(float).values,
        "y_3035": idx["lat"].astype(float).values,
        "lon_deg": lons,
        "lat_deg": lats,
    })
    return out


def _melt_signatures(sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return sig
    obs_cols = [c for c in sig.columns if c.startswith("obs_")]
    long = []
    for ocol in obs_cols:
        sig_name = ocol[len("obs_"):]
        scol = f"sim_{sig_name}"
        ecol = f"abs_err_{sig_name}"
        if scol not in sig.columns:
            continue
        chunk = sig[["model", "split", "ID", ocol, scol]].copy()
        if ecol in sig.columns:
            chunk[ecol] = sig[ecol]
        chunk = chunk.rename(columns={ocol: "obs", scol: "sim"})
        chunk["signature"] = sig_name
        long.append(chunk)
    return pd.concat(long, ignore_index=True) if long else pd.DataFrame()


def export_plot_data(out: Path) -> None:
    pd_dir = out / "plot_data"
    pd_dir.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"

    metrics = _read(tables / "metrics_by_basin.csv")
    sig = _read(tables / "signature_errors.csv")
    idx = _read(tables / "catchment_index.csv")
    attrs = _read(tables / "catchment_process_attributes.csv")
    deltas = _read(tables / "model_delta_diagnostics.csv")
    zones = _read(tables / "process_zone_summary.csv")
    ranks = _read(tables / "signature_model_ranking.csv")
    flood_events = _read(tables / "flood_events.csv")
    flood_skill = _read(tables / "flood_event_model_skill.csv")
    flood_summary = _read(tables / "flood_typology_summary.csv")
    flood_model_summary = _read(tables / "flood_typology_model_summary.csv")

    # 1. Basin skill (long format)
    if not metrics.empty:
        metrics.to_parquet(pd_dir / "basin_skill_long.parquet", index=False)

    # 2. Signature obs vs sim (long, one row per basin x signature x model x split)
    sig_long = _melt_signatures(sig)
    if not sig_long.empty:
        sig_long.to_parquet(pd_dir / "signature_obs_sim_long.parquet", index=False)

    # 3. Process attributes + impact classes joined to lon/lat
    geo = _basin_geo(idx)
    if not attrs.empty:
        joined = attrs.copy()
        if not geo.empty:
            joined = joined.merge(geo, on="ID", how="left")
        joined.to_parquet(pd_dir / "basin_attributes_geo.parquet", index=False)

    # 4. Process zone summary (already long); copy as parquet
    if not zones.empty:
        zones.to_parquet(pd_dir / "process_zone_summary.parquet", index=False)
    if not ranks.empty:
        ranks.to_parquet(pd_dir / "signature_model_ranking.parquet", index=False)
    if not deltas.empty:
        deltas.to_parquet(pd_dir / "model_delta_diagnostics.parquet", index=False)
    if not flood_events.empty:
        flood_events.to_parquet(pd_dir / "flood_events.parquet", index=False)
    if not flood_skill.empty:
        flood_skill.to_parquet(pd_dir / "flood_event_model_skill.parquet", index=False)
    if not flood_summary.empty:
        flood_summary.to_parquet(pd_dir / "flood_typology_summary.parquet", index=False)
    if not flood_model_summary.empty:
        flood_model_summary.to_parquet(pd_dir / "flood_typology_model_summary.parquet", index=False)

    # 5. Example basin selection + their hydrographs (only the chosen subset, tiny)
    examples = _select_example_basins(metrics, attrs)
    if not examples.empty:
        examples.to_parquet(pd_dir / "example_basins.parquet", index=False)
        hydrographs = _gather_predictions(out, examples["ID"].tolist())
        if not hydrographs.empty:
            hydrographs.to_parquet(pd_dir / "example_hydrographs_long.parquet", index=False)
        # Pull raw forcings (precip especially) for the example basins from the
        # full model frame, but keep only the columns we need so the file stays small.
        frame_path = out / "tables" / "model_frame.parquet"
        if frame_path.exists():
            try:
                wanted = ["ID", "date", "prec", "swe", "2m_temp_mean", "q_mm_day"]
                raw = pd.read_parquet(frame_path, columns=[c for c in wanted if True])
                raw = raw[raw["ID"].isin(examples["ID"].astype(int).tolist())].copy()
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
                raw = raw.dropna(subset=["date"])
                raw.to_parquet(pd_dir / "example_raw_long.parquet", index=False)
            except Exception:
                pass

    # 6. Skill vs attribute (test split, primary attributes) — flat helper for fast plots
    if not metrics.empty and not attrs.empty:
        split = _eval_split(metrics)
        test = metrics[metrics["split"].eq(split)][["model", "ID", "kge", "nse", "log_nse", "pbias"]].copy() if split else pd.DataFrame()
        if not test.empty:
            wide = test.merge(
                attrs[["ID", "frac_snow", "aridity", "bfi", "area_calc", "soil_condu", "geol_poros",
                       "forest_fra", "urban_fra", "is_impacted", "snow_class", "aridity_class",
                       "storage_class", "impact_class", "size_class"]],
                on="ID",
                how="left",
            )
            wide.to_parquet(pd_dir / "skill_attributes_test.parquet", index=False)
