from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EVENT_SPLITS = ("val", "test", "spatial_test")
EPS = 1e-9


FRAME_COLUMNS = [
    "ID",
    "date",
    "split",
    "q_mm_day",
    "prec",
    "prec_sum_3",
    "prec_sum_7",
    "prec_sum_30",
    "2m_temp_mean",
    "2m_temp_mean_mean_7",
    "2m_temp_mean_mean_30",
    "swe",
    "swe_lag_7",
    "swe_mean_7",
    "swe_delta_7",
    "volsw_123_mean_7",
    "volsw_123_mean_30",
    "frac_snow",
    "arid_1",
    "arid_2",
    "p_season",
    "area_calc",
]


PRED_COLUMNS = {"ID", "date", "split", "model", "q_mm_day", "q_pred_mm_day"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return FRAME_COLUMNS


def _outputs_current(outputs: list[Path], inputs: list[Path]) -> bool:
    if not outputs or any(not p.exists() for p in outputs):
        return False
    existing_inputs = [p for p in inputs if p.exists()]
    if not existing_inputs:
        return False
    newest_input = max(p.stat().st_mtime for p in existing_inputs)
    oldest_output = min(p.stat().st_mtime for p in outputs)
    return oldest_output >= newest_input


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _climate_sensitivity(event_type: str) -> str:
    if event_type in {"rain_on_snow", "snowmelt"}:
        return "warming_snow_transition"
    if event_type == "warm_season_short_rain":
        return "warm_season_rainfall_intensity"
    if event_type in {"long_rain_wet_soil", "storage_release"}:
        return "antecedent_wetness_storage"
    if event_type == "cool_season_rain":
        return "cool_season_rainfall"
    return "mixed_rainfall_runoff"


def _safe_get(row: pd.Series, key: str, default=np.nan) -> float:
    val = row.get(key, default)
    try:
        return float(val)
    except Exception:
        return float("nan")


def _frame_with_columns(frame_path: Path) -> pd.DataFrame:
    cols = [c for c in FRAME_COLUMNS if c in set(_parquet_columns(frame_path))]
    frame = pd.read_parquet(frame_path, columns=cols)
    for col in FRAME_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ID"] = pd.to_numeric(frame["ID"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["ID", "date", "split", "q_mm_day"])
    frame["ID"] = frame["ID"].astype(int)
    for col in [c for c in frame.columns if c not in {"date", "split"}]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _basin_climatology(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["split"].eq("train")].copy()
    if train.empty:
        train = frame.copy()
    agg = (
        train.groupby("ID")
        .agg(
            q95=("q_mm_day", lambda s: float(np.nanquantile(s, 0.95))),
            p3_q90=("prec_sum_3", lambda s: float(np.nanquantile(s, 0.90))),
            p7_q80=("prec_sum_7", lambda s: float(np.nanquantile(s, 0.80))),
            p30_q75=("prec_sum_30", lambda s: float(np.nanquantile(s, 0.75))),
            swe_q75=("swe_lag_7", lambda s: float(np.nanquantile(s, 0.75))),
            melt_q20=("swe_delta_7", lambda s: float(np.nanquantile(s, 0.20))),
            soil_q75=("volsw_123_mean_7", lambda s: float(np.nanquantile(s, 0.75))),
            temp7_median=("2m_temp_mean_mean_7", "median"),
        )
        .reset_index()
    )
    return agg.replace([np.inf, -np.inf], np.nan)


def _classify_event(row: pd.Series, clim: pd.Series) -> tuple[str, dict[str, bool]]:
    month = int(row["peak_date"].month)
    p3 = _safe_get(row, "p3")
    p7 = _safe_get(row, "p7")
    p30 = _safe_get(row, "p30")
    temp7 = _safe_get(row, "temp7")
    swe_before = _safe_get(row, "swe_before")
    swe_delta = _safe_get(row, "swe_delta_7")
    soil = _safe_get(row, "soil_wetness")
    frac_snow = _safe_get(row, "frac_snow")

    p3_q90 = _safe_get(clim, "p3_q90")
    p7_q80 = _safe_get(clim, "p7_q80")
    p30_q75 = _safe_get(clim, "p30_q75")
    swe_q75 = _safe_get(clim, "swe_q75")
    melt_q20 = _safe_get(clim, "melt_q20")
    soil_q75 = _safe_get(clim, "soil_q75")

    heavy_rain = (
        (np.isfinite(p7) and np.isfinite(p7_q80) and p7 >= p7_q80)
        or (np.isfinite(p3) and np.isfinite(p3_q90) and p3 >= p3_q90)
    )
    rain_concentrated = np.isfinite(p3) and np.isfinite(p7) and p7 > EPS and p3 / p7 >= 0.55
    snowpack = np.isfinite(swe_before) and swe_before > EPS and (
        not np.isfinite(swe_q75) or swe_before >= swe_q75 or (np.isfinite(frac_snow) and frac_snow > 0.15)
    )
    melt = np.isfinite(swe_delta) and swe_delta < -EPS and (
        not np.isfinite(melt_q20) or swe_delta <= melt_q20
    )
    wet_antecedent = (
        (np.isfinite(p30) and np.isfinite(p30_q75) and p30 >= p30_q75)
        or (np.isfinite(soil) and np.isfinite(soil_q75) and soil >= soil_q75)
    )
    warm = np.isfinite(temp7) and temp7 >= 8.0
    cool_season = month in (10, 11, 12, 1, 2, 3, 4)
    warm_season = month in (5, 6, 7, 8, 9)

    if snowpack and heavy_rain and melt:
        event_type = "rain_on_snow"
    elif snowpack and melt:
        event_type = "snowmelt"
    elif heavy_rain and rain_concentrated and warm and warm_season:
        event_type = "warm_season_short_rain"
    elif heavy_rain and wet_antecedent:
        event_type = "long_rain_wet_soil"
    elif heavy_rain and cool_season:
        event_type = "cool_season_rain"
    elif wet_antecedent:
        event_type = "storage_release"
    else:
        event_type = "rainfall_runoff"

    flags = {
        "heavy_rain_flag": bool(heavy_rain),
        "rain_concentrated_flag": bool(rain_concentrated),
        "snowpack_flag": bool(snowpack),
        "melt_flag": bool(melt),
        "wet_antecedent_flag": bool(wet_antecedent),
    }
    return event_type, flags


def _event_rows_for_split(g: pd.DataFrame, split: str, q95: float, clim: pd.Series) -> list[dict]:
    sub = g[g["split"].eq(split)].sort_values("date").copy()
    if sub.empty or not np.isfinite(q95) or q95 <= 0:
        return []
    high = sub[pd.to_numeric(sub["q_mm_day"], errors="coerce") >= q95].copy()
    if high.empty:
        return []

    events = []
    event_no = 0
    split_min = sub["date"].min()
    split_max = sub["date"].max()
    gaps = high["date"].diff().dt.days.fillna(9999)
    high["group"] = (gaps > 3).cumsum()
    for _, ev in high.groupby("group"):
        if ev.empty:
            continue
        first_high = ev["date"].min()
        last_high = ev["date"].max()
        win_start = max(first_high - pd.Timedelta(days=2), split_min)
        win_end = min(last_high + pd.Timedelta(days=2), split_max)
        win = sub[sub["date"].between(win_start, win_end)].copy()
        if win.empty:
            continue
        peak_idx = win["q_mm_day"].idxmax()
        peak = win.loc[peak_idx]
        event_no += 1
        peak_date = peak["date"]
        row = {
            "event_id": f"{split}_{int(peak['ID'])}_{event_no:04d}",
            "split": split,
            "ID": int(peak["ID"]),
            "event_start": win_start,
            "event_end": win_end,
            "first_high_date": first_high,
            "last_high_date": last_high,
            "peak_date": peak_date,
            "season": _season(int(peak_date.month)),
            "water_year": int(peak_date.year + (peak_date.month >= 10)),
            "month": int(peak_date.month),
            "duration_days": int((win_end - win_start).days + 1),
            "high_flow_days": int(len(ev)),
            "threshold_q95": float(q95),
            "peak_q": float(peak["q_mm_day"]),
            "event_volume_q": float(win["q_mm_day"].sum()),
            "p3": _safe_get(peak, "prec_sum_3"),
            "p7": _safe_get(peak, "prec_sum_7"),
            "p30": _safe_get(peak, "prec_sum_30"),
            "event_precip": float(win["prec"].sum()) if "prec" in win.columns else np.nan,
            "temp7": _safe_get(peak, "2m_temp_mean_mean_7"),
            "temp30": _safe_get(peak, "2m_temp_mean_mean_30"),
            "swe_before": _safe_get(peak, "swe_lag_7"),
            "swe_mean7": _safe_get(peak, "swe_mean_7"),
            "swe_delta_7": _safe_get(peak, "swe_delta_7"),
            "soil_wetness": _safe_get(peak, "volsw_123_mean_7"),
            "soil_wetness_30": _safe_get(peak, "volsw_123_mean_30"),
            "frac_snow": _safe_get(peak, "frac_snow"),
            "aridity": _safe_get(peak, "arid_1")
            if np.isfinite(_safe_get(peak, "arid_1"))
            else _safe_get(peak, "arid_2"),
            "p_season": _safe_get(peak, "p_season"),
            "area_calc": _safe_get(peak, "area_calc"),
        }
        event_type, flags = _classify_event(pd.Series(row), clim)
        row["event_type"] = event_type
        row["climate_sensitivity"] = _climate_sensitivity(event_type)
        row.update(flags)
        events.append(row)
    return events


def build_flood_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    clim = _basin_climatology(frame).set_index("ID")
    events: list[dict] = []
    for bid, g in frame.groupby("ID", sort=False):
        if bid not in clim.index:
            continue
        q95 = _safe_get(clim.loc[bid], "q95")
        for split in EVENT_SPLITS:
            events.extend(_event_rows_for_split(g, split, q95, clim.loc[bid]))
    out = pd.DataFrame(events)
    if out.empty:
        return out
    for col in ["event_start", "event_end", "first_high_date", "last_high_date", "peak_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.sort_values(["split", "ID", "peak_date"]).reset_index(drop=True)


def _event_day_map(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in events.itertuples(index=False):
        dates = pd.date_range(row.event_start, row.event_end, freq="D")
        rows.append(pd.DataFrame({
            "event_id": row.event_id,
            "ID": int(row.ID),
            "split": row.split,
            "date": dates,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _prediction_files(tables: Path) -> list[tuple[Path, str | None]]:
    return [
        (tables / "predictions_tabular.csv", None),
        (tables / "predictions_xlstm.csv", "xlstm"),
        (tables / "predictions_transformer.csv", "transformer"),
    ]


def score_flood_events(out_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    tables = out_dir / "tables"
    event_days = _event_day_map(events)
    if event_days.empty:
        return pd.DataFrame()
    use_splits = set(events["split"].astype(str).unique())
    event_days["date"] = pd.to_datetime(event_days["date"], errors="coerce")
    parts = []
    for path, model_label in _prediction_files(tables):
        if not path.exists():
            continue
        try:
            reader = pd.read_csv(
                path,
                usecols=lambda c: c in PRED_COLUMNS,
                chunksize=800_000,
            )
        except pd.errors.EmptyDataError:
            continue
        for chunk in reader:
            if chunk.empty or "split" not in chunk.columns:
                continue
            chunk = chunk[chunk["split"].astype(str).isin(use_splits)].copy()
            if chunk.empty:
                continue
            if "model" not in chunk.columns:
                chunk["model"] = model_label
            elif model_label is not None:
                chunk["model"] = chunk["model"].fillna(model_label)
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            hit = chunk.merge(event_days, on=["ID", "date", "split"], how="inner")
            if not hit.empty:
                parts.append(hit)
    if not parts:
        return pd.DataFrame()
    daily = pd.concat(parts, ignore_index=True)
    daily["q_mm_day"] = pd.to_numeric(daily["q_mm_day"], errors="coerce")
    daily["q_pred_mm_day"] = pd.to_numeric(daily["q_pred_mm_day"], errors="coerce")
    meta = events.set_index("event_id")
    rows = []
    for (event_id, model), g in daily.groupby(["event_id", "model"], sort=False):
        if event_id not in meta.index or g.empty:
            continue
        ev = meta.loc[event_id]
        sim = g.dropna(subset=["q_pred_mm_day"]).sort_values("date")
        if sim.empty:
            continue
        sim_peak_idx = sim["q_pred_mm_day"].idxmax()
        sim_peak = sim.loc[sim_peak_idx]
        obs_peak_q = float(ev["peak_q"])
        obs_vol = float(ev["event_volume_q"])
        sim_peak_q = float(sim_peak["q_pred_mm_day"])
        sim_vol = float(sim["q_pred_mm_day"].sum())
        peak_date = pd.to_datetime(ev["peak_date"])
        sim_peak_date = pd.to_datetime(sim_peak["date"])
        rows.append({
            "event_id": event_id,
            "split": ev["split"],
            "ID": int(ev["ID"]),
            "model": str(model),
            "event_type": ev["event_type"],
            "season": ev["season"],
            "climate_sensitivity": ev["climate_sensitivity"],
            "peak_q_obs": obs_peak_q,
            "peak_q_sim": sim_peak_q,
            "peak_ratio": sim_peak_q / (obs_peak_q + EPS),
            "peak_bias_pct": 100.0 * (sim_peak_q - obs_peak_q) / (obs_peak_q + EPS),
            "abs_peak_bias_pct": abs(100.0 * (sim_peak_q - obs_peak_q) / (obs_peak_q + EPS)),
            "volume_q_obs": obs_vol,
            "volume_q_sim": sim_vol,
            "volume_ratio": sim_vol / (obs_vol + EPS),
            "volume_bias_pct": 100.0 * (sim_vol - obs_vol) / (obs_vol + EPS),
            "abs_volume_bias_pct": abs(100.0 * (sim_vol - obs_vol) / (obs_vol + EPS)),
            "peak_timing_error_days": int((sim_peak_date - peak_date).days),
            "abs_peak_timing_error_days": abs(int((sim_peak_date - peak_date).days)),
            "threshold_hit": int(sim_peak_q >= float(ev["threshold_q95"])),
        })
    return pd.DataFrame(rows)


def summarize_flood_typology(events: pd.DataFrame, skill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    event_summary = (
        events.groupby(["split", "event_type", "season", "climate_sensitivity"])
        .agg(
            events=("event_id", "nunique"),
            basins=("ID", "nunique"),
            median_peak_q=("peak_q", "median"),
            median_p7=("p7", "median"),
            median_p30=("p30", "median"),
            median_temp7=("temp7", "median"),
            median_swe_before=("swe_before", "median"),
            median_swe_delta_7=("swe_delta_7", "median"),
            median_soil_wetness=("soil_wetness", "median"),
        )
        .reset_index()
    )
    if skill.empty:
        return event_summary, pd.DataFrame()
    model_summary = (
        skill.groupby(["split", "event_type", "model"])
        .agg(
            events=("event_id", "nunique"),
            basins=("ID", "nunique"),
            median_peak_ratio=("peak_ratio", "median"),
            median_peak_bias_pct=("peak_bias_pct", "median"),
            median_abs_peak_bias_pct=("abs_peak_bias_pct", "median"),
            median_volume_ratio=("volume_ratio", "median"),
            median_volume_bias_pct=("volume_bias_pct", "median"),
            median_abs_volume_bias_pct=("abs_volume_bias_pct", "median"),
            median_peak_timing_error_days=("peak_timing_error_days", "median"),
            median_abs_peak_timing_error_days=("abs_peak_timing_error_days", "median"),
            hit_rate=("threshold_hit", "mean"),
        )
        .reset_index()
    )
    model_summary["hit_rate"] = 100.0 * model_summary["hit_rate"]
    return event_summary, model_summary


def generate_flood_typology(out_dir: Path) -> None:
    tables = out_dir / "tables"
    frame_path = tables / "model_frame.parquet"
    if not frame_path.exists():
        return
    outputs = [
        tables / "flood_events.csv",
        tables / "flood_event_model_skill.csv",
        tables / "flood_typology_summary.csv",
        tables / "flood_typology_model_summary.csv",
    ]
    inputs = [frame_path] + [p for p, _ in _prediction_files(tables) if p.exists()]
    if _outputs_current(outputs, inputs):
        return
    frame = _frame_with_columns(frame_path)
    events = build_flood_events(frame)
    events.to_csv(tables / "flood_events.csv", index=False)
    skill = score_flood_events(out_dir, events)
    skill.to_csv(tables / "flood_event_model_skill.csv", index=False)
    event_summary, model_summary = summarize_flood_typology(events, skill)
    event_summary.to_csv(tables / "flood_typology_summary.csv", index=False)
    model_summary.to_csv(tables / "flood_typology_model_summary.csv", index=False)
