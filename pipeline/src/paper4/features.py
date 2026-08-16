from __future__ import annotations

import numpy as np
import pandas as pd


STATIC_FAMILIES = {
    "climate": [
        "p_mean", "et0_mean", "eta_mean", "arid_1", "arid_2", "p_season",
        "frac_snow", "hi_prec_fr", "hi_prec_du", "lo_prec_fr", "lo_prec_du",
    ],
    "topography": [
        "area_calc", "area_gov", "elev_mean", "elev_med", "elev_std", "elev_ran",
        "slope_mean", "mvert_dist", "mvert_ang", "elon_ratio", "strm_dens",
    ],
    "landcover": [
        "agr_fra", "bare_fra", "forest_fra", "glac_fra", "lake_fra", "urban_fra",
        "lai_max", "lai_diff", "ndvi_max", "ndvi_min", "gvf_max", "gvf_diff",
    ],
    "soil": [
        "bedrk_dep", "root_dep", "soil_poros", "soil_condu", "soil_tawc",
        "sand_fra", "silt_fra", "clay_fra", "grav_fra", "oc_fra",
    ],
    "geology": [
        "gc_ig_fra", "gc_mt_fra", "gc_pa_fra", "gc_pb_fra", "gc_pi_fra",
        "gc_py_fra", "gc_sc_fra", "gc_sm_fra", "gc_ss_fra", "gc_su_fra",
        "gc_va_fra", "gc_vb_fra", "gc_wb_fra", "geol_perme", "geol_poros",
    ],
    "human": ["gaps_pre", "gaps_post", "area_ratio", "diur_art", "diur_glac", "is_impacted", "has_transfer", "transfer_count"],
    "network": ["net_HIERARCHY"],
}


def static_columns(cfg: dict, attrs: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for family in cfg["features"].get("static_families", []):
        cols.extend(STATIC_FAMILIES.get(family, []))
    return [c for c in dict.fromkeys(cols) if c in attrs.columns]


def add_dynamic_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.sort_values(["ID", "date"]).copy()
    dyn_cols = [c for c in cfg["features"]["dynamic"] if c in df.columns]
    windows = cfg["features"].get("rolling_windows", [3, 7, 30])

    df["doy_sin"] = np.sin(2 * np.pi * df["DOY"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["DOY"] / 366.0)

    grouped = df.groupby("ID", group_keys=False)
    for col in dyn_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for w in windows:
            if col == "prec" or col == "total_et":
                df[f"{col}_sum_{w}"] = grouped[col].transform(lambda s, ww=w: s.rolling(ww, min_periods=1).sum())
            else:
                df[f"{col}_mean_{w}"] = grouped[col].transform(lambda s, ww=w: s.rolling(ww, min_periods=1).mean())
        if col in {"prec", "swe", "2m_temp_mean", "volsw_123"}:
            df[f"{col}_lag_1"] = grouped[col].shift(1)
            df[f"{col}_lag_7"] = grouped[col].shift(7)
    if "swe" in df.columns:
        df["swe_delta_7"] = grouped["swe"].transform(lambda s: s - s.shift(7))
        df["swe_delta_30"] = grouped["swe"].transform(lambda s: s - s.shift(30))
    return df


def attach_static(df: pd.DataFrame, attrs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = ["ID"] + static_columns(cfg, attrs)
    static = attrs[cols].copy()
    for c in static.columns:
        if c != "ID":
            static[c] = pd.to_numeric(static[c], errors="coerce")
    return df.merge(static, on="ID", how="left")


def build_model_frame(raw: pd.DataFrame, attrs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = add_dynamic_features(raw, cfg)
    df = attach_static(df, attrs, cfg)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["q_mm_day", "qobs"])
    df = df[df["q_mm_day"] >= 0].copy()

    excluded = {
        "date", "qobs", "q_mm_day", "ckhs", "qceq", "qcol", "water_year",
        "YYYY", "MM", "DD", "DOY", "area_km2", "ID",
    }
    feature_cols = [
        c for c in df.columns
        if c not in excluded and c != "target" and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df, feature_cols
