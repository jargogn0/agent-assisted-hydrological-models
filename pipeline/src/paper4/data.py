from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASIN_DIRS = {
    "A": "A_basins_total_upstrm",
    "B": "B_basins_intermediate_all",
    "C": "C_basins_intermediate_lowimp",
}


def read_lamah_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", **kwargs)


def basin_dir(root: Path, basin_set: str) -> Path:
    key = basin_set.upper()
    if key not in BASIN_DIRS:
        raise ValueError(f"Unknown basin_set {basin_set!r}; expected one of {sorted(BASIN_DIRS)}")
    return root / BASIN_DIRS[key]


def forcing_path(root: Path, basin_set: str, time_step: str, basin_id: int) -> Path:
    return basin_dir(root, basin_set) / "2_timeseries" / time_step / f"ID_{basin_id}.csv"


def runoff_path(root: Path, time_step: str, basin_id: int) -> Path:
    return root / "D_gauges" / "2_timeseries" / time_step / f"ID_{basin_id}.csv"


def load_attributes(root: Path, basin_set: str) -> pd.DataFrame:
    bdir = basin_dir(root, basin_set)
    catchment = read_lamah_csv(bdir / "1_attributes" / "Catchment_attributes.csv")
    gauge = read_lamah_csv(root / "D_gauges" / "1_attributes" / "Gauge_attributes.csv")
    hydro_1989 = read_lamah_csv(root / "D_gauges" / "1_attributes" / "Hydro_indices_1989_2009.csv")
    hydro_1981 = read_lamah_csv(root / "D_gauges" / "1_attributes" / "Hydro_indices_1981_2017.csv")

    attrs = catchment.merge(gauge, on="ID", how="left", suffixes=("", "_gauge"))
    attrs = attrs.merge(hydro_1989.add_prefix("h1989_").rename(columns={"h1989_ID": "ID"}), on="ID", how="left")
    attrs = attrs.merge(hydro_1981.add_prefix("h1981_").rename(columns={"h1981_ID": "ID"}), on="ID", how="left")

    water_balance = bdir / "1_attributes" / "Water_balance.csv"
    if water_balance.exists():
        wb = read_lamah_csv(water_balance)
        attrs = attrs.merge(wb.add_prefix("wb_").rename(columns={"wb_ID": "ID"}), on="ID", how="left")

    hierarchy = bdir / "1_attributes" / "Gauge_hierarchy.csv"
    if hierarchy.exists():
        gh = read_lamah_csv(hierarchy)
        attrs = attrs.merge(gh.add_prefix("net_").rename(columns={"net_ID": "ID"}), on="ID", how="left")

    transfers = bdir / "1_attributes" / "Crossbasin_water_transfers.csv"
    if transfers.exists():
        tr = read_lamah_csv(transfers)
        tr_summary = (
            tr.groupby("ID")
            .agg(
                transfer_count=("To_ID", "count"),
                transfer_upper_max=("upper_thres", "max"),
                transfer_estimated_frac=("estimated", "mean"),
            )
            .reset_index()
        )
        attrs = attrs.merge(tr_summary, on="ID", how="left")

    diagnostic_cols = pd.DataFrame(index=attrs.index)
    if "transfer_count" in attrs.columns:
        diagnostic_cols["transfer_count"] = attrs["transfer_count"].fillna(0)
    else:
        diagnostic_cols["transfer_count"] = 0
    diagnostic_cols["has_transfer"] = (diagnostic_cols["transfer_count"] > 0).astype(int)
    diagnostic_cols["is_impacted"] = (attrs.get("typimpact", "-").fillna("-").astype(str) != "-").astype(int)
    attrs = pd.concat([attrs.drop(columns=[c for c in diagnostic_cols.columns if c in attrs.columns]), diagnostic_cols], axis=1)
    return attrs


def available_ids(root: Path, basin_set: str, time_step: str) -> list[int]:
    fdir = basin_dir(root, basin_set) / "2_timeseries" / time_step
    rdir = root / "D_gauges" / "2_timeseries" / time_step
    forcing_ids = {int(p.stem.split("_")[1]) for p in fdir.glob("ID_*.csv")}
    runoff_ids = {int(p.stem.split("_")[1]) for p in rdir.glob("ID_*.csv")}
    return sorted(forcing_ids & runoff_ids)


def _valid_runoff_days(root: Path, time_step: str, basin_id: int) -> int:
    path = runoff_path(root, time_step, basin_id)
    if not path.exists():
        return 0
    try:
        q = read_lamah_csv(path, usecols=["qobs"])
    except ValueError:
        return 0
    qobs = pd.to_numeric(q["qobs"], errors="coerce")
    return int(((qobs.notna()) & (qobs != -999)).sum())


def _qcut_class(values: pd.Series, labels: tuple[str, ...]) -> pd.Series:
    """Three-class quantile binning compatible with diagnostics._qclass."""
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < 6 or x.nunique(dropna=True) < 3:
        return pd.Series(["unknown"] * len(values), index=values.index, dtype=object)
    q1, q2 = x.quantile([1 / 3, 2 / 3])
    out = pd.Series(labels[1], index=values.index, dtype="object")
    out[x <= q1] = labels[0]
    out[x >= q2] = labels[2]
    out[x.isna()] = "unknown"
    return out


def _stratified_subset(idx: pd.DataFrame, target_n: int, stratify_by: list[str],
                       seed: int = 42) -> pd.DataFrame:
    """Pick ~target_n basins, stratified across the joint product of `stratify_by`.

    Falls back gracefully when a stratum has fewer basins than its quota.
    Within a stratum, basins are ranked by (valid_runoff_days desc, ID asc) so
    longer-record gauges are preferred — matches the existing default ordering.
    """
    if idx.empty or target_n <= 0 or not stratify_by:
        return idx.head(target_n) if target_n else idx.iloc[0:0]

    work = idx.copy()
    if "frac_snow" in work.columns and "snow_class" in stratify_by:
        work["snow_class"] = _qcut_class(work["frac_snow"],
                                         ("low_snow", "mixed_snow", "snow_influenced"))
    if "arid_1" in work.columns and "aridity_class" in stratify_by:
        work["aridity_class"] = _qcut_class(work["arid_1"],
                                            ("humid", "moderate_aridity", "dry"))
    if "geol_perme" in work.columns and "geology_class" in stratify_by:
        work["geology_class"] = _qcut_class(work["geol_perme"],
                                            ("low_perm", "mixed_perm", "high_perm"))

    sort_cols, ascending = [], []
    if "valid_runoff_days" in work.columns:
        sort_cols.append("valid_runoff_days"); ascending.append(False)
    sort_cols.append("ID"); ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending)

    keys = [c for c in stratify_by if c in work.columns]
    if not keys:
        return work.head(target_n).reset_index(drop=True)

    groups = list(work.groupby(keys, sort=True))
    n_groups = max(len(groups), 1)
    base_quota = target_n // n_groups
    remainder = target_n - base_quota * n_groups

    rng = np.random.default_rng(seed)
    group_order = list(range(len(groups)))
    rng.shuffle(group_order)
    quotas = [base_quota] * len(groups)
    for i in group_order[:remainder]:
        quotas[i] += 1

    picked: list[pd.DataFrame] = []
    leftovers: list[pd.DataFrame] = []
    for (_, sub), q in zip(groups, quotas):
        take = min(q, len(sub))
        picked.append(sub.head(take))
        if len(sub) > take:
            leftovers.append(sub.iloc[take:])

    out = pd.concat(picked, ignore_index=True) if picked else work.iloc[0:0]
    deficit = target_n - len(out)
    if deficit > 0 and leftovers:
        pool = pd.concat(leftovers, ignore_index=True)
        out = pd.concat([out, pool.head(deficit)], ignore_index=True)
    return out.reset_index(drop=True)


def _random_subset(idx: pd.DataFrame, target_n: int, seed: int = 42) -> pd.DataFrame:
    if idx.empty or target_n <= 0 or target_n >= len(idx):
        return idx.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(idx.index.to_numpy(), size=int(target_n), replace=False)
    out = idx.loc[chosen].copy()
    return out.sort_values("ID").reset_index(drop=True)


def build_catchment_index(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    basin_set = cfg["data"]["basin_set"]
    time_step = cfg["data"]["time_step"]
    attrs = load_attributes(root, basin_set)
    ids = available_ids(root, basin_set, time_step)
    idx = attrs[attrs["ID"].isin(ids)].copy()

    if cfg["data"].get("low_impact_only", False):
        idx = idx[idx["degimpact"].fillna("-").astype(str).str.lower().eq("l")]
    if not cfg["data"].get("include_impacted", True):
        idx = idx[idx["is_impacted"] == 0]
    if cfg["data"].get("exclude_transfers", False):
        idx = idx[idx.get("has_transfer", 0).fillna(0).astype(int) == 0]
    impact_allow = cfg["data"].get("impact_class_allow")
    if impact_allow:
        deg = idx["degimpact"].fillna("-").astype(str).str.lower()
        typ = idx["typimpact"].fillna("-").astype(str).str.strip()
        impact_class = pd.Series("near_natural", index=idx.index, dtype="object")
        nonnat = (typ != "-") & (typ != "")
        impact_class[nonnat & (deg.isin(["s", "m", "l"]))] = "impact_" + deg[nonnat & (deg.isin(["s", "m", "l"]))]
        impact_class[nonnat & (~deg.isin(["s", "m", "l"]))] = "impact_unknown"
        if "has_transfer" in idx.columns:
            impact_class[idx["has_transfer"].fillna(0).astype(int) == 1] = "transfer"
        allow = set(impact_allow)
        idx = idx[impact_class.isin(allow)]

    precheck_max = cfg["data"].get("precheck_max_catchments")
    if precheck_max:
        idx = idx.sort_values("ID").head(int(precheck_max))

    min_valid = int(cfg["data"].get("min_valid_days") or 0)
    if min_valid > 0:
        idx["valid_runoff_days"] = [
            _valid_runoff_days(root, time_step, int(i)) for i in idx["ID"].tolist()
        ]
        idx = idx[idx["valid_runoff_days"] >= min_valid]
    else:
        idx["valid_runoff_days"] = pd.NA
    idx = idx.sort_values(["valid_runoff_days", "ID"], ascending=[False, True])

    random_subset = cfg["data"].get("random_subset")
    stratify = cfg["data"].get("stratified_subset")
    if random_subset:
        target = int(random_subset.get("target_n") or cfg["data"].get("max_catchments") or len(idx))
        idx = _random_subset(idx, target, seed=int(random_subset.get("seed", 42)))
    elif stratify:
        target = int(stratify.get("target_n") or cfg["data"].get("max_catchments") or len(idx))
        keys = list(stratify.get("stratify_by")
                    or ["snow_class", "aridity_class", "geology_class"])
        idx = _stratified_subset(idx, target, keys,
                                 seed=int(stratify.get("seed", 42)))
    else:
        max_catchments = cfg["data"].get("max_catchments")
        if max_catchments:
            idx = idx.head(int(max_catchments))
    return idx.reset_index(drop=True)


def load_daily_basin(cfg: dict, basin_id: int, attrs: pd.DataFrame | None = None) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    basin_set = cfg["data"]["basin_set"]
    time_step = cfg["data"]["time_step"]
    if time_step != "daily":
        raise NotImplementedError("Tabular feature builder currently supports daily data.")

    met = read_lamah_csv(forcing_path(root, basin_set, time_step, basin_id))
    q = read_lamah_csv(runoff_path(root, time_step, basin_id))
    for df in (met, q):
        df["date"] = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))

    df = met.merge(q[["date", "qobs", "ckhs", "qceq", "qcol"]], on="date", how="inner")
    df["ID"] = int(basin_id)
    df["qobs"] = pd.to_numeric(df["qobs"], errors="coerce").replace(-999, np.nan)

    if attrs is None:
        attrs = load_attributes(root, basin_set)
    row = attrs.loc[attrs["ID"] == basin_id]
    if row.empty:
        area_km2 = np.nan
    else:
        area_km2 = float(row.iloc[0].get("area_gov", np.nan))
        if not np.isfinite(area_km2) or area_km2 <= 0:
            area_km2 = float(row.iloc[0].get("area_calc", np.nan))

    df["area_km2"] = area_km2
    df["q_mm_day"] = df["qobs"] * 86400.0 / (area_km2 * 1_000_000.0) * 1000.0
    df["water_year"] = np.where(df["MM"] >= 10, df["YYYY"] + 1, df["YYYY"])
    start_year = cfg["data"].get("start_year")
    end_year = cfg["data"].get("end_year")
    if start_year is not None:
        df = df[df["YYYY"] >= int(start_year)]
    if end_year is not None:
        df = df[df["YYYY"] <= int(end_year)]
    return df


def concat_basins(cfg: dict, ids: Iterable[int], attrs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    ids = list(ids)
    for k, basin_id in enumerate(ids, start=1):
        print(f"[paper4] loading basin {k}/{len(ids)}: ID_{int(basin_id)}", flush=True)
        try:
            frames.append(load_daily_basin(cfg, int(basin_id), attrs))
        except FileNotFoundError:
            print(f"[paper4] missing basin files for ID_{int(basin_id)}; skipping", flush=True)
            continue
    if not frames:
        raise RuntimeError("No basin time series could be loaded.")
    return pd.concat(frames, ignore_index=True)
