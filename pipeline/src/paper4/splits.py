from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_qcut(values: pd.Series, bins: int, labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().nunique() < 2:
        return pd.Series(["all"] * len(values), index=values.index, dtype=object)
    try:
        codes = pd.qcut(numeric, q=bins, labels=False, duplicates="drop")
        return codes.map(lambda x: labels[int(x)] if pd.notna(x) and int(x) < len(labels) else "all").astype(str)
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index, dtype=object)


def _stratified_hydro_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Assign deterministic spatial folds balanced by broad hydrologic regimes."""
    attrs = df.groupby("ID", sort=True).first()
    strata = pd.Series("", index=attrs.index, dtype=object)

    if "frac_snow" in attrs.columns:
        snow = pd.to_numeric(attrs["frac_snow"], errors="coerce").fillna(0.0)
        strata += np.where(snow >= 0.15, "snow_hi", np.where(snow >= 0.03, "snow_mid", "snow_low"))
    else:
        strata += "snow_all"

    if "arid_1" in attrs.columns:
        strata += "_" + _safe_qcut(attrs["arid_1"], 3, ["humid", "mid_arid", "dry"]).to_numpy()
    elif "p_mean" in attrs.columns and "et0_mean" in attrs.columns:
        aridity = pd.to_numeric(attrs["et0_mean"], errors="coerce") / (pd.to_numeric(attrs["p_mean"], errors="coerce") + 1e-9)
        strata += "_" + _safe_qcut(aridity, 3, ["humid", "mid_arid", "dry"]).to_numpy()
    else:
        strata += "_arid_all"

    if "is_impacted" in attrs.columns:
        impacted = pd.to_numeric(attrs["is_impacted"], errors="coerce").fillna(0).astype(int)
        strata += np.where(impacted > 0, "_impacted", "_natural")

    rng = np.random.default_rng(seed)
    fold_by_id: dict[int, int] = {}
    for _, ids in strata.groupby(strata, sort=True):
        ordered = np.asarray(ids.index.tolist(), dtype=int)
        ordered = ordered[rng.permutation(len(ordered))]
        for i, basin_id in enumerate(ordered):
            fold_by_id[int(basin_id)] = int(i % n_folds)
    return df["ID"].astype(int).map(fold_by_id)


def _random_basin_holdout(df: pd.DataFrame, train_basins: int, test_basins: int, seed: int) -> tuple[pd.Series, set[int], set[int]]:
    ids = np.asarray(sorted(df["ID"].astype(int).unique().tolist()), dtype=int)
    if train_basins + test_basins > len(ids):
        raise ValueError(
            f"Requested train_basins + test_basins = {train_basins + test_basins}, "
            f"but only {len(ids)} basins are available after filtering."
        )
    rng = np.random.default_rng(seed)
    shuffled = ids[rng.permutation(len(ids))]
    train_ids = set(shuffled[:train_basins].tolist())
    test_ids = set(shuffled[train_basins: train_basins + test_basins].tolist())
    role = pd.Series("unused", index=df.index, dtype=object)
    role[df["ID"].astype(int).isin(train_ids)] = "train_group"
    role[df["ID"].astype(int).isin(test_ids)] = "test_group"
    return role, train_ids, test_ids


def add_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    temporal = cfg["splits"]["temporal"]
    df["split"] = "unused"
    df.loc[df["YYYY"] <= int(temporal["train_end_year"]), "split"] = "train"
    df.loc[
        (df["YYYY"] >= int(temporal["val_start_year"])) &
        (df["YYYY"] <= int(temporal["val_end_year"])),
        "split",
    ] = "val"
    df.loc[
        (df["YYYY"] >= int(temporal["test_start_year"])) &
        (df["YYYY"] <= int(temporal["test_end_year"])),
        "split",
    ] = "test"

    basin_holdout = cfg["splits"].get("basin_holdout", {})
    if basin_holdout.get("enabled", False):
        role, train_ids, test_ids = _random_basin_holdout(
            df,
            int(basin_holdout["train_basins"]),
            int(basin_holdout["test_basins"]),
            int(basin_holdout.get("seed", 42)),
        )
        df.loc[~df["ID"].astype(int).isin(train_ids | test_ids), "split"] = "unused"
        df.loc[df["ID"].astype(int).isin(test_ids), "split"] = "unused"
        test_years = (
            (df["YYYY"] >= int(temporal["test_start_year"]))
            & (df["YYYY"] <= int(temporal["test_end_year"]))
        )
        df.loc[df["ID"].astype(int).isin(test_ids) & test_years, "split"] = "spatial_test"
    else:
        spatial = cfg["splits"].get("spatial", {})
        if not spatial.get("enabled", False):
            return df
        n_folds = int(spatial.get("n_folds", 5))
        test_fold = int(spatial.get("test_fold", 0))
        strategy = str(spatial.get("fold_strategy", "modulo_id"))
        if strategy == "stratified_hydro":
            fold = _stratified_hydro_folds(df, n_folds, int(spatial.get("fold_seed", 42)))
        else:
            fold = df["ID"].astype(int).map(lambda x: x % n_folds)
        heldout = fold == test_fold
        eval_period = str(spatial.get("evaluation_period", "all"))
        if eval_period == "test":
            test_years = (
                (df["YYYY"] >= int(temporal["test_start_year"]))
                & (df["YYYY"] <= int(temporal["test_end_year"]))
            )
            df.loc[heldout, "split"] = "unused"
            df.loc[heldout & test_years, "split"] = "spatial_test"
        else:
            df.loc[heldout, "split"] = "spatial_test"
        df.loc[(fold != test_fold) & (df["split"] == "test"), "split"] = "test"
    return df
