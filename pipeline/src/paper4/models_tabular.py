from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


def target_values(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    y = df[cfg["target"]["variable"]].to_numpy(dtype=float)
    if cfg["target"].get("transform") == "log1p":
        return np.log1p(np.maximum(y, 0))
    return y


def inverse_target(yhat, cfg: dict) -> np.ndarray:
    yhat = np.asarray(yhat, dtype=float)
    if cfg["target"].get("transform") == "log1p":
        return np.maximum(np.expm1(yhat), 0)
    return yhat


def _runtime_n_jobs(cfg: dict, raw_n_jobs) -> int:
    """Cap tabular parallelism for final runs on memory-limited laptops."""
    requested = int(raw_n_jobs)
    env_cap = os.environ.get("PAPER4_TABULAR_N_JOBS")
    cfg_cap = cfg.get("models", {}).get("final_n_jobs")
    cap = int(env_cap if env_cap is not None else cfg_cap) if (env_cap is not None or cfg_cap is not None) else None
    if cap is None or cap <= 0:
        return requested
    if requested < 0 or requested > cap:
        return cap
    return requested


def _runtime_params(model_name: str, params: dict, cfg: dict) -> dict:
    limited = dict(params)
    if "n_jobs" in limited:
        limited["n_jobs"] = _runtime_n_jobs(cfg, limited["n_jobs"])
    return limited


def _rf(cfg: dict) -> Pipeline:
    p = cfg["models"]["random_forest"]
    max_samples = p.get("max_samples")
    max_features = p.get("max_features")
    rf_kwargs = {}
    if max_samples is not None:
        rf_kwargs["max_samples"] = float(max_samples) if isinstance(max_samples, float) else max_samples
    if max_features is not None:
        rf_kwargs["max_features"] = float(max_features) if isinstance(max_features, float) else max_features
    model = RandomForestRegressor(
        n_estimators=int(p["n_estimators"]),
        max_depth=p.get("max_depth"),
        min_samples_leaf=int(p["min_samples_leaf"]),
        n_jobs=_runtime_n_jobs(cfg, p["n_jobs"]),
        random_state=int(p["random_state"]),
        **rf_kwargs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _xgb(cfg: dict) -> Pipeline:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed.")
    p = cfg["models"]["xgboost"]
    extra = {}
    if p.get("reg_alpha") is not None:
        extra["reg_alpha"] = float(p["reg_alpha"])
    if p.get("min_child_weight") is not None:
        extra["min_child_weight"] = float(p["min_child_weight"])
    if p.get("gamma") is not None:
        extra["gamma"] = float(p["gamma"])
    model = XGBRegressor(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        reg_lambda=float(p["reg_lambda"]),
        objective=p.get("objective", "reg:squarederror"),
        n_jobs=_runtime_n_jobs(cfg, p["n_jobs"]),
        random_state=int(p["random_state"]),
        tree_method="hist",
        **extra,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def enabled_models(cfg: dict) -> dict[str, Pipeline]:
    models = {}
    if cfg["models"]["random_forest"].get("enabled", True):
        models["random_forest"] = _rf(cfg)
    if cfg["models"]["xgboost"].get("enabled", True):
        models["xgboost"] = _xgb(cfg)
    return models


def _enabled_model_names(cfg: dict) -> list[str]:
    names = []
    if cfg["models"]["random_forest"].get("enabled", True):
        names.append("random_forest")
    if cfg["models"]["xgboost"].get("enabled", True):
        names.append("xgboost")
    return names


def _update_model_params(cfg: dict, model_name: str, params: dict) -> dict:
    tuned = copy.deepcopy(cfg)
    tuned["models"][model_name].update(params)
    return tuned


def _model_from_name(model_name: str, cfg: dict) -> Pipeline:
    if model_name == "random_forest":
        return _rf(cfg)
    if model_name == "xgboost":
        return _xgb(cfg)
    raise ValueError(model_name)


def _default_candidates(model_name: str, cfg: dict) -> list[dict]:
    base = dict(cfg["models"][model_name])
    base.pop("enabled", None)
    if model_name == "random_forest":
        return [
            {},
            {"n_estimators": 500, "max_depth": 18, "min_samples_leaf": 3, "max_samples": 0.8, "max_features": 0.7},
            {"n_estimators": 700, "max_depth": 28, "min_samples_leaf": 2, "max_samples": 0.75, "max_features": 0.6},
            {"n_estimators": 500, "max_depth": 14, "min_samples_leaf": 8, "max_samples": 0.9, "max_features": 0.8},
        ]
    if model_name == "xgboost":
        return [
            {},
            {"n_estimators": 900, "max_depth": 4, "learning_rate": 0.035, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 3.0},
            {"n_estimators": 1200, "max_depth": 5, "learning_rate": 0.025, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 4.0},
            {"n_estimators": 900, "max_depth": 6, "learning_rate": 0.030, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 6.0},
            {"n_estimators": 1400, "max_depth": 3, "learning_rate": 0.025, "subsample": 0.90, "colsample_bytree": 0.95, "reg_lambda": 2.0},
            {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.050, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        ]
    return [{}]


def _candidate_params(model_name: str, cfg: dict) -> list[dict]:
    tuning = cfg.get("models", {}).get("tuning", {})
    configured = (
        tuning.get("candidates", {}).get(model_name)
        or cfg.get("models", {}).get(model_name, {}).get("candidates")
    )
    candidates = configured if configured else _default_candidates(model_name, cfg)
    if not candidates:
        candidates = [{}]
    max_candidates = tuning.get("max_candidates_per_model")
    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]
    return [dict(c) for c in candidates]


def _sample_training_rows(train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    model_cfg = cfg.get("models", {})
    seed = int(model_cfg.get("sample_seed", 42))
    per_basin = model_cfg.get("sample_per_basin_train")
    limit = model_cfg.get("sample_limit_train")

    sampled = train
    if per_basin is not None:
        n_per_basin = int(per_basin)
        pieces = []
        for basin_id, group in sampled.groupby("ID", sort=True):
            n = min(len(group), n_per_basin)
            random_state = (seed + int(basin_id) * 1009) % (2**32 - 1)
            pieces.append(group.sample(n=n, random_state=random_state) if len(group) > n else group)
        sampled = pd.concat(pieces, ignore_index=False).sort_index()

    if limit is not None:
        n_limit = int(limit)
        if len(sampled) > n_limit:
            sampled = sampled.sample(n=n_limit, random_state=seed).sort_index()

    return sampled


def _sample_training_rows_for_model(train: pd.DataFrame, cfg: dict, model_name: str) -> pd.DataFrame:
    sampled = _sample_training_rows(train, cfg)
    model_cfg = cfg.get("models", {}).get(model_name, {})
    peak_quantile = model_cfg.get("peak_flow_quantile")
    extra_per_basin = model_cfg.get("peak_flow_extra_per_basin")
    if peak_quantile is None or extra_per_basin is None:
        return sampled

    q = float(peak_quantile)
    n_extra = int(extra_per_basin)
    if not (0.0 < q < 1.0) or n_extra <= 0:
        return sampled

    seed = int(cfg.get("models", {}).get("sample_seed", 42))
    boosted = [sampled]
    for basin_id, group in train.groupby("ID", sort=True):
        flow = pd.to_numeric(group["q_mm_day"], errors="coerce")
        if flow.notna().sum() < 20:
            continue
        threshold = float(flow.quantile(q))
        peaks = group[flow >= threshold]
        if peaks.empty:
            continue
        random_state = (seed + int(basin_id) * 2017) % (2**32 - 1)
        boosted.append(
            peaks.sample(
                n=min(len(peaks), n_extra),
                replace=len(peaks) < n_extra,
                random_state=random_state,
            )
        )
    return pd.concat(boosted, ignore_index=False).sort_index()


def _feature_cols_for_model(feature_cols: list[str], cfg: dict, model_name: str) -> list[str]:
    model_cfg = cfg.get("models", {}).get(model_name, {})
    max_window = model_cfg.get("max_dynamic_window")
    if max_window is None:
        return feature_cols

    max_window = int(max_window)
    keep: list[str] = []
    for col in feature_cols:
        if "_sum_" in col or "_mean_" in col or "swe_delta_" in col:
            tail = col.rsplit("_", 1)[-1]
            if tail.isdigit() and int(tail) > max_window:
                continue
        keep.append(col)
    return keep


def _prediction_frame(model: Pipeline, rows: pd.DataFrame, feature_cols: list[str], cfg: dict, model_name: str) -> pd.DataFrame:
    yhat = inverse_target(model.predict(rows[feature_cols].astype(np.float32, copy=False)), cfg)
    part = rows[["ID", "date", "YYYY", "MM", "DD", "DOY", "water_year", "split", "q_mm_day", "prec"]].copy()
    part["model"] = model_name
    part["q_pred_mm_day"] = yhat
    return part


def _safe_metric(series: pd.Series, fallback: float) -> float:
    value = float(pd.to_numeric(series, errors="coerce").median())
    return value if np.isfinite(value) else fallback


def _candidate_score(metrics: pd.DataFrame) -> dict[str, float]:
    kge = pd.to_numeric(metrics["kge"], errors="coerce")
    nse = pd.to_numeric(metrics["nse"], errors="coerce")
    log_nse = pd.to_numeric(metrics["log_nse"], errors="coerce")
    pbias_abs = pd.to_numeric(metrics["pbias"], errors="coerce").abs()
    failure = ((kge < 0) | (nse < 0)).astype(float)
    median_kge = _safe_metric(kge, -1.0)
    q25_kge = float(kge.quantile(0.25)) if kge.notna().any() else -1.0
    median_nse = _safe_metric(nse, -1.0)
    median_log_nse = _safe_metric(log_nse, -1.0)
    median_abs_pbias = _safe_metric(pbias_abs, 100.0)
    failure_rate = float(failure.mean()) if len(failure) else 1.0
    score = (
        80.0 * (1.0 - median_kge)
        + 40.0 * (1.0 - q25_kge)
        + 25.0 * (1.0 - median_nse)
        + 20.0 * (1.0 - median_log_nse)
        + 1.2 * median_abs_pbias
        + 100.0 * failure_rate
    )
    return {
        "score": float(score),
        "median_kge": float(median_kge),
        "q25_kge": float(q25_kge),
        "median_nse": float(median_nse),
        "median_log_nse": float(median_log_nse),
        "median_abs_pbias": float(median_abs_pbias),
        "failure_rate": float(failure_rate),
    }


def _tuning_train_cfg(cfg: dict) -> dict:
    tuning = cfg.get("models", {}).get("tuning", {})
    tuned = copy.deepcopy(cfg)
    if "sample_per_basin_train" in tuning:
        tuned["models"]["sample_per_basin_train"] = tuning["sample_per_basin_train"]
    if "sample_limit_train" in tuning:
        tuned["models"]["sample_limit_train"] = tuning["sample_limit_train"]
    return tuned


def tune_tabular_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    out_dir: Path,
) -> dict:
    from paper4.metrics import evaluate_predictions

    tuning = cfg.get("models", {}).get("tuning", {})
    if not tuning.get("enabled", False):
        return cfg

    scoring_split = str(tuning.get("scoring_split", "val"))
    selected_path = out_dir / "tables" / "selected_hyperparameters.json"
    if tuning.get("reuse_existing", True) and selected_path.exists():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            selected = {}
        if selected:
            tuned_cfg = copy.deepcopy(cfg)
            selected = {
                model_name: _runtime_params(model_name, params, tuned_cfg)
                for model_name, params in selected.items()
            }
            for model_name, params in selected.items():
                if model_name in tuned_cfg.get("models", {}):
                    tuned_cfg["models"][model_name].update(params)
            print(
                f"[paper4] reusing selected tabular hyperparameters from {selected_path}",
                flush=True,
            )
            setup_summary_path = out_dir / "tables" / "model_setup_summary.json"
            setup_summary = json.loads(setup_summary_path.read_text(encoding="utf-8")) if setup_summary_path.exists() else {}
            setup_summary["tabular_models"] = {
                "selection_rule": "Validation-only candidate search; final test metrics are not used for model selection.",
                "scoring_split": scoring_split,
                "selected_hyperparameters": selected,
                "reused_existing_selection": True,
            }
            setup_summary_path.write_text(json.dumps(setup_summary, indent=2, sort_keys=True), encoding="utf-8")
            return tuned_cfg

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == scoring_split].copy()
    if train.empty or val.empty:
        print("[paper4] tabular tuning skipped: train or validation split is empty", flush=True)
        return cfg

    tune_cfg = _tuning_train_cfg(cfg)
    print(
        f"[paper4] tabular tuning on split={scoring_split} "
        f"train_rows={len(train):,} val_rows={len(val):,}",
        flush=True,
    )

    rows = []
    selected: dict[str, dict] = {}
    for model_name in _enabled_model_names(cfg):
        best_score = float("inf")
        best_params: dict | None = None
        candidates = _candidate_params(model_name, cfg)
        for i, params in enumerate(candidates, start=1):
            candidate_cfg = _update_model_params(tune_cfg, model_name, params)
            candidate_feature_cols = _feature_cols_for_model(feature_cols, candidate_cfg, model_name)
            train_fit = _sample_training_rows_for_model(train, candidate_cfg, model_name)
            model = _model_from_name(model_name, candidate_cfg)
            x_train = train_fit[candidate_feature_cols].astype(np.float32, copy=False)
            y_train = target_values(train_fit, candidate_cfg)
            model.fit(x_train, y_train)
            val_pred = _prediction_frame(model, val, candidate_feature_cols, candidate_cfg, model_name)
            metrics, _ = evaluate_predictions(val_pred)
            score_row = _candidate_score(metrics)
            full_params = dict(candidate_cfg["models"][model_name])
            full_params.pop("enabled", None)
            full_params.pop("candidates", None)
            row = {
                "model": model_name,
                "candidate": i,
                "scoring_split": scoring_split,
                "params": json.dumps(full_params, sort_keys=True),
                **score_row,
            }
            rows.append(row)
            print(
                f"[paper4] tuning {model_name} candidate={i}/{len(candidates)} "
                f"features={len(candidate_feature_cols)} train_rows={len(train_fit):,} "
                f"score={score_row['score']:.2f} median_kge={score_row['median_kge']:.3f} "
                f"q25_kge={score_row['q25_kge']:.3f} abs_pbias={score_row['median_abs_pbias']:.2f}",
                flush=True,
            )
            if score_row["score"] < best_score:
                best_score = score_row["score"]
                best_params = _runtime_params(model_name, full_params, cfg)
        if best_params is not None:
            selected[model_name] = best_params
            print(f"[paper4] selected {model_name} params: {json.dumps(best_params, sort_keys=True)}", flush=True)

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "tables" / "tuning_log.csv", index=False)
    if selected:
        (out_dir / "tables" / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        setup_summary_path = out_dir / "tables" / "model_setup_summary.json"
        setup_summary = json.loads(setup_summary_path.read_text(encoding="utf-8")) if setup_summary_path.exists() else {}
        setup_summary["tabular_models"] = {
            "selection_rule": "Validation-only candidate search; final test metrics are not used for model selection.",
            "scoring_split": scoring_split,
            "tuning_train_rows": int(len(_sample_training_rows(train, tune_cfg))),
            "full_train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "selected_hyperparameters": selected,
            "candidate_count": {
                model_name: int(sum(1 for r in rows if r["model"] == model_name))
                for model_name in selected
            },
        }
        setup_summary_path.write_text(json.dumps(setup_summary, indent=2, sort_keys=True), encoding="utf-8")
        tuned_cfg = copy.deepcopy(cfg)
        for model_name, params in selected.items():
            tuned_cfg["models"][model_name].update(params)
        return tuned_cfg
    return cfg


def train_and_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = tune_tabular_models(df, feature_cols, cfg, out_dir)
    fit_splits = cfg.get("models", {}).get("fit_splits", ["train"])
    train = df[df["split"].isin(fit_splits)].copy()
    pred_frames = []
    importance_frames = []

    for name, model in enabled_models(cfg).items():
        model_feature_cols = _feature_cols_for_model(feature_cols, cfg, name)
        train_fit = _sample_training_rows_for_model(train, cfg, name)
        if len(train_fit) != len(train) or len(model_feature_cols) != len(feature_cols):
            print(
                f"[paper4] {name} training sample rows={len(train_fit):,}/{len(train):,} features={len(model_feature_cols)}",
                flush=True,
            )
        X_train = train_fit[model_feature_cols].astype(np.float32, copy=False)
        y_train = target_values(train_fit, cfg)
        model.fit(X_train, y_train)
        if cfg.get("artifacts", {}).get("save_models", True):
            joblib.dump(model, out_dir / "models" / f"{name}.joblib")

        pred_splits = cfg.get("evaluation", {}).get("prediction_splits", ["train", "val", "test", "spatial_test"])
        for split, g in df[df["split"].isin(pred_splits)].groupby("split"):
            pred_frames.append(_prediction_frame(model, g, model_feature_cols, cfg, name))

        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            importance_frames.append(pd.DataFrame({
                "model": name,
                "feature": model_feature_cols,
                "importance": estimator.feature_importances_,
            }).sort_values("importance", ascending=False))

    if not pred_frames:
        return pd.DataFrame(), pd.DataFrame()

    preds = pd.concat(pred_frames, ignore_index=True)
    imps = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return preds, imps
