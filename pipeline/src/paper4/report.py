from __future__ import annotations

from pathlib import Path

import pandas as pd


FIGURE_BLOCKS = [
    ("fig00_study_domain.png",
     "Figure 1. LamaH-CE study domain and selected catchment set. Catchment locations "
     "are shown with the active modelling subset and available hydroclimatic gradients."),
    ("fig01_skill_cdf.png",
     "Figure 1. Cumulative distribution of catchment skill (KGE, NSE, log NSE, PBIAS) for "
     "each available diagnostic model on the validation, temporal test, and spatial-transfer "
     "splits. Per-panel inset reports the median for every model. Curves shifted toward "
     "the upper right (or zero PBIAS) indicate higher skill."),
    ("fig02_transfer_skill.png",
     "Figure 2. Temporal versus spatial transfer skill. Paired bars compare median KGE, "
     "NSE, log NSE, and PBIAS on the temporal test catchments and the spatially held-out "
     "catchments. Solid bars denote temporal extrapolation; hatched bars denote spatial "
     "transfer to basins excluded from fitting."),
    ("fig02_basin_map_kge.png",
     "Figure 3. Geographic distribution of test KGE per model across the LamaH-CE Central "
     "European domain, including the COSERO conceptual benchmark (Lambert Azimuthal Equal Area projection, Natural Earth basemap). "
     "Inset locates the study area within Europe. The diverging colour scale (centred at "
     "KGE = 0.5) highlights the process-controlled skill gradient."),
    ("fig03_signature_errors.png",
     "Figure 4. Median absolute error of twelve runoff signatures by model on the test "
     "split. Lower bars indicate more faithful signature reproduction across catchments."),
    ("fig04_regime_violins.png",
     "Figure 5. Test-split KGE distribution stratified by snow regime, aridity, "
     "subsurface storage, and human alteration. Violins show the full distribution; "
     "white circles mark the median. Catchment counts beneath each category."),
    ("fig05_skill_vs_attribute.png",
     "Figure 6. Test KGE conditioned on the four most diagnostic catchment attributes — "
     "snow fraction, aridity, baseflow index, log-area. Translucent points are basins; "
     "thick lines are LOWESS smoothers per model. Per-model Spearman ρ in each panel."),
    ("fig06_hydrographs.png",
     "Figure 7. Daily observed (black) and simulated (model colours) hydrographs at six "
     "representative LamaH-CE catchments — one per hydrological process regime. "
     "Precipitation is shown as inverted blue bars from the top axis. The displayed "
     "window is centred on the most hydrologically active test year. Per-model KGE "
     "is reported on the right margin."),
    ("fig07_flow_duration_curves.png",
     "Figure 8. Flow-duration curves at representative catchments (log y-axis). "
     "Observed FDC in black; per-model simulations in reference colours. Q$_{05}$ and "
     "Q$_{95}$ exceedance probabilities are marked."),
    ("fig08_signature_obs_vs_sim.png",
     "Figure 9. Per-catchment observed vs simulated runoff signatures (test split) for "
     "six core signatures most relevant to regional rainfall-runoff modelling. Q$_{mean}$, "
     "Q$_{05}$, Q$_{95}$ on log–log axes. Per-model Spearman ρ shown in each panel."),
    ("fig09_storm_event_zoom.png",
     "Figure 10. Storm-event zoom (±60 days around the largest observed peak) at four "
     "representative basins, with precipitation overlay. Reveals event-timing errors "
     "and peak-magnitude bias hidden by aggregate metrics."),
    ("fig10_feature_importance.png",
     "Figure 11. Top feature importances for the two tree-based diagnostic models. "
     "Compares which forcings, antecedent windows, and static attributes drive Random "
     "Forest regime partitioning vs XGBoost threshold-interaction skill."),
    ("fig11_signature_winners.png",
     "Figure 12. Number of runoff signatures best preserved by each model on the test "
     "split. Counts the diagnostic niche of each architecture."),
    ("fig12_human_impact_pbias.png",
     "Figure 13. Median PBIAS by human-alteration class for each model on the test split, "
     "with catchment counts beneath each category. Systematic non-zero residuals identify "
     "regimes where observed discharge departs from the natural climate–catchment response."),
    ("fig13_hydrograph_mosaic.png",
     "Figure 14. Compact rainfall-runoff hydrograph mosaic across representative process "
     "regimes. Each panel shows the most hydrologically active test year for one basin, "
     "with precipitation as inverted blue bars and discharge in mm d$^{-1}$."),
    ("fig14_kge_pbias_landscape.png",
     "Figure 15. Joint KGE-PBIAS skill landscape by model. The shaded band marks "
     "+/-10 % PBIAS and the vertical reference marks KGE = 0.5."),
    ("fig15_regime_kge_heatmap.png",
     "Figure 16. Median test KGE by hydrological regime and model, showing where model "
     "skill changes with snow influence, aridity, storage, and human alteration."),
    ("fig16_runoff_coefficient.png",
     "Figure 17. Observed versus simulated runoff coefficient $Q/P$ for each model. "
     "The +/-20 % envelope diagnoses water-balance consistency rather than only timing skill."),
    ("fig17a_hydrograph_atlas.png",
     "Figure 18a. Rainfall-runoff atlas part 1 — eight representative basins from the "
     "snow / aridity / storage process zones. Each panel shows the most hydrologically "
     "active test year with precipitation overlay. Exposes missed peaks, delayed "
     "recession, and low-flow bias across regimes."),
    ("fig17b_hydrograph_atlas.png",
     "Figure 18b. Rainfall-runoff atlas part 2 — eight further basins covering geology, "
     "size, and human-impact regimes. Same conventions as part 1; the two parts are "
     "designed to be read side-by-side."),
    ("fig18_signature_bias_ratios.png",
     "Figure 19. Signed relative bias in hydrological ratios and flow-regime signatures. "
     "Panel (a) shows median and interquartile range; panel (b) reports the fraction of "
     "catchments within +/-20 % signature bias."),
    ("fig19_flood_typology_seasonality.png",
     "Figure 20. Observed flood typologies derived from basin-specific high-flow events. "
     "Panel (a) shows seasonal event composition; panel (b) positions each flood type "
     "by event temperature and 7-day precipitation, highlighting snow-transition and "
     "warm-season intensity mechanisms relevant to climate-change interpretation."),
    ("fig20_flood_typology_model_skill.png",
     "Figure 21. Model performance conditioned on observed flood type. Peak-ratio bias "
     "and absolute peak-timing error diagnose whether a model fails because of flood "
     "magnitude, event timing, or typology-specific hydrological memory."),
    ("fig21_snow_timing_error.png",
     "Figure 22. Snow-timing error diagnostic. Panel (a) plots the signed half-flow-date "
     "error (sim − obs, days) against snow fraction, with LOWESS smoothers per model. "
     "Panel (b) shows the absolute HFD error distribution stratified by snow-regime "
     "class. Reveals where xLSTM trades early-spring melt timing against high-snow skill."),
    ("fig22_predictability_zones.png",
     "Figure 23. Hydrological predictability-zones synthesis. Panel (a): each catchment "
     "is classified into one of five process-controlled regimes from its static attributes "
     "(snow fraction, baseflow index, aridity, human impact); marker shape indicates the "
     "model that achieves the highest test KGE. Panel (b): zone composition with median "
     "best-model KGE. Identifies where regional rainfall-runoff is transferable and where "
     "human alteration limits skill."),
    ("fig23_conceptual_framework.png",
     "Figure 24. Conceptual framework placing the four diagnostic models in process-control "
     "space. The horizontal axis is temporal-memory requirement; the vertical axis is "
     "process abstraction (threshold/regime → continuous storage). Each model occupies a "
     "diagnostic niche, with the AutoResearch loop binding the experiments together."),
    ("fig25_transfer_kge_maps.png",
     "Figure 25. Geographic comparison of temporal-test and spatial-transfer KGE by model. "
     "Rows separate temporal extrapolation from spatial holdout basins; columns separate "
     "model instruments. This figure shows whether transfer skill is spatially coherent "
     "or concentrated in specific hydroclimatic regions."),
    ("fig26_memory_gain_maps.png",
     "Figure 26. Spatial distribution of sequence-model gain. Panel columns show xLSTM "
     "minus Random Forest and best-sequence minus best-tree KGE; rows separate temporal "
     "test and spatial holdout. Blue values indicate where sequence memory improves "
     "regional rainfall-runoff prediction; red values indicate tree-model advantage."),
    ("fig27_transfer_regime_delta.png",
     "Figure 27. Regime-level transfer robustness. Bars show spatial-test median KGE minus "
     "temporal-test median KGE by snow, aridity, storage, and human-impact regime. Values "
     "near zero indicate robust transfer; negative values identify regimes where spatial "
     "generalisation is weaker than temporal extrapolation."),
    ("fig28_seasonal_residual_cycle.png",
     "Figure 28. Seasonal residual structure. Monthly PBIAS lines and heatmaps diagnose "
     "when each model creates systematic wet or dry bias on the temporal and spatial "
     "test splits. This separates annual aggregate skill from snowmelt, summer-storm, "
     "and recession-season errors."),
    ("fig29_budyko_aridity_partitioning.png",
     "Figure 29. Budyko-style runoff partitioning. Observed and simulated runoff "
     "coefficients are conditioned on aridity and snow fraction to test whether models "
     "preserve water-balance partitioning across humid, moderate, and dry regimes."),
    ("fig30_memory_gradient_diagnostics.png",
     "Figure 30. Hydrological memory-gain gradients. Sequence-model gain over the best "
     "tree model is plotted against snow fraction, baseflow index, FDC slope, and aridity "
     "for temporal and spatial splits. Positive values indicate regimes where explicit "
     "sequence memory improves transferability."),
    ("fig31_spatial_novelty_transfer.png",
     "Figure 31. Spatial novelty and transfer failure. Hydrological novelty is the "
     "standardised distance from each spatial holdout basin to its nearest temporal-test "
     "basin in process-attribute space. Panels diagnose whether out-of-distribution "
     "catchments require memory and where spatial transfer skill degrades."),
    ("fig32_climate_sensitivity_flood_errors.png",
     "Figure 32. Climate-sensitive flood-error anatomy. Event counts, peak-ratio bias, "
     "event-volume ratio, and high-flow threshold hit rate are summarised by climate "
     "sensitivity class, linking model errors to warm-season rainfall intensity, "
     "antecedent wetness, snow-transition, and cool-season rainfall mechanisms."),
    ("fig33_cosero_benchmark_dashboard.png",
     "Figure 33. COSERO benchmark dashboard. Median KGE, basin-level data-driven gain "
     "over COSERO, parity against COSERO, and signature-ranking diagnostics place the "
     "machine-learning models against the LamaH-CE conceptual hydrological benchmark."),
    ("fig34_transferability_dashboard.png",
     "Figure 34. Transferability dashboard. Temporal-test and spatial-holdout skill are "
     "compared with slopegraphs, generalisation deltas, reliability thresholds, and "
     "process-regime coverage of the temporal and spatial evaluation sets."),
    ("fig35_cosero_transfer_maps.png",
     "Figure 35. COSERO transfer benchmark maps. COSERO and xLSTM KGE are mapped for "
     "the temporal and spatial splits, together with basin-wise xLSTM minus COSERO "
     "KGE, showing where the regional neural model exceeds or falls short of the "
     "conceptual benchmark."),
    ("fig36_training_tuning_audit.png",
     "Figure 36. Internal model-selection audit. Sequence-model panels show train and "
     "validation loss by epoch, the best-validation checkpoint, and early-stopping "
     "behaviour. Tabular panels show validation-only hyperparameter candidate scores "
     "for Random Forest and XGBoost."),
    ("fig37_model_setup_audit.png",
     "Figure 37. Actual model setup audit. Sequence-model parameter counts, lookback, "
     "best epoch, and final epoch are shown together with validation-selected tabular "
     "hyperparameters. This figure documents the fitted model configuration used for "
     "paper results."),
    ("fig38_uncertainty_dashboard.png",
     "Figure 38. Uncertainty audit. Basin-bootstrap confidence intervals, paired model "
     "contrast intervals, and spatial-fold stability are shown where available. Fold "
     "stability is populated after the five-fold spatial aggregate is created."),
]


MAIN_FIGURES = [
    "fig00_study_domain.png",
    "fig01_skill_cdf.png",
    "fig33_cosero_benchmark_dashboard.png",
    "fig34_transferability_dashboard.png",
    "fig38_uncertainty_dashboard.png",
    "fig06_hydrographs.png",
    "fig03_signature_errors.png",
    "fig04_regime_violins.png",
    "fig29_budyko_aridity_partitioning.png",
    "fig30_memory_gradient_diagnostics.png",
    "fig32_climate_sensitivity_flood_errors.png",
    "fig12_human_impact_pbias.png",
]


SUPPLEMENT_FIGURES = [
    fname for fname, _ in FIGURE_BLOCKS if fname not in set(MAIN_FIGURES)
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows available._"
    small = df.head(max_rows).copy()
    small = small.fillna("")
    cols = list(small.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in small.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _eval_split(df: pd.DataFrame) -> str:
    seen = set(df["split"].dropna().astype(str).unique()) if "split" in df.columns else set()
    for split in ("test", "spatial_test", "val"):
        if split in seen:
            return split
    return "test"


def _figure_section(figures: Path, filenames: list[str], captions: dict[str, str]) -> str:
    lines = []
    for fname in filenames:
        if (figures / fname).exists():
            lines.append(f"![{Path(fname).stem}](figures/{fname})\n\n*{captions[fname]}*")
    return "\n\n".join(lines) if lines else "_No selected figures available._"


def write_report(out_dir: Path, cfg: dict):
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    metrics = _read_csv(tables / "metrics_by_basin.csv")
    sig = _read_csv(tables / "signature_errors.csv")
    idx = _read_csv(tables / "catchment_index.csv")
    imp = _read_csv(tables / "feature_importance.csv")
    deltas = _read_csv(tables / "model_delta_diagnostics.csv")
    zones = _read_csv(tables / "process_zone_summary.csv")
    ranks = _read_csv(tables / "signature_model_ranking.csv")
    corr = _read_csv(tables / "process_control_correlations.csv")
    flood_summary = _read_csv(tables / "flood_typology_summary.csv")
    flood_model_summary = _read_csv(tables / "flood_typology_model_summary.csv")
    metric_ci = _read_csv(tables / "metric_uncertainty.csv")
    contrast_ci = _read_csv(tables / "model_contrast_uncertainty.csv")
    signature_ci = _read_csv(tables / "signature_error_uncertainty.csv")
    fold_summary = _read_csv(tables / "spatial_fold_summary.csv")
    training_history = _read_csv(tables / "sequence_training_history.csv")
    tuning_log = _read_csv(tables / "tuning_log.csv")
    findings = (
        (tables / "hydrologic_findings.md").read_text(encoding="utf-8")
        if (tables / "hydrologic_findings.md").exists()
        else ""
    )

    summary = pd.DataFrame()
    if not metrics.empty:
        summary = (
            metrics.groupby(["model", "split"])
            .agg(
                basins=("ID", "nunique"),
                median_kge=("kge", "median"),
                median_nse=("nse", "median"),
                median_log_nse=("log_nse", "median"),
                median_pbias=("pbias", "median"),
            )
            .reset_index()
        )

    metric_ci_focus = pd.DataFrame()
    if not metric_ci.empty:
        metric_ci_focus = metric_ci[
            metric_ci["metric"].isin(["kge", "nse", "log_nse", "pbias"])
        ].sort_values(["split", "metric", "model"])

    contrast_ci_focus = pd.DataFrame()
    if not contrast_ci.empty:
        contrast_ci_focus = contrast_ci[
            contrast_ci["contrast"].isin([
                "xgboost_gain_vs_cosero",
                "xlstm_gain_vs_cosero",
                "transformer_gain_vs_cosero",
                "memory_gain_xlstm_vs_rf",
                "sequence_gain_vs_tree",
            ])
        ].sort_values(["split", "contrast"])

    signature_ci_focus = pd.DataFrame()
    if not signature_ci.empty:
        signature_ci_focus = signature_ci.sort_values(["split", "signature", "median_abs_error"]).groupby(
            ["split", "signature"], as_index=False
        ).head(2)

    sig_summary = pd.DataFrame()
    if not sig.empty:
        cols = [c for c in sig.columns if c.startswith("abs_err_")]
        sig_summary = sig.groupby(["model", "split"])[cols].median().reset_index()

    training_summary = pd.DataFrame()
    if not training_history.empty:
        rows = []
        for model, g in training_history.sort_values("epoch").groupby("model"):
            best = g.loc[g["val_loss"].idxmin()] if "val_loss" in g.columns else g.iloc[-1]
            last = g.iloc[-1]
            rows.append({
                "model": model,
                "epochs_run": int(last["epoch"]),
                "best_epoch": int(best["epoch"]),
                "best_val_loss": float(best["val_loss"]) if "val_loss" in best else "",
                "final_train_loss": float(last["train_loss"]) if "train_loss" in last else "",
                "final_val_loss": float(last["val_loss"]) if "val_loss" in last else "",
                "loss": last.get("loss", ""),
                "early_stopping_patience": cfg.get("sequence_models", {}).get("early_stopping_patience", ""),
            })
        training_summary = pd.DataFrame(rows)

    tuning_summary = pd.DataFrame()
    if not tuning_log.empty:
        tuning_summary = (
            tuning_log.sort_values("score")
            .groupby("model", as_index=False)
            .head(1)
            .sort_values("model")
        )

    delta_summary = pd.DataFrame()
    if not deltas.empty:
        delta_cols = [
            "xgboost_gain_vs_cosero",
            "xlstm_gain_vs_cosero",
            "transformer_gain_vs_cosero",
            "memory_gain_xlstm_vs_rf",
            "transformer_gain_vs_rf",
            "sequence_gain_vs_tree",
            "tree_advantage",
            "xlstm_minus_transformer",
        ]
        delta_summary = (
            deltas.groupby("split")[[c for c in delta_cols if c in deltas.columns]]
            .median()
            .reset_index()
        )

    eval_split = _eval_split(metrics)

    zone_test = pd.DataFrame()
    if not zones.empty:
        zone_test = zones[zones["split"].eq(eval_split)].sort_values(
            ["zone", "zone_value", "model"]
        )

    rank_test = pd.DataFrame()
    if not ranks.empty:
        rank_test = ranks[(ranks["split"].eq(eval_split)) & (ranks["rank"].le(2))].sort_values(
            ["signature", "rank"]
        )

    corr_top = pd.DataFrame()
    if not corr.empty:
        corr_top = corr[corr["split"].eq(eval_split)].head(20)

    flood_test = pd.DataFrame()
    if not flood_summary.empty:
        flood_test = flood_summary[flood_summary["split"].eq(eval_split)].sort_values(
            ["event_type", "season"]
        )

    flood_model_test = pd.DataFrame()
    if not flood_model_summary.empty:
        flood_model_test = flood_model_summary[flood_model_summary["split"].eq(eval_split)].sort_values(
            ["event_type", "model"]
        )

    captions = dict(FIGURE_BLOCKS)
    main_figure_section = _figure_section(figures, MAIN_FIGURES, captions)
    supplement_figure_section = _figure_section(figures, SUPPLEMENT_FIGURES, captions)

    text = f"""# Paper 4 Diagnostic Report

Run: `{cfg['project']['run_name']}`

Config: `{cfg.get('_config_path', '')}`

## Catchment Set

- Selected catchments: {len(idx)}
- Basin delineation: `{cfg['data']['basin_set']}`
- Time step: `{cfg['data']['time_step']}`
- Target: `{cfg['target']['variable']}` with `{cfg['target']['transform']}` transform

## Hydrograph Skill Summary

{_md_table(summary, 50)}

## Signature Consistency Summary

{_md_table(sig_summary, 50)}

## Diagnostic Model Contrasts

{_md_table(delta_summary, 20)}

## Model-Selection and Training Audit

Sequence models use the promoted regional neural-hydrology setup: long lookback,
static catchment attributes, per-epoch shuffled sequence batches, basin-normalised
NSE loss when configured, validation checkpoint selection, cosine learning-rate
schedule, gradient clipping, and early stopping. Tabular models are selected by
validation-only candidate search; final temporal and spatial test metrics are not
used for hyperparameter selection.

### Sequence Training Convergence

{_md_table(training_summary, 20)}

### Tabular Hyperparameter Selection

{_md_table(tuning_summary, 20)}

## Uncertainty and Fold Stability

Bootstrap intervals are basin-level non-parametric 95 % confidence intervals around medians. Spatial fold stability is populated only for aggregated multi-fold spatial-transfer outputs.

### Median Skill Uncertainty

{_md_table(metric_ci_focus, 80)}

### Paired Model-Contrast Uncertainty

{_md_table(contrast_ci_focus, 60)}

### Signature-Error Uncertainty

{_md_table(signature_ci_focus, 80)}

### Spatial-Fold Stability

{_md_table(fold_summary, 60)}

## Process-Zone Skill Summary

{_md_table(zone_test, 80)}

## Runoff-Signature Model Ranking

{_md_table(rank_test, 80)}

## Process-Control Correlations

{_md_table(corr_top, 30)}

## Flood Typology Summary

{_md_table(flood_test, 80)}

## Flood Typology Model Skill

{_md_table(flood_model_test, 80)}

## Top Feature Importances

{_md_table(imp.groupby('model').head(12) if not imp.empty else imp, 60)}

## Auto-Interpreted Hydrologic Findings

{findings}

## Recommended Main-Paper Figures

{main_figure_section}

## Supplementary Figure Candidates

{supplement_figure_section}

## Interpretation Checklist

- Does strong KGE/NSE also imply low runoff-signature error?
- Are low-flow-sensitive metrics weaker than all-flow metrics?
- Do impacted or transfer-affected basins show systematic bias?
- Which flood typologies are missed: rain-on-snow, snowmelt, warm-season short rain, or wet-soil long rain?
- Are peak errors magnitude errors, timing errors, or climate-regime errors?
- Which attributes dominate RF/XGBoost explanations: climate, snow, storage, geology, land cover, or human impact?
- Which basins should be flagged for xLSTM/Transformer sequence experiments?
"""

    (out_dir / "paper4_diagnostic_report.md").write_text(text, encoding="utf-8")
