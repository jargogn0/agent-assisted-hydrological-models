"""HESS publication-grade visual style.

The palette is intentionally restrained: black, cool greys, two blues, and one
red accent. There are no green/brown model colours. The aim is to make the
figures read as hydrological rainfall-runoff diagnostics, not dashboard plots.

Typography: Helvetica/Arial 8.5 pt body, 9.5 pt panel titles, oriented for the
single-column (84 mm) and double-column (174 mm) HESS layout.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


# ─── HESS hydrology palette ──────────────────────────────────────────────────
# Conservative monochromatic-with-accent set — black, two blues, slate, red.
# Chosen to read as a serious rainfall-runoff figure rather than a dashboard.
# Tested for B/W print, deuteranopia, protanopia.
HESS_PALETTE = {
    "ink":       "#000000",   # observed
    "slate":     "#586471",   # slate grey       — random forest (oldest baseline)
    "navy":      "#0E2A55",   # deep navy        — xgboost
    "marine":    "#1F77BD",   # marine blue      — xLSTM
    "crimson":   "#B5302A",   # firebrick red    — transformer
    "rain":      "#486B9A",   # muted marker blue — precipitation overlay
    "land":      "#F1F4F7",   # cool paper grey   — land background
    "haze":      "#DDE8F0",   # cool haze         — ocean / lakes
    "border":    "#7A8590",   # state borders
    "river":     "#9FB7CC",   # river polylines
    "grid":      "#E8E8E8",   # very faint grid
    "axis":      "#1F1F1F",   # axis spine + tick
    "annot":     "#1F1F1F",   # in-axes annotations
    "rule":      "#B5B5B5",   # threshold rules (KGE=0.5 etc.)
    "muted":     "#9CA3AB",   # muted tone for neutral elements
}

# Paul Tol "vibrant" scheme — the modern geoscience-journal standard
# (colourblind-safe, B/W-print safe, high contrast without garishness).
# Semantics: conceptual benchmark black; tree baselines cool (grey, blue);
# the sequence models carry the accents — xLSTM in hydrological teal,
# Transformer in warm orange.
MODEL_COLORS = {
    "cosero":        "#000000",
    "random_forest": "#BBBBBB",   # neutral grey  — baseline
    "xgboost":       "#0077BB",   # strong blue
    "xlstm":         "#009988",   # teal
    "transformer":   "#EE7733",   # orange
}

OBS_COLOR    = HESS_PALETTE["ink"]
PRECIP_COLOR = HESS_PALETTE["rain"]
SWE_COLOR    = "#A9CBE2"
LAND_COLOR   = HESS_PALETTE["land"]
OCEAN_COLOR  = HESS_PALETTE["haze"]
BORDER_COLOR = HESS_PALETTE["border"]
RIVER_COLOR  = HESS_PALETTE["river"]
GRID_COLOR   = HESS_PALETTE["grid"]
ANNOT_COLOR  = HESS_PALETTE["annot"]
RULE_COLOR   = HESS_PALETTE["rule"]
AXIS_COLOR   = HESS_PALETTE["axis"]
SLATE_COLOR  = HESS_PALETTE["slate"]
MUTED_COLOR  = HESS_PALETTE["muted"]

# Diverging colormap for skill maps: red = poor / failure, blue = high skill.
# This keeps hydrological "good water signal" figures visually consistent.
SKILL_CMAP = "RdBu"

# HESS / Copernicus column widths in inches.
COL_SINGLE = 3.31    # 84 mm  — single column
COL_HALF15 = 4.72    # 120 mm — 1.5 column
COL_DOUBLE = 6.85    # 174 mm — double column / page width

# Categorical palettes for regime classes (used in study-domain map etc.)
REGIME_PALETTES = {
    "snow_class": {
        "low_snow":         "#737E88",   # cool grey  — little snow
        "mixed_snow":       "#4E7FAE",   # mid blue   — mixed
        "snow_influenced":  "#0E2A55",   # navy       — snow-dominated
    },
    "aridity_class": {
        "humid":            "#1F77BD",   # marine blue
        "moderate_aridity": "#9CA3AB",   # neutral grey
        "dry":              "#B5302A",   # red
    },
    "impact_class": {
        "near_natural":     "#1F77BD",
        "impact_s":         "#7AA9D6",
        "impact_m":         "#9CA3AB",
        "impact_l":         "#B5302A",
        "transfer":         "#0E2A55",
        "impact_unknown":   "#D5D5D5",
    },
    "size_class": {
        "small":            "#9CA3AB",
        "medium":           "#1F77BD",
        "large":            "#0E2A55",
    },
}

MODEL_LABELS = {
    "cosero":        "COSERO",
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "xlstm":         "xLSTM",
    "transformer":   "Transformer",
}

MODEL_ORDER = ["cosero", "random_forest", "xgboost", "xlstm", "transformer"]

FLOOD_TYPE_ORDER = [
    "rain_on_snow",
    "snowmelt",
    "warm_season_short_rain",
    "long_rain_wet_soil",
    "cool_season_rain",
    "storage_release",
    "rainfall_runoff",
]

FLOOD_TYPE_LABELS = {
    "rain_on_snow": "rain-on-snow",
    "snowmelt": "snowmelt",
    "warm_season_short_rain": "warm-season short rain",
    "long_rain_wet_soil": "long rain / wet soil",
    "cool_season_rain": "cool-season rain",
    "storage_release": "storage release",
    "rainfall_runoff": "rainfall-runoff",
}

FLOOD_TYPE_COLORS = {
    "rain_on_snow": "#B5302A",
    "snowmelt": "#0E2A55",
    "warm_season_short_rain": "#1F77BD",
    "long_rain_wet_soil": "#486B9A",
    "cool_season_rain": "#586471",
    "storage_release": "#9CA3AB",
    "rainfall_runoff": "#111111",
}

ZONE_LABELS = {
    "snow_class":     "Snow regime",
    "aridity_class":  "Aridity regime",
    "storage_class":  "Subsurface storage",
    "geology_class":  "Geological permeability",
    "impact_class":   "Human alteration",
    "size_class":     "Catchment size",
}

ZONE_VALUE_ORDER = {
    "snow_class":     ["low_snow", "mixed_snow", "snow_influenced"],
    "aridity_class":  ["humid", "moderate_aridity", "dry"],
    "storage_class":  ["low_storage", "mixed_storage", "high_storage", "unknown"],
    "geology_class":  ["low_perm", "mixed_perm", "high_perm", "unknown"],
    "impact_class":   ["near_natural", "impact_s", "impact_m", "impact_l", "transfer", "impact_unknown"],
    "size_class":     ["small", "medium", "large"],
}

ZONE_VALUE_PRETTY = {
    "low_snow":            "low-snow",
    "mixed_snow":          "mixed-snow",
    "snow_influenced":     "snow-influenced",
    "humid":               "humid",
    "moderate_aridity":    "moderate",
    "dry":                 "dry",
    "low_storage":         "low storage",
    "mixed_storage":       "mixed storage",
    "high_storage":        "high storage",
    "low_perm":            "low perm.",
    "mixed_perm":          "mixed perm.",
    "high_perm":           "high perm.",
    "unknown":             "unknown",
    "near_natural":        "near-natural",
    "impact_s":            "small impact",
    "impact_m":            "medium impact",
    "impact_l":            "large impact",
    "impact_unknown":      "unknown impact",
    "transfer":            "transfer",
    "small":               "small",
    "medium":              "medium",
    "large":               "large",
}

SIG_LABELS = {
    "abs_err_q_mean":          "mean discharge",
    "abs_err_runoff_ratio":    "runoff ratio",
    "abs_err_fdc_slope":       "FDC slope",
    "abs_err_baseflow_index":  "baseflow index",
    "abs_err_hfd_mean":        "half-flow date (d)",
    "abs_err_q05":             "Q05",
    "abs_err_q95":             "Q95",
    "abs_err_high_q_freq":     "high-flow freq.",
    "abs_err_high_q_dur":      "high-flow dur.",
    "abs_err_low_q_freq":      "low-flow freq.",
    "abs_err_low_q_dur":       "low-flow dur.",
    "abs_err_zero_q_freq":     "zero-flow freq.",
}

SIG_PRETTY = {
    "q_mean":          "mean discharge $\\bar{Q}$",
    "runoff_ratio":    "runoff ratio $R_q$",
    "fdc_slope":       "FDC slope",
    "baseflow_index":  "baseflow index",
    "hfd_mean":        "half-flow date",
    "q05":             "$Q_{05}$",
    "q95":             "$Q_{95}$",
    "high_q_freq":     "high-flow freq.",
    "high_q_dur":      "high-flow duration",
    "low_q_freq":      "low-flow freq.",
    "low_q_dur":       "low-flow duration",
    "zero_q_freq":     "zero-flow freq.",
}

SPLIT_DISPLAY = {"val": "Validation", "test": "Test", "spatial_test": "Spatial test", "train": "Train"}


def apply_style() -> None:
    plt.rcdefaults()
    mpl.rcParams.update({
        # Output quality
        "figure.dpi":          140,
        "savefig.dpi":         600,
        "savefig.bbox":        None,
        "savefig.pad_inches":  0.04,
        "savefig.facecolor":   "white",
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",

        # Typography (HESS / Copernicus styling: 8–9 pt main, sans-serif)
        "font.family":         ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           8.5,
        "axes.titlesize":      9.0,
        "axes.titleweight":    "regular",
        "axes.titlepad":       4.5,
        "axes.titlelocation":  "left",
        "axes.labelsize":      8.5,
        "axes.labelpad":       2.5,
        "axes.labelweight":    "regular",
        "axes.linewidth":      0.6,

        # Spines and ticks
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.edgecolor":      AXIS_COLOR,
        "axes.labelcolor":     "#111111",
        "axes.titlecolor":     "#111111",
        "xtick.color":         AXIS_COLOR,
        "ytick.color":         AXIS_COLOR,
        "xtick.labelsize":     7.5,
        "ytick.labelsize":     7.5,
        "xtick.major.width":   0.55,
        "ytick.major.width":   0.55,
        "xtick.major.size":    2.8,
        "ytick.major.size":    2.8,
        "xtick.minor.size":    1.4,
        "ytick.minor.size":    1.4,
        "xtick.minor.width":   0.4,
        "ytick.minor.width":   0.4,
        "xtick.direction":     "out",
        "ytick.direction":     "out",

        # Legends
        "legend.frameon":      True,
        "legend.facecolor":    "white",
        "legend.edgecolor":    "#D0D0D0",
        "legend.framealpha":   0.98,
        "legend.fontsize":     7.5,
        "legend.title_fontsize": 7.8,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.3,
        "legend.borderaxespad": 0.4,

        # Grid
        "grid.color":          GRID_COLOR,
        "grid.linewidth":      0.45,
        "grid.alpha":          0.9,
        "grid.linestyle":      "-",

        # Lines
        "lines.linewidth":     1.2,
        "lines.solid_capstyle": "round",
        "patch.linewidth":     0.4,

        # Default colormap
        "image.cmap":          "viridis",

        # Math text
        "mathtext.fontset":    "stix",
    })


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#888888")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def palette_for(models: Sequence[str]) -> list[str]:
    return [model_color(m) for m in models]


def zone_value_pretty(value: str) -> str:
    return ZONE_VALUE_PRETTY.get(value, value)


def signature_pretty(value: str) -> str:
    return SIG_PRETTY.get(value, value.replace("_", " "))
