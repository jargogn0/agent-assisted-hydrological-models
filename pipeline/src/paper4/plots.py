"""HESS publication-grade figure stack for the LamaH-CE paper.

Conventions:
  * Curated HESS palette (style.py). Each model keeps its hue everywhere.
  * Observed: solid black, slightly thicker line, behind models on hydrographs.
  * Twin-axis hydrographs: precipitation as inverted bars from the top axis.
  * Maps: Lambert Azimuthal Equal Area (cartopy when available), RdBu_r diverging
    colormap centered on KGE = 0.5, prints to grayscale legibly.
  * Trend lines: LOWESS with thin band; never linear regression on KGE.
  * Each multi-panel figure carries (a), (b), … panel labels in the upper-left.
  * Figures emit 600-dpi PNG and vector PDF in pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from paper4.plot_data import export_plot_data
from paper4.style import (
    ANNOT_COLOR,
    AXIS_COLOR,
    BORDER_COLOR,
    COL_DOUBLE,
    COL_HALF15,
    COL_SINGLE,
    FLOOD_TYPE_COLORS,
    FLOOD_TYPE_LABELS,
    FLOOD_TYPE_ORDER,
    GRID_COLOR,
    LAND_COLOR,
    MODEL_LABELS,
    MODEL_ORDER,
    MUTED_COLOR,
    OBS_COLOR,
    OCEAN_COLOR,
    PRECIP_COLOR,
    REGIME_PALETTES,
    RIVER_COLOR,
    RULE_COLOR,
    SIG_LABELS,
    SKILL_CMAP,
    SLATE_COLOR,
    SPLIT_DISPLAY,
    ZONE_LABELS,
    ZONE_VALUE_ORDER,
    ZONE_VALUE_PRETTY,
    apply_style,
    model_color,
    model_label,
    palette_for,
    signature_pretty,
    zone_value_pretty,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

PAPER_FIGSIZE = (COL_DOUBLE, 4.35)
PAPER_SHORT_FIGSIZE = (COL_DOUBLE, 3.85)
PAPER_TALL_FIGSIZE = (COL_DOUBLE, 4.85)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Honour each function's declared canvas size so layouts that need extra
    # vertical room (multi-row maps, atlas pages) don't get crushed back into
    # the default short canvas at save time. A small tight pad guarantees axis
    # labels and legends survive the export.
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


_CARTOPY_CACHE = {"tried": False, "ok": False, "proj": None, "data_crs": None,
                  "feats": None}


def _init_cartopy():
    """Return (use_cartopy, proj, data_crs, features) — memoised across calls.

    The replot script sets CARTOPY_DATA_DIR; any missing data triggers a
    one-time fallback to plain lon/lat. After the first failure we don't try
    again, which keeps log output quiet for the rest of the figure stack.
    """
    if _CARTOPY_CACHE["tried"]:
        return (_CARTOPY_CACHE["ok"], _CARTOPY_CACHE["proj"],
                _CARTOPY_CACHE["data_crs"], _CARTOPY_CACHE["feats"])
    _CARTOPY_CACHE["tried"] = True
    try:
        import os as _os
        cdir = _os.environ.get("CARTOPY_DATA_DIR")
        import cartopy
        if cdir:
            Path(cdir).mkdir(parents=True, exist_ok=True)
            cartopy.config["data_dir"] = Path(cdir)
            cartopy.config["pre_existing_data_dir"] = Path(cdir)
        import cartopy.crs as ccrs
        import cartopy.feature as cfeat
        proj = ccrs.LambertAzimuthalEqualArea(central_latitude=49,
                                              central_longitude=12.5)
        data_crs = ccrs.PlateCarree()
        feats = {
            "land":    cfeat.LAND.with_scale("50m"),
            "ocean":   cfeat.OCEAN.with_scale("50m"),
            "borders": cfeat.BORDERS.with_scale("50m"),
            "coast":   cfeat.COASTLINE.with_scale("50m"),
            "lakes":   cfeat.LAKES.with_scale("50m"),
        }
        for f in feats.values():
            list(f.geometries())
        _CARTOPY_CACHE.update(ok=True, proj=proj, data_crs=data_crs, feats=feats)
        return True, proj, data_crs, feats
    except Exception as exc:
        print(f"[plots] cartopy not usable ({exc.__class__.__name__}); "
              "maps fall back to lon/lat scatter")
        return False, None, None, None


def _add_basemap(ax, feats, *, extent=(7.5, 18.5, 45.0, 51.5),
                 data_crs=None, gridlines=True, label_size=6.5,
                 label_left=True, label_bottom=True):
    """Add land/ocean/borders + optional gridline labels.

    label_left / label_bottom let multi-panel callers suppress redundant
    interior labels (e.g. only label the leftmost column of a 3x2 grid)
    so adjacent panels' left labels don't crowd the previous panel's
    right edge.
    """
    ax.add_feature(feats["land"],    facecolor=LAND_COLOR,  edgecolor="none", zorder=0)
    ax.add_feature(feats["ocean"],   facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(feats["lakes"],   facecolor=OCEAN_COLOR, edgecolor="none", zorder=0)
    ax.add_feature(feats["borders"], edgecolor=BORDER_COLOR, lw=0.45, zorder=2)
    ax.add_feature(feats["coast"],   edgecolor="#444444",   lw=0.45, zorder=2)
    if data_crs is not None:
        ax.set_extent(list(extent), crs=data_crs)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, lw=0.25, color="0.75", alpha=0.55)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = bool(label_left)
        gl.bottom_labels = bool(label_bottom)
        gl.xlabel_style = {"size": label_size, "color": "#333333"}
        gl.ylabel_style = {"size": label_size, "color": "#333333"}
        return gl
    return None


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _prediction_monthly_summary(out_dir: Path) -> pd.DataFrame:
    """Aggregate prediction files into monthly water-balance diagnostics."""
    tables = out_dir / "tables"
    paths = [
        tables / "predictions_tabular.csv",
        tables / "predictions_xlstm.csv",
        tables / "predictions_transformer.csv",
    ]
    usecols = ["model", "split", "MM", "q_mm_day", "q_pred_mm_day"]
    parts: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            reader = pd.read_csv(path, usecols=usecols, chunksize=300_000)
            for chunk in reader:
                chunk = chunk[chunk["split"].isin(["test", "spatial_test"])].copy()
                if chunk.empty:
                    continue
                chunk["q_mm_day"] = pd.to_numeric(chunk["q_mm_day"], errors="coerce")
                chunk["q_pred_mm_day"] = pd.to_numeric(chunk["q_pred_mm_day"], errors="coerce")
                chunk = chunk.dropna(subset=["q_mm_day", "q_pred_mm_day", "MM"])
                if chunk.empty:
                    continue
                chunk["err"] = chunk["q_pred_mm_day"] - chunk["q_mm_day"]
                chunk["abs_err"] = chunk["err"].abs()
                agg = (
                    chunk.groupby(["model", "split", "MM"])
                    .agg(
                        obs_sum=("q_mm_day", "sum"),
                        sim_sum=("q_pred_mm_day", "sum"),
                        err_sum=("err", "sum"),
                        abs_err_sum=("abs_err", "sum"),
                        n=("q_mm_day", "size"),
                    )
                    .reset_index()
                )
                parts.append(agg)
        except ValueError:
            continue
    if not parts:
        return pd.DataFrame()
    out = (
        pd.concat(parts, ignore_index=True)
        .groupby(["model", "split", "MM"], as_index=False)
        [["obs_sum", "sim_sum", "err_sum", "abs_err_sum", "n"]]
        .sum()
    )
    out["pbias"] = np.where(out["obs_sum"].abs() > 1e-12,
                             100.0 * (out["sim_sum"] - out["obs_sum"]) / out["obs_sum"],
                             np.nan)
    out["mae"] = out["abs_err_sum"] / out["n"].replace(0, np.nan)
    out["mean_obs"] = out["obs_sum"] / out["n"].replace(0, np.nan)
    out["mean_sim"] = out["sim_sum"] / out["n"].replace(0, np.nan)
    return out


def _present_models(df: pd.DataFrame) -> list[str]:
    seen = set(df["model"].dropna().astype(str).unique())
    return [m for m in MODEL_ORDER if m in seen]


def _present_splits(df: pd.DataFrame, preferred=("val", "test", "spatial_test")) -> list[str]:
    seen = set(df["split"].dropna().astype(str).unique())
    return [s for s in preferred if s in seen]


def _pick_eval_split(df: pd.DataFrame) -> str | None:
    if df.empty or "split" not in df.columns:
        return None
    seen = set(df["split"].dropna().astype(str).unique())
    for s in ("test", "spatial_test", "val"):
        if s in seen:
            return s
    return None


def _model_legend_handles(models: list[str], lw: float = 2.2) -> list[Line2D]:
    return [Line2D([0], [0], color=model_color(m), lw=lw,
                   linestyle=_model_linestyle(m), label=MODEL_LABELS.get(m, m))
            for m in models]


def _model_linestyle(model: str):
    styles = {
        "cosero": (0, (5.0, 2.2)),
        "random_forest": (0, (3.0, 1.8)),
        "xgboost": "-",
        "xlstm": "-",
        "transformer": "-",
    }
    return styles.get(model, "-")


def _flood_label(event_type: str) -> str:
    return FLOOD_TYPE_LABELS.get(event_type, str(event_type).replace("_", " "))


def _flood_color(event_type: str) -> str:
    return FLOOD_TYPE_COLORS.get(event_type, MUTED_COLOR)


def _sensitivity_label(value: str) -> str:
    labels = {
        "warm_season_rainfall_intensity": "warm-season\nrain intensity",
        "antecedent_wetness_storage": "antecedent\nwetness/storage",
        "warming_snow_transition": "warming snow\ntransition",
        "cool_season_rainfall": "cool-season\nrainfall",
        "mixed_rainfall_runoff": "mixed rainfall-\nrunoff",
    }
    return labels.get(str(value), str(value).replace("_", "\n"))


def _split_color(split: str) -> str:
    return {"test": "#0E2A55", "spatial_test": "#B5302A", "val": "#586471"}.get(split, MUTED_COLOR)


def _grid(ax, axis: str = "y"):
    ax.grid(True, axis=axis, color=GRID_COLOR, lw=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _despine(ax, keep=("left", "bottom")):
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(spine in keep)


def _cdf(values: np.ndarray):
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return None, None
    y = np.arange(1, v.size + 1) / v.size
    return v, y


def _lowess(x, y, frac: float = 0.5):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:
        return None
    if len(x) < 8:
        return None
    z = lowess(y, x, frac=frac, it=1, return_sorted=True)
    return z


def _binned_median_curve(x, y, n_bins: int = 8, min_count: int = 4):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < min_count * 2:
        return None
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == hi:
            continue
        sel = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if int(sel.sum()) < min_count:
            continue
        rows.append((float(np.nanmedian(x[sel])), float(np.nanmedian(y[sel]))))
    return np.asarray(rows) if len(rows) >= 2 else None


def _annot(ax, text: str, x=0.02, y=0.96, **kw):
    # Default zorder=20 keeps the white-background box ABOVE LOWESS curves and
    # scatter so labelled medians/ρ values stay legible against dense data.
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=kw.pop("fontsize", 7.0),
            color=kw.pop("color", ANNOT_COLOR),
            ha=kw.pop("ha", "left"), va=kw.pop("va", "top"),
            bbox=kw.pop("bbox", dict(facecolor="white", edgecolor="0.85",
                                     alpha=0.98, pad=1.5)),
            zorder=kw.pop("zorder", 20),
            **kw)


def _panel_label(ax, letter: str, x: float = -0.16, y: float = 1.04):
    """Add a bold (a) (b) (c) panel label outside the axes — HESS convention.

    Letter is rendered without surrounding parentheses if already provided
    parenthesized, to avoid the historical `((a))` bug.
    """
    txt = letter.strip()
    if txt.startswith("(") and txt.endswith(")"):
        label = txt
    else:
        label = f"({txt})"
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#111111",
            ha="left", va="bottom", clip_on=False)


def _panel_letter(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    if index < len(letters):
        return letters[index]
    first = letters[(index // len(letters)) - 1]
    second = letters[index % len(letters)]
    return first + second


def _format_date_axis(ax, span_years: int = 2):
    if span_years <= 2:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=6.8, pad=1.5)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _peak_years(sub: pd.DataFrame, n_years: int = 1) -> list[int]:
    """Select the most hydrologically active test year, plus a neighbour if asked."""
    if sub.empty or "date" not in sub.columns or "q_obs" not in sub.columns:
        return []
    obs = sub.drop_duplicates(subset=["date"]).dropna(subset=["q_obs"]).copy()
    if obs.empty:
        return []
    obs["year"] = obs["date"].dt.year
    peaks = obs.groupby("year")["q_obs"].max().sort_values(ascending=False)
    if peaks.empty:
        return []
    anchor = int(peaks.index[0])
    available = set(int(y) for y in obs["year"].unique())
    candidates = [anchor, anchor + 1, anchor - 1, anchor + 2, anchor - 2]
    picked = []
    for y in candidates:
        if y in available and y not in picked:
            picked.append(y)
        if len(picked) >= n_years:
            break
    return sorted(picked)


def _add_precip_axis(ax, raw: pd.DataFrame, bid: int, start, end,
                     label: bool = True, compact: bool = False):
    """Add inverted precipitation bars to a hydrograph panel."""
    if raw.empty or "prec" not in raw.columns or "date" not in raw.columns:
        return None
    r = raw[(raw["ID"].eq(bid)) & (raw["date"].between(start, end))]
    if r.empty:
        return None
    ax2 = ax.twinx()
    ax2.patch.set_alpha(0.0)
    ax2.bar(r["date"], r["prec"], width=1.0, color=PRECIP_COLOR,
            alpha=0.34 if compact else 0.38, edgecolor="none", zorder=1)
    ax2.invert_yaxis()
    ymax = float(pd.to_numeric(r["prec"], errors="coerce").max())
    ax2.set_ylim(max(ymax * 4.0, 1.0), 0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(PRECIP_COLOR)
    if compact:
        ax2.set_yticks([])
        ax2.set_ylabel("")
    elif label:
        ax2.set_ylabel("$P$ (mm d$^{-1}$)", color=PRECIP_COLOR, fontsize=7.5)
        ax2.tick_params(axis="y", labelsize=7, colors=PRECIP_COLOR)
    else:
        ax2.tick_params(axis="y", labelsize=7, colors=PRECIP_COLOR, labelright=False)
    return ax2


def _balanced_examples(examples: pd.DataFrame, n_per_zone: int = 2,
                        max_total: int = 12) -> pd.DataFrame:
    """Pick example basins evenly across regime zones, preserving the file order
    within each zone (which encodes representative-ness)."""
    if examples.empty:
        return examples
    zones = ["snow_class", "aridity_class", "storage_class", "impact_class"]
    keep = []
    for z in zones:
        sub = examples[examples["regime_zone"].eq(z)].head(n_per_zone)
        keep.append(sub)
    out = pd.concat(keep, ignore_index=True).head(max_total)
    return out


def _kge_from_arrays(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float); sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]; sim = sim[mask]
    if obs.size < 4 or obs.std() == 0 or sim.std() == 0:
        return float("nan")
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = sim.std() / obs.std()
    beta = sim.mean() / max(obs.mean(), 1e-9)
    return 1 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ─── Figure 1. Cumulative skill across catchments ────────────────────────────

def plot_skill_cdfs(metrics: pd.DataFrame, fig_dir: Path):
    if metrics.empty:
        return
    models = _present_models(metrics)
    splits = [s for s in ("test", "spatial_test")
              if s in set(metrics["split"].dropna().astype(str).unique())]
    if not splits:
        splits = _present_splits(metrics)
    if not models or not splits:
        return
    panels = [("kge",     "KGE",        (-0.2, 1.0), 0.5),
              ("nse",     "NSE",        (-0.2, 1.0), 0.5),
              ("log_nse", "log NSE",    (-0.2, 1.0), 0.5),
              ("pbias",   "PBIAS (%)",  (-40.0, 40.0), 0.0)]
    n_rows = len(splits); n_cols = len(panels)
    # Split rows and metric columns: same information as the old 4x3 figure,
    # but on a single landscape manuscript canvas.
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=PAPER_FIGSIZE,
                             sharey=True, squeeze=False)
    letters = "abcdefghij"
    li = 0
    for i, split in enumerate(splits):
        sub = metrics[metrics["split"].eq(split)]
        for j, (col, label, lim, ref) in enumerate(panels):
            ax = axes[i, j]
            for m in models:
                v = pd.to_numeric(sub.loc[sub["model"].eq(m), col],
                                  errors="coerce").to_numpy()
                xs, ys = _cdf(v)
                if xs is None:
                    continue
                ax.plot(xs, ys, color=model_color(m), lw=1.5,
                        label=model_label(m), zorder=3)
                med = np.nanmedian(v)
                if np.isfinite(med):
                    ax.plot([med, med], [0, 0.5], color=model_color(m),
                            lw=0.7, ls=":", alpha=0.65, zorder=2)
            ax.axhline(0.5, color=RULE_COLOR, lw=0.5, ls=":", zorder=1)
            ax.axvline(ref, color=RULE_COLOR, lw=0.5, ls=":", zorder=1)
            ax.set_xlim(*lim); ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            _grid(ax, axis="both")
            # Combined panel-letter + metric title — single anchor on the
            # left, leaves room for the median table in the empty corner.
            if i == 0:
                ax.set_title(f"({_panel_letter(li)})  {label}",
                             fontsize=9.0, fontweight="semibold",
                             loc="left", pad=4)
            else:
                ax.set_title(f"({_panel_letter(li)})", fontsize=8.6,
                             fontweight="semibold", loc="left", pad=2)
            if j == 0:
                ax.set_ylabel(f"{SPLIT_DISPLAY.get(split, split)}\nnon-exceedance")
            if i == n_rows - 1:
                ax.set_xlabel(label)
            # Median table — anchored in whichever corner the curves leave
            # blank.  PBIAS panels: curves cluster at the centre, so we use
            # the upper-left; quality panels (KGE/NSE/log NSE): curves rise
            # from the lower-left, so we use the lower-right.
            table_lines = []
            for m in models:
                v = pd.to_numeric(sub.loc[sub["model"].eq(m), col],
                                  errors="coerce")
                med = v.median()
                tag = MODEL_LABELS.get(m, m)
                table_lines.append((tag, med))
            # Shorter monospace lines (8-char tag) so the box never overflows
            # the panel at narrow column widths.
            text_lines = [f"{tag[:10]:<10s}{med:6.2f}"
                          for tag, med in table_lines]
            if col == "pbias":
                ann_x, ann_y, ha, va = 0.04, 0.96, "left", "top"
            else:
                ann_x, ann_y, ha, va = 0.97, 0.04, "right", "bottom"
            ax.text(ann_x, ann_y, "\n".join(text_lines),
                    transform=ax.transAxes, ha=ha, va=va, fontsize=5.8,
                    family="monospace", color=ANNOT_COLOR,
                    bbox=dict(facecolor="white", edgecolor="0.85",
                              alpha=0.98, pad=1.8,
                              boxstyle="round,pad=0.20"), zorder=20)
            li += 1
    fig.legend(handles=_model_legend_handles(models), loc="lower center",
               ncol=len(models), bbox_to_anchor=(0.5, -0.005),
               frameon=False, fontsize=7.6)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.12,
                        wspace=0.18, hspace=0.32)
    _save(fig, fig_dir / "fig01_skill_cdf.png")


# ─── Figure 2. Temporal vs spatial transfer ─────────────────────────────────

def plot_temporal_spatial_transfer(metrics: pd.DataFrame, fig_dir: Path):
    """Show the transferability result explicitly instead of burying it in CDFs."""
    if metrics.empty or "split" not in metrics.columns:
        return
    splits = [s for s in ("test", "spatial_test") if s in set(metrics["split"].astype(str))]
    if len(splits) < 2:
        return

    models = []
    for m in _present_models(metrics):
        sub = metrics[metrics["model"].eq(m)]
        if sub["kge"].notna().any():
            models.append(m)
    if not models:
        return

    metrics_to_plot = [
        ("kge", "KGE", (-0.15, 1.0), 0.5),
        ("nse", "NSE", (-0.15, 1.0), 0.5),
        ("log_nse", "log NSE", (-0.15, 1.0), 0.5),
        ("pbias", "PBIAS (%)", (-30.0, 30.0), 0.0),
    ]
    n_by_split = {
        s: int(metrics.loc[metrics["split"].eq(s), "ID"].nunique())
        if "ID" in metrics.columns else int(metrics["split"].eq(s).sum())
        for s in splits
    }
    split_labels = {
        "test": f"temporal test\n(n={n_by_split.get('test', 0)})",
        "spatial_test": f"spatial holdout\n(n={n_by_split.get('spatial_test', 0)})",
    }

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)
    x = np.arange(len(models), dtype=float)
    width = 0.34
    offsets = {"test": -width / 2, "spatial_test": width / 2}
    alphas = {"test": 1.0, "spatial_test": 0.45}

    for idx, (col, label, ylim, ref) in enumerate(metrics_to_plot):
        ax = axes.ravel()[idx]
        if col == "pbias":
            ax.axhspan(-10, 10, color="#DDE8F0", alpha=0.55, zorder=0)
        for split in splits:
            vals = []
            for m in models:
                v = pd.to_numeric(
                    metrics.loc[
                        metrics["split"].eq(split) & metrics["model"].eq(m),
                        col,
                    ],
                    errors="coerce",
                )
                vals.append(float(v.median()) if v.notna().any() else np.nan)
            ax.bar(
                x + offsets[split],
                vals,
                width=width,
                color=[model_color(m) for m in models],
                edgecolor=AXIS_COLOR,
                linewidth=0.45,
                alpha=alphas[split],
                zorder=3,
            )
        ax.axhline(ref, color=RULE_COLOR, lw=0.75, ls="--", zorder=2)
        if col == "pbias":
            ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=2)
            ax.set_ylabel("median PBIAS (%)")
        else:
            ax.set_ylabel(f"median {label}")
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([model_label(m) for m in models], rotation=22, ha="right")
        ax.set_title(label, loc="left", fontweight="semibold")
        _grid(ax, axis="y")
        _despine(ax)
        _panel_label(ax, _panel_letter(idx), x=-0.13, y=1.03)

    split_handles = [
        Patch(facecolor="#5B6770", edgecolor=AXIS_COLOR, alpha=1.0,
              label=split_labels["test"]),
        Patch(facecolor="#5B6770", edgecolor=AXIS_COLOR, alpha=0.45,
              label=split_labels["spatial_test"]),
    ]
    model_handles = _model_legend_handles(models, lw=2.4)
    fig.legend(handles=split_handles + model_handles, loc="lower center",
               ncol=min(len(split_handles) + len(model_handles), 6),
               bbox_to_anchor=(0.5, -0.015), frameon=False,
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 1.0))
    _save(fig, fig_dir / "fig02_transfer_skill.png")


def plot_transfer_kge_maps(metrics: pd.DataFrame, attrs_geo: pd.DataFrame, fig_dir: Path):
    """Geographic KGE maps for temporal test and spatial holdout."""
    if metrics.empty or attrs_geo.empty:
        return
    required = {"ID", "lon_deg", "lat_deg"}
    if not required.issubset(attrs_geo.columns):
        return
    splits = [s for s in ("test", "spatial_test") if s in set(metrics["split"].astype(str))]
    if len(splits) < 2:
        return
    models = [m for m in _present_models(metrics) if m != "cosero"]
    if not models:
        return

    geo = attrs_geo[["ID", "lon_deg", "lat_deg"]].copy()
    df = metrics.merge(geo, on="ID", how="left").dropna(subset=["lon_deg", "lat_deg", "kge"])
    if df.empty:
        return

    lon_min, lon_max = df["lon_deg"].min(), df["lon_deg"].max()
    lat_min, lat_max = df["lat_deg"].min(), df["lat_deg"].max()
    lon_pad = max((lon_max - lon_min) * 0.06, 0.2)
    lat_pad = max((lat_max - lat_min) * 0.06, 0.2)
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.5, vmax=1.0)
    use_cartopy, proj, data_crs, feats = _init_cartopy()
    extent = (max(lon_min - lon_pad, 5.5), min(lon_max + lon_pad, 22.0),
              max(lat_min - lat_pad, 44.5), min(lat_max + lat_pad, 52.5))

    width_ratios = [1.0] * len(models) + [0.045]
    # Height matched to the map aspect (wide Central-European domain in four
    # narrow columns) so the rows sit tight without dead whitespace.
    fig = plt.figure(figsize=(COL_DOUBLE, 2.9 if len(splits) == 2 else 1.75 * len(splits)))
    gs = fig.add_gridspec(
        len(splits), len(models) + 1,
        width_ratios=width_ratios,
        wspace=0.06,
        hspace=0.04,
    )
    axes_grid = []
    for i in range(len(splits)):
        row = []
        for j in range(len(models)):
            if use_cartopy:
                ax = fig.add_subplot(gs[i, j], projection=proj)
                _add_basemap(ax, feats, extent=extent, data_crs=data_crs,
                             gridlines=False)
            else:
                ax = fig.add_subplot(gs[i, j])
                ax.set_facecolor(LAND_COLOR)
            row.append(ax)
        axes_grid.append(row)
    axes = np.array(axes_grid, dtype=object)
    cax = fig.add_subplot(gs[:, -1])
    sc = None
    li = 0
    for i, split in enumerate(splits):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = df[df["split"].eq(split) & df["model"].eq(model)]
            scatter_kw = dict(cmap=SKILL_CMAP, norm=norm,
                              s=16 if split == "test" else 28,
                              edgecolor="white", linewidth=0.35,
                              alpha=0.96, zorder=3)
            if use_cartopy:
                scatter_kw["transform"] = data_crs
            if not sub.empty:
                sc = ax.scatter(sub["lon_deg"], sub["lat_deg"], c=sub["kge"],
                                **scatter_kw)
                med = pd.to_numeric(sub["kge"], errors="coerce").median()
                ax.text(0.03, 0.97,
                        f"median KGE={med:.2f}\n$n$={sub['ID'].nunique()}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.4, color=ANNOT_COLOR,
                        bbox=dict(facecolor="white", edgecolor="0.85",
                                  alpha=0.98, pad=1.6,
                                  boxstyle="round,pad=0.25"), zorder=20)
            if not use_cartopy:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xticks([])
                ax.set_yticks([])
                _despine(ax, keep=())
            if i == 0:
                ax.set_title(model_label(model), fontsize=8.8,
                             fontweight="semibold", pad=6)
            if j == 0:
                ax.text(-0.05, 0.5, SPLIT_DISPLAY.get(split, split),
                        transform=ax.transAxes, rotation=90,
                        fontsize=8.4, fontweight="semibold",
                        ha="center", va="center")
            ax.text(0.97, 0.04, f"({_panel_letter(li)})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.8, fontweight="bold", color=AXIS_COLOR,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.82, pad=1.1), zorder=20)
            li += 1
    if sc is not None:
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("KGE", fontsize=7.6, labelpad=4)
        cbar.ax.tick_params(labelsize=7.0)
    else:
        cax.set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.90, bottom=0.03,
                        wspace=0.06, hspace=0.04)
    _save(fig, fig_dir / "fig25_transfer_kge_maps.png")


def plot_memory_gain_maps(delta: pd.DataFrame, attrs_geo: pd.DataFrame, fig_dir: Path):
    """Map where memory/sequence models improve over tabular alternatives."""
    if delta.empty or attrs_geo.empty:
        return
    if "ID" not in delta.columns:
        return
    if "lon_deg" not in delta.columns or "lat_deg" not in delta.columns:
        geo = attrs_geo[["ID", "lon_deg", "lat_deg"]].copy()
        delta = delta.merge(geo, on="ID", how="left")
    delta = delta.dropna(subset=["lon_deg", "lat_deg"])
    if delta.empty:
        return
    splits = [s for s in ("test", "spatial_test") if s in set(delta["split"].astype(str))]
    cols = [
        ("memory_gain_xlstm_vs_rf", "xLSTM − RF"),
        ("sequence_gain_vs_tree", "best sequence − best tree"),
    ]
    cols = [(c, t) for c, t in cols if c in delta.columns]
    if not splits or not cols:
        return

    all_vals = pd.to_numeric(delta[[c for c, _ in cols]].stack(), errors="coerce")
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.empty:
        return
    lim = float(np.nanquantile(np.abs(all_vals), 0.95))
    lim = max(min(lim, 0.5), 0.12)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    lon_min, lon_max = delta["lon_deg"].min(), delta["lon_deg"].max()
    lat_min, lat_max = delta["lat_deg"].min(), delta["lat_deg"].max()
    lon_pad = max((lon_max - lon_min) * 0.06, 0.2)
    lat_pad = max((lat_max - lat_min) * 0.06, 0.2)
    use_cartopy, proj, data_crs, feats = _init_cartopy()
    extent = (max(lon_min - lon_pad, 5.5), min(lon_max + lon_pad, 22.0),
              max(lat_min - lat_pad, 44.5), min(lat_max + lat_pad, 52.5))

    width_ratios = [1.0] * len(cols) + [0.045]
    fig = plt.figure(figsize=(COL_DOUBLE, 4.65))
    gs = fig.add_gridspec(
        len(splits), len(cols) + 1,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.18,
    )
    axes_grid = []
    for i in range(len(splits)):
        row = []
        for j in range(len(cols)):
            if use_cartopy:
                ax = fig.add_subplot(gs[i, j], projection=proj)
                _add_basemap(ax, feats, extent=extent, data_crs=data_crs,
                             gridlines=False)
            else:
                ax = fig.add_subplot(gs[i, j])
                ax.set_facecolor(LAND_COLOR)
            row.append(ax)
        axes_grid.append(row)
    axes = np.array(axes_grid, dtype=object)
    cax = fig.add_subplot(gs[:, -1])
    sc = None
    li = 0
    for i, split in enumerate(splits):
        for j, (col, title) in enumerate(cols):
            ax = axes[i, j]
            sub = delta[delta["split"].eq(split)].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[col])
            scatter_kw = dict(cmap="RdBu", norm=norm,
                              s=18 if split == "test" else 30,
                              edgecolor="white", linewidth=0.35,
                              alpha=0.96, zorder=3)
            if use_cartopy:
                scatter_kw["transform"] = data_crs
            if not sub.empty:
                sc = ax.scatter(sub["lon_deg"], sub["lat_deg"], c=sub[col],
                                **scatter_kw)
                med = sub[col].median()
                ax.text(0.03, 0.97,
                        f"median = {med:+.3f}\n$n$ = {sub['ID'].nunique()}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.4, color=ANNOT_COLOR,
                        bbox=dict(facecolor="white", edgecolor="0.85",
                                  alpha=0.9, pad=1.6,
                                  boxstyle="round,pad=0.25"), zorder=20)
            if not use_cartopy:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xticks([])
                ax.set_yticks([])
                _despine(ax, keep=())
            if i == 0:
                ax.set_title(title, fontsize=8.8, fontweight="semibold", pad=6)
            if j == 0:
                ax.text(-0.05, 0.5, SPLIT_DISPLAY.get(split, split),
                        transform=ax.transAxes, rotation=90,
                        fontsize=8.4, fontweight="semibold",
                        ha="center", va="center")
            ax.text(0.97, 0.04, f"({_panel_letter(li)})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.8, fontweight="bold", color=AXIS_COLOR,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.82, pad=1.1), zorder=20)
            li += 1
    if sc is not None:
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("KGE gain", fontsize=7.6, labelpad=4)
        cbar.ax.tick_params(labelsize=7.0)
    else:
        cax.set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.93, bottom=0.04,
                        wspace=0.08, hspace=0.18)
    _save(fig, fig_dir / "fig26_memory_gain_maps.png")


def plot_transfer_regime_delta(zone_summary: pd.DataFrame, fig_dir: Path):
    """Show which regimes are robust or fragile under spatial transfer."""
    if zone_summary.empty:
        return
    needed = {"split", "model", "zone", "zone_value", "median_kge"}
    if not needed.issubset(zone_summary.columns):
        return
    zones = ["snow_class", "aridity_class", "storage_class", "impact_class"]
    models = _present_models(zone_summary)
    models = [m for m in models if m != "cosero"]
    if not models:
        return

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)
    handles = _model_legend_handles(models, lw=2.4)
    for idx, zone in enumerate(zones):
        ax = axes.ravel()[idx]
        sub = zone_summary[zone_summary["zone"].eq(zone)].copy()
        values = [v for v in ZONE_VALUE_ORDER.get(zone, list(sub["zone_value"].unique()))
                  if v in set(sub["zone_value"].astype(str))]
        if not values:
            continue
        x = np.arange(len(values), dtype=float)
        width = min(0.74 / max(len(models), 1), 0.18)
        for mi, model in enumerate(models):
            deltas = []
            for value in values:
                temporal = pd.to_numeric(
                    sub.loc[
                        sub["split"].eq("test")
                        & sub["model"].eq(model)
                        & sub["zone_value"].eq(value),
                        "median_kge",
                    ],
                    errors="coerce",
                )
                spatial = pd.to_numeric(
                    sub.loc[
                        sub["split"].eq("spatial_test")
                        & sub["model"].eq(model)
                        & sub["zone_value"].eq(value),
                        "median_kge",
                    ],
                    errors="coerce",
                )
                if temporal.empty or spatial.empty:
                    deltas.append(np.nan)
                else:
                    deltas.append(float(spatial.iloc[0] - temporal.iloc[0]))
            xpos = x + (mi - (len(models) - 1) / 2) * width
            ax.bar(xpos, deltas, width=width, color=model_color(model),
                   edgecolor=AXIS_COLOR, linewidth=0.35, alpha=0.9, zorder=3)
        ax.axhline(0, color=AXIS_COLOR, lw=0.75, zorder=2)
        ax.axhspan(-0.05, 0.05, color="#DDE8F0", alpha=0.55, zorder=0)
        ax.set_title(ZONE_LABELS.get(zone, zone), loc="left",
                     fontweight="semibold")
        ax.set_ylabel("spatial − temporal KGE")
        ax.set_xticks(x)
        ax.set_xticklabels([zone_value_pretty(v) for v in values],
                           rotation=22, ha="right")
        ax.set_ylim(-0.28, 0.28)
        _grid(ax, axis="y")
        _despine(ax)
        _panel_label(ax, _panel_letter(idx), x=-0.13, y=1.03)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.012), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, fig_dir / "fig27_transfer_regime_delta.png")


# ─── Figure 28. Seasonal residual structure ─────────────────────────────────

def plot_seasonal_residual_cycle(out_dir: Path, fig_dir: Path):
    monthly = _prediction_monthly_summary(out_dir)
    if monthly.empty:
        return
    models = _present_models(monthly)
    splits = [s for s in ("test", "spatial_test")
              if s in set(monthly["split"].astype(str))]
    if not models or not splits:
        return

    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    vals = pd.to_numeric(monthly["pbias"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    lim = float(np.nanmax(np.abs(vals.dropna()))) * 1.08 if vals.notna().any() else 40.0
    lim = max(25.0, min(lim, 80.0))

    fig, axes = plt.subplots(2, len(splits), figsize=PAPER_FIGSIZE,
                             squeeze=False, sharex=True)
    li = 0
    for j, split in enumerate(splits):
        ax = axes[0, j]
        sub = monthly[monthly["split"].eq(split)].copy()
        ax.axhspan(-10, 10, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
        ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=1)
        for m in models:
            d = sub[sub["model"].eq(m)].set_index("MM").reindex(months)
            ax.plot(months, d["pbias"], color=model_color(m), lw=1.8,
                    linestyle=_model_linestyle(m), label=model_label(m), zorder=3)
        ax.set_title(SPLIT_DISPLAY.get(split, split), fontweight="semibold")
        ax.set_ylabel("Monthly PBIAS (%)")
        ax.set_ylim(-lim, lim)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        _grid(ax, axis="both")
        _panel_label(ax, _panel_letter(li), x=-0.11, y=1.02)
        li += 1

        axh = axes[1, j]
        heat = np.full((len(models), len(months)), np.nan)
        for r, m in enumerate(models):
            d = sub[sub["model"].eq(m)].set_index("MM").reindex(months)
            heat[r, :] = d["pbias"].to_numpy(dtype=float)
        im = axh.imshow(heat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for r in range(heat.shape[0]):
            for c in range(heat.shape[1]):
                v = heat[r, c]
                if not np.isfinite(v):
                    continue
                txt = "white" if abs(v) > lim * 0.55 else "#111111"
                axh.text(c, r, f"{v:+.0f}", ha="center", va="center",
                         fontsize=6.1, color=txt)
        axh.set_yticks(range(len(models)))
        axh.set_yticklabels([model_label(m) for m in models], fontsize=7.0)
        axh.set_xticks(range(len(months)))
        axh.set_xticklabels(month_labels)
        axh.tick_params(axis="both", length=0)
        axh.set_xlabel("Month")
        _panel_label(axh, _panel_letter(li), x=-0.11, y=1.02)
        li += 1

    handles = _model_legend_handles(models, lw=2.0)
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.065, 0.91, 1.0))
    cax = fig.add_axes([0.93, 0.18, 0.012, 0.31])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("PBIAS (%)", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.0)
    _save(fig, fig_dir / "fig28_seasonal_residual_cycle.png")


# ─── Figure 29. Budyko-style runoff partitioning ────────────────────────────

def plot_budyko_aridity_partitioning(sig_long: pd.DataFrame, attrs_geo: pd.DataFrame,
                                     fig_dir: Path):
    if sig_long.empty or attrs_geo.empty:
        return
    needed = {"ID", "p_mean", "aridity", "frac_snow", "aridity_class"}
    if not needed.issubset(attrs_geo.columns):
        return
    split = _pick_eval_split(sig_long)
    if split is None:
        return
    sub = sig_long[(sig_long["split"].eq(split)) &
                   (sig_long["signature"].eq("q_mean"))].copy()
    if sub.empty:
        return
    cols = ["ID", "p_mean", "aridity", "frac_snow", "aridity_class"]
    sub = sub.merge(attrs_geo[cols], on="ID", how="left")
    for c in ["obs", "sim", "p_mean", "aridity", "frac_snow"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["obs", "sim", "p_mean", "aridity"])
    sub = sub[sub["p_mean"] > 0]
    if sub.empty:
        return
    sub["rc_obs"] = sub["obs"] / sub["p_mean"]
    sub["rc_sim"] = sub["sim"] / sub["p_mean"]
    sub["rc_bias_pct"] = np.where(sub["rc_obs"].abs() > 1e-12,
                                   100.0 * (sub["rc_sim"] - sub["rc_obs"]) / sub["rc_obs"],
                                   np.nan)
    models = _present_models(sub)
    if not models:
        return

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)
    obs = sub.drop_duplicates("ID").copy()

    ax = axes[0, 0]
    sc = ax.scatter(obs["aridity"], obs["rc_obs"], c=obs["frac_snow"],
                    cmap="Blues", s=18, edgecolor="white", linewidth=0.25,
                    alpha=0.78, zorder=3)
    z = _lowess(obs["aridity"].to_numpy(), obs["rc_obs"].to_numpy(), frac=0.45)
    if z is not None:
        ax.plot(z[:, 0], z[:, 1], color=OBS_COLOR, lw=2.0, zorder=4)
    ax.set_title("Observed runoff partitioning", fontweight="semibold")
    ax.set_xlabel("Aridity index (PET / P)")
    ax.set_ylabel("Observed runoff coefficient $Q/P$")
    _grid(ax, axis="both")
    _panel_label(ax, "a", x=-0.12, y=1.02)

    ax = axes[0, 1]
    if z is not None:
        ax.plot(z[:, 0], z[:, 1], color=OBS_COLOR, lw=2.2, label="Observed", zorder=5)
    for m in models:
        d = sub[sub["model"].eq(m)]
        zz = _lowess(d["aridity"].to_numpy(), d["rc_sim"].to_numpy(), frac=0.45)
        if zz is not None:
            ax.plot(zz[:, 0], zz[:, 1], color=model_color(m), lw=1.7,
                    linestyle=_model_linestyle(m), label=model_label(m), zorder=4)
    ax.set_title("Modelled partitioning curves", fontweight="semibold")
    ax.set_xlabel("Aridity index (PET / P)")
    ax.set_ylabel("Runoff coefficient $Q/P$")
    ax.legend(loc="upper right", fontsize=6.8,
              frameon=True, facecolor="white", edgecolor="0.85",
              framealpha=0.98).set_zorder(20)
    _grid(ax, axis="both")
    _panel_label(ax, "b", x=-0.12, y=1.02)

    ax = axes[1, 0]
    ax.axhspan(-20, 20, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
    ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=1)
    rho_lines = []
    for m in models:
        d = sub[sub["model"].eq(m)].dropna(subset=["aridity", "rc_bias_pct"])
        ax.scatter(d["aridity"], d["rc_bias_pct"].clip(-120, 120), s=8,
                   color=model_color(m), alpha=0.16, edgecolor="none", zorder=2)
        zz = _lowess(d["aridity"].to_numpy(), d["rc_bias_pct"].clip(-120, 120).to_numpy(), frac=0.45)
        if zz is not None:
            ax.plot(zz[:, 0], zz[:, 1], color=model_color(m), lw=1.8,
                    linestyle=_model_linestyle(m), zorder=4)
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(d["aridity"], d["rc_bias_pct"])
        except Exception:
            rho = np.nan
        rho_lines.append((model_label(m), rho))
    ax.set_title("Aridity-dependent water-balance bias", fontweight="semibold")
    ax.set_xlabel("Aridity index (PET / P)")
    ax.set_ylabel("Runoff-coefficient bias (%)")
    ax.set_ylim(-105, 105)
    if rho_lines:
        _annot(ax, "\n".join(f"{lab}: $\\rho$={rho:+.2f}" for lab, rho in rho_lines),
               x=0.03, y=0.96, fontsize=6.2)
    _grid(ax, axis="both")
    _panel_label(ax, "c", x=-0.12, y=1.02)

    ax = axes[1, 1]
    order = [v for v in ZONE_VALUE_ORDER["aridity_class"]
             if v in set(sub["aridity_class"].dropna().astype(str))]
    x = np.arange(len(order))
    width = min(0.74 / max(len(models), 1), 0.18)
    ax.axhspan(-20, 20, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
    ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=1)
    for i, m in enumerate(models):
        vals = []
        for cls in order:
            v = pd.to_numeric(
                sub.loc[(sub["model"].eq(m)) & (sub["aridity_class"].astype(str).eq(cls)),
                        "rc_bias_pct"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            vals.append(float(v.median()) if len(v) else np.nan)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals,
               width=width, color=model_color(m), edgecolor="white",
               linewidth=0.45, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([zone_value_pretty(v) for v in order])
    ax.set_ylabel("Median $Q/P$ bias (%)")
    ax.set_title("Bias by aridity regime", fontweight="semibold")
    _grid(ax, axis="y")
    _panel_label(ax, "d", x=-0.12, y=1.02)

    cbar = fig.colorbar(sc, ax=axes[0, 0], fraction=0.045, pad=0.02)
    cbar.set_label("Snow fraction", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7.0)
    fig.tight_layout()
    _save(fig, fig_dir / "fig29_budyko_aridity_partitioning.png")


# ─── Figure 30. Memory-gain gradients ───────────────────────────────────────

def plot_memory_gradient_diagnostics(delta: pd.DataFrame, fig_dir: Path):
    if delta.empty or "sequence_gain_vs_tree" not in delta.columns:
        return
    panels = [
        ("frac_snow", "Snow fraction"),
        ("bfi", "Baseflow index"),
        ("fdc_slope", "FDC slope"),
        ("aridity", "Aridity index"),
    ]
    panels = [(c, lab) for c, lab in panels if c in delta.columns]
    if not panels:
        return
    d0 = delta[delta["split"].isin(["test", "spatial_test"])].copy()
    d0["sequence_gain_vs_tree"] = pd.to_numeric(d0["sequence_gain_vs_tree"], errors="coerce")
    if d0.empty:
        return
    splits = [s for s in ("test", "spatial_test") if s in set(d0["split"].astype(str))]

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)
    for k, (col, label) in enumerate(panels):
        ax = axes[k // 2, k % 2]
        ax.axhspan(-0.05, 0.05, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
        ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=1)
        rho_lines = []
        for split in splits:
            sub = d0[d0["split"].eq(split)].dropna(subset=[col, "sequence_gain_vs_tree"])
            if sub.empty:
                continue
            x = pd.to_numeric(sub[col], errors="coerce")
            y = pd.to_numeric(sub["sequence_gain_vs_tree"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]
            if len(x) > 25:
                lo, hi = np.nanquantile(x, [0.02, 0.98])
                show = (x >= lo) & (x <= hi)
            else:
                show = np.ones(len(x), dtype=bool)
            ax.scatter(x[show], y[show], s=13 if split == "test" else 22,
                       color=_split_color(split), alpha=0.25,
                       edgecolor="white", linewidth=0.2, zorder=2)
            zz = _binned_median_curve(x[show], y[show], n_bins=8, min_count=5)
            if zz is not None:
                ax.plot(zz[:, 0], zz[:, 1], color=_split_color(split), lw=1.9,
                        label=SPLIT_DISPLAY.get(split, split), zorder=4)
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(x, y)
            except Exception:
                rho = np.nan
            rho_lines.append((SPLIT_DISPLAY.get(split, split), rho))
        all_x = pd.to_numeric(d0[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(all_x) > 25:
            lo, hi = np.nanquantile(all_x, [0.01, 0.99])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = (hi - lo) * 0.04
                ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(label)
        ax.set_ylabel("Sequence gain over tree KGE")
        ax.set_ylim(-0.35, 0.35)
        ax.set_title(label, fontweight="semibold")
        if rho_lines:
            _annot(ax, "\n".join(f"{lab}: $\\rho$={rho:+.2f}" for lab, rho in rho_lines),
                   x=0.03, y=0.96, fontsize=6.3)
        _grid(ax, axis="both")
        _panel_label(ax, _panel_letter(k), x=-0.12, y=1.02)
    handles = [Line2D([0], [0], color=_split_color(s), lw=2.0,
                      label=SPLIT_DISPLAY.get(s, s)) for s in splits]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    _save(fig, fig_dir / "fig30_memory_gradient_diagnostics.png")


# ─── Figure 31. Spatial novelty and transfer failure ────────────────────────

def plot_spatial_novelty_transfer(delta: pd.DataFrame, fig_dir: Path):
    if delta.empty or "split" not in delta.columns:
        return
    features = ["frac_snow", "aridity", "bfi", "fdc_slope", "area_calc", "soil_condu"]
    features = [c for c in features if c in delta.columns]
    if len(features) < 3:
        return
    d = delta[delta["split"].isin(["test", "spatial_test"])].drop_duplicates(["split", "ID"]).copy()
    if d.empty:
        return
    work = d[["split", "ID", "storage_class", "sequence_gain_vs_tree"] + features].copy()
    for c in features + ["sequence_gain_vs_tree"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if "area_calc" in work.columns:
        work["area_calc"] = np.log10(work["area_calc"].clip(lower=0.1))
    if "soil_condu" in work.columns:
        work["soil_condu"] = np.log10(work["soil_condu"].clip(lower=1e-5))
    ref = work[work["split"].eq("test")].dropna(subset=features)
    spatial = work[work["split"].eq("spatial_test")].dropna(subset=features).copy()
    if ref.empty or spatial.empty:
        return
    mu = ref[features].median()
    sd = ref[features].std().replace(0, np.nan).fillna(1.0)
    ref_x = ((ref[features] - mu) / sd).to_numpy(dtype=float)
    sp_x = ((spatial[features] - mu) / sd).to_numpy(dtype=float)
    distances = []
    for row in sp_x:
        dist = np.sqrt(np.nanmean((ref_x - row) ** 2, axis=1))
        distances.append(float(np.nanmin(dist)))
    spatial["hydrologic_novelty"] = distances

    kge_cols = [
        ("kge_rf", "random_forest"),
        ("kge_xgb", "xgboost"),
        ("kge_xlstm", "xlstm"),
        ("kge_transformer", "transformer"),
    ]
    kge_cols = [(c, m) for c, m in kge_cols if c in delta.columns]
    spatial_kge = spatial[["split", "ID", "storage_class", "hydrologic_novelty", "sequence_gain_vs_tree"]].merge(
        delta[delta["split"].eq("spatial_test")][["ID"] + [c for c, _ in kge_cols]],
        on="ID", how="left"
    )

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)

    ax = axes[0, 0]
    ax.hist(spatial["hydrologic_novelty"], bins=18, color=SLATE_COLOR,
            edgecolor="white", linewidth=0.45, zorder=3)
    ax.axvline(spatial["hydrologic_novelty"].median(), color=OBS_COLOR,
               lw=1.1, ls="--", zorder=4)
    ax.set_xlabel("Hydrological novelty\n(nearest temporal-test basin)")
    ax.set_ylabel("Spatial holdout basins")
    ax.set_title("Novelty distribution", fontweight="semibold")
    _grid(ax, axis="y")
    _panel_label(ax, "a", x=-0.12, y=1.02)

    ax = axes[0, 1]
    for col, model in kge_cols:
        spatial_kge[col] = pd.to_numeric(spatial_kge[col], errors="coerce")
        ax.scatter(spatial_kge["hydrologic_novelty"], spatial_kge[col],
                   s=18, color=model_color(model), alpha=0.28,
                   edgecolor="white", linewidth=0.2, zorder=2)
        zz = _binned_median_curve(spatial_kge["hydrologic_novelty"],
                                  spatial_kge[col], n_bins=7, min_count=5)
        if zz is not None:
            ax.plot(zz[:, 0], zz[:, 1], color=model_color(model), lw=1.8,
                    linestyle=_model_linestyle(model), label=model_label(model), zorder=4)
    ax.axhline(0.5, color=RULE_COLOR, lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("Hydrological novelty")
    ax.set_ylabel("Spatial-test KGE")
    ax.set_title("Transfer skill vs novelty", fontweight="semibold")
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc="lower left", fontsize=6.5,
              frameon=True, facecolor="white", edgecolor="0.85",
              framealpha=0.98).set_zorder(20)
    _grid(ax, axis="both")
    _panel_label(ax, "b", x=-0.12, y=1.02)

    ax = axes[1, 0]
    ax.axhspan(-0.05, 0.05, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
    ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=1)
    y = pd.to_numeric(spatial["sequence_gain_vs_tree"], errors="coerce")
    x = spatial["hydrologic_novelty"]
    ax.scatter(x, y, s=22, color="#1F77BD", alpha=0.40,
               edgecolor="white", linewidth=0.25, zorder=3)
    zz = _binned_median_curve(x, y, n_bins=7, min_count=5)
    if zz is not None:
        ax.plot(zz[:, 0], zz[:, 1], color="#1F77BD", lw=2.0, zorder=4)
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y, nan_policy="omit")
    except Exception:
        rho = np.nan
    _annot(ax, f"$\\rho$={rho:+.2f}", x=0.03, y=0.95, fontsize=7.2)
    ax.set_xlabel("Hydrological novelty")
    ax.set_ylabel("Sequence gain over tree KGE")
    ax.set_title("Does novelty require memory?", fontweight="semibold")
    ax.set_ylim(-0.35, 0.35)
    _grid(ax, axis="both")
    _panel_label(ax, "c", x=-0.12, y=1.02)

    ax = axes[1, 1]
    classes = [c for c in ZONE_VALUE_ORDER["storage_class"]
               if c in set(spatial["storage_class"].dropna().astype(str))]
    if not classes:
        classes = list(spatial["storage_class"].dropna().astype(str).unique())
    data = [
        spatial.loc[spatial["storage_class"].astype(str).eq(cls), "hydrologic_novelty"].dropna().to_numpy()
        for cls in classes
    ]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, whis=(10, 90),
                    medianprops=dict(color="white", lw=1.0))
    colors = [REGIME_PALETTES.get("size_class", {}).get("medium", "#1F77BD"),
              "#0E2A55", "#586471", MUTED_COLOR]
    for patch, color in zip(bp["boxes"], colors * 3):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels([f"{zone_value_pretty(c)}\n$n$={len(v)}"
                        for c, v in zip(classes, data)], fontsize=7.2)
    ax.set_ylabel("Hydrological novelty")
    ax.set_title("Novelty by storage regime", fontweight="semibold")
    _grid(ax, axis="y")
    _panel_label(ax, "d", x=-0.12, y=1.02)

    fig.tight_layout()
    _save(fig, fig_dir / "fig31_spatial_novelty_transfer.png")


# ─── Figure 32. Climate-sensitive flood errors ──────────────────────────────

def plot_climate_sensitivity_flood_errors(flood_skill: pd.DataFrame,
                                          flood_events: pd.DataFrame,
                                          fig_dir: Path):
    if flood_skill.empty or flood_events.empty:
        return
    needed = {"split", "climate_sensitivity", "model", "event_id"}
    if not needed.issubset(flood_skill.columns):
        return
    split = _pick_eval_split(flood_skill)
    if split is None:
        return
    fs = flood_skill[flood_skill["split"].eq(split)].copy()
    fe = flood_events[flood_events["split"].eq(split)].copy()
    if fs.empty or fe.empty:
        return
    order = (
        fe.groupby("climate_sensitivity")["event_id"].nunique()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    order = [o for o in order if o in set(fs["climate_sensitivity"].astype(str))]
    if not order:
        return
    models = _present_models(fs)
    if not models:
        return
    summary = (
        fs.groupby(["climate_sensitivity", "model"])
        .agg(
            events=("event_id", "nunique"),
            peak_ratio=("peak_ratio", "median"),
            volume_ratio=("volume_ratio", "median"),
            timing_error=("abs_peak_timing_error_days", "median"),
            hit_rate=("threshold_hit", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce")) * 100)),
        )
        .reset_index()
    )
    counts = fe.groupby("climate_sensitivity")["event_id"].nunique().reindex(order)

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)
    ax = axes[0, 0]
    y = np.arange(len(order))
    ax.barh(y, counts.values, color=SLATE_COLOR, edgecolor="white",
            linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_sensitivity_label(o) for o in order], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Observed flood events")
    ax.set_title("Climate-sensitive flood sample", fontweight="semibold")
    _grid(ax, axis="x")
    _panel_label(ax, "a", x=-0.17, y=1.02)

    plot_specs = [
        ("peak_ratio", "Median peak ratio", "$Q_{sim,peak}/Q_{obs,peak}$", (0, 2.05), 1.0, "b"),
        ("volume_ratio", "Median event-volume ratio", "event volume ratio", (0, 2.05), 1.0, "c"),
        ("hit_rate", "High-flow threshold hit rate", "hit rate (%)", (0, 100), 50.0, "d"),
    ]
    y = np.arange(len(order), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(models)) if len(models) > 1 else np.array([0.0])
    ylabels = [_sensitivity_label(o) for o in order]
    for ax, (col, title, xlabel, xlim, ref, letter) in zip(axes.ravel()[1:], plot_specs):
        if col in {"peak_ratio", "volume_ratio"}:
            ax.axvspan(0.8, 1.2, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
        for i, (m, off) in enumerate(zip(models, offsets)):
            d = summary[summary["model"].eq(m)].set_index("climate_sensitivity")
            vals = [float(d.loc[o, col]) if o in d.index else np.nan for o in order]
            ypos = y + off
            ax.scatter(vals, ypos, s=24, color=model_color(m),
                       edgecolor="white", linewidth=0.35, zorder=4,
                       label=model_label(m))
        ax.axvline(ref, color=AXIS_COLOR if col != "hit_rate" else RULE_COLOR,
                   lw=0.7, ls="-" if col != "hit_rate" else ":", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_title(title, fontweight="semibold")
        _grid(ax, axis="x")
        _panel_label(ax, letter, x=-0.12, y=1.02)
    handles = _model_legend_handles(models, lw=1.9)
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, fig_dir / "fig32_climate_sensitivity_flood_errors.png")


# ─── Figure 33. COSERO benchmark dashboard ─────────────────────────────────

def plot_cosero_benchmark_dashboard(metrics: pd.DataFrame,
                                    sig_summary: pd.DataFrame,
                                    fig_dir: Path):
    if metrics.empty or "cosero" not in set(metrics.get("model", pd.Series(dtype=str)).astype(str)):
        return
    splits = [s for s in ("test", "spatial_test")
              if s in set(metrics["split"].dropna().astype(str))]
    models = _present_models(metrics)
    models = [m for m in models if m in set(metrics["model"].astype(str))]
    if not splits or len(models) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)

    # (a) Median KGE by split and model.
    ax = axes[0, 0]
    x = np.arange(len(splits), dtype=float)
    width = min(0.78 / max(len(models), 1), 0.14)
    for i, m in enumerate(models):
        vals = []
        for split in splits:
            vals.append(float(pd.to_numeric(
                metrics.loc[(metrics["split"].eq(split)) & (metrics["model"].eq(m)), "kge"],
                errors="coerce"
            ).median()))
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width,
               color=model_color(m), edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.5, color=RULE_COLOR, lw=0.65, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_DISPLAY.get(s, s) for s in splits])
    ax.set_ylabel("Median KGE")
    ax.set_ylim(0, 1.0)
    ax.set_title("Benchmark skill", fontweight="semibold")
    _grid(ax, axis="y")
    _panel_label(ax, "a", x=-0.12, y=1.02)

    # Build COSERO-paired gains at basin level.
    paired_rows = []
    for split in splits:
        wide = metrics[metrics["split"].eq(split)].pivot_table(
            index="ID", columns="model", values="kge", aggfunc="median"
        )
        if "cosero" not in wide.columns:
            continue
        for m in [mm for mm in models if mm != "cosero" and mm in wide.columns]:
            gain = pd.to_numeric(wide[m], errors="coerce") - pd.to_numeric(wide["cosero"], errors="coerce")
            for val in gain.dropna():
                paired_rows.append({"split": split, "model": m, "gain": float(val)})
    gains = pd.DataFrame(paired_rows)

    # (b) Median gain over COSERO.
    ax = axes[0, 1]
    gain_models = [m for m in models if m != "cosero" and m in set(gains.get("model", pd.Series(dtype=str)))]
    x = np.arange(len(gain_models), dtype=float)
    width = 0.34 if len(splits) > 1 else 0.58
    split_offsets = np.linspace(-width / 2, width / 2, len(splits)) if len(splits) > 1 else np.array([0.0])
    for off, split in zip(split_offsets, splits):
        vals = [
            float(gains.loc[(gains["split"].eq(split)) & (gains["model"].eq(m)), "gain"].median())
            if not gains.empty else np.nan
            for m in gain_models
        ]
        ax.bar(x + off, vals, width=width / max(len(splits), 1),
               color=[model_color(m) for m in gain_models],
               alpha=1.0 if split == "test" else 0.45,
               edgecolor=AXIS_COLOR, linewidth=0.35, zorder=3,
               label=SPLIT_DISPLAY.get(split, split))
    ax.axhline(0, color=AXIS_COLOR, lw=0.75, zorder=1)
    ax.axhspan(-0.05, 0.05, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([model_label(m) for m in gain_models], rotation=18, ha="right")
    ax.set_ylabel("Median KGE gain over COSERO")
    ax.set_title("Data-driven gain over benchmark", fontweight="semibold",
                 loc="left", pad=4)
    ax.legend(loc="upper right", fontsize=6.6,
              frameon=True, facecolor="white", edgecolor="0.85",
              framealpha=0.98).set_zorder(20)
    _grid(ax, axis="y")
    _panel_label(ax, "b", x=-0.12, y=1.02)

    # (c) Basin-level benchmark parity plot.
    ax = axes[1, 0]
    split = "test" if "test" in splits else splits[0]
    wide = metrics[metrics["split"].eq(split)].pivot_table(
        index="ID", columns="model", values="kge", aggfunc="median"
    )
    if "cosero" in wide.columns:
        for m in [mm for mm in ["random_forest", "xgboost", "xlstm", "transformer"] if mm in wide.columns]:
            ax.scatter(wide["cosero"], wide[m], s=12, color=model_color(m),
                       alpha=0.28, edgecolor="white", linewidth=0.15, label=model_label(m), zorder=3)
        lo = float(np.nanpercentile(wide.to_numpy(dtype=float), 2))
        hi = float(np.nanpercentile(wide.to_numpy(dtype=float), 98))
        lo, hi = min(lo, -0.3), max(hi, 1.0)
        ax.plot([lo, hi], [lo, hi], color=AXIS_COLOR, lw=0.8, ls="--", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("COSERO KGE")
    ax.set_ylabel("Data-driven model KGE")
    ax.set_title(f"Basin-level parity ({SPLIT_DISPLAY.get(split, split)})",
                 fontweight="semibold", loc="left", pad=4)
    ax.legend(loc="lower right", fontsize=6.2,
              frameon=True, facecolor="white", edgecolor="0.85",
              framealpha=0.98).set_zorder(20)
    _grid(ax, axis="both")
    _panel_label(ax, "c", x=-0.12, y=1.02)

    # (d) Signature ranking, including COSERO.
    ax = axes[1, 1]
    core = ["q_mean", "runoff_ratio", "baseflow_index", "fdc_slope",
            "q05", "q95", "high_flow_duration", "low_flow_duration"]
    if sig_summary.empty:
        ax.axis("off")
    else:
        ssplit = "test" if "test" in set(sig_summary["split"].astype(str)) else _pick_eval_split(sig_summary)
        sig = sig_summary[sig_summary["split"].eq(ssplit)].copy()
        sig_rows = []
        labels = []
        for signature in core:
            col = f"abs_err_{signature}"
            if col not in sig.columns:
                continue
            med = (
                sig.groupby("model")[col]
                .median()
                .reindex(models)
                .dropna()
            )
            if len(med) < 2:
                continue
            ranks = med.rank(method="min", ascending=True)
            sig_rows.append([float(ranks.get(m, np.nan)) for m in models])
            labels.append(SIG_LABELS.get(f"abs_err_{signature}", signature))
        if sig_rows:
            grid = np.asarray(sig_rows, dtype=float)
            im = ax.imshow(grid, cmap="Blues_r", vmin=1, vmax=max(2, len(models)), aspect="auto")
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if np.isfinite(grid[r, c]):
                        ax.text(c, r, f"{grid[r, c]:.0f}", ha="center", va="center",
                                fontsize=6.6, color="#111111")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([model_label(m) for m in models], rotation=20, ha="right", fontsize=6.8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6.8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
            cbar.set_label("Signature rank\n(1 = best)", fontsize=6.8)
            cbar.ax.tick_params(labelsize=6.5)
        else:
            ax.axis("off")
    ax.set_title("Signature benchmark rank", fontweight="semibold",
                 loc="left", pad=4)
    _panel_label(ax, "d", x=-0.12, y=1.02)

    ax_a = axes[0, 0]
    ax_a.set_title("Benchmark skill", fontweight="semibold",
                   loc="left", pad=4)

    handles = _model_legend_handles(models, lw=2.0)
    # Move the bottom legend a bit further down — at -0.005 it overlapped
    # the rotated x-tick labels of (b)/(c) when the dashboard was tight.
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7.6)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12,
                        wspace=0.30, hspace=0.55)
    _save(fig, fig_dir / "fig33_cosero_benchmark_dashboard.png")


# ─── Figure 34. Transferability dashboard ───────────────────────────────────

def plot_transferability_dashboard(metrics: pd.DataFrame,
                                   attrs_geo: pd.DataFrame,
                                   fig_dir: Path):
    if metrics.empty or attrs_geo.empty:
        return
    splits = [s for s in ("test", "spatial_test")
              if s in set(metrics["split"].dropna().astype(str))]
    if len(splits) < 2:
        return
    models = _present_models(metrics)
    if not models:
        return

    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)

    # (a) Slopegraph: temporal vs spatial KGE.
    ax = axes[0, 0]
    x = np.array([0, 1], dtype=float)
    endpoints = []
    for m in models:
        vals = []
        counts = []
        for split in ("test", "spatial_test"):
            d = pd.to_numeric(metrics.loc[(metrics["split"].eq(split)) & (metrics["model"].eq(m)), "kge"],
                              errors="coerce").dropna()
            vals.append(float(d.median()) if len(d) else np.nan)
            counts.append(int(len(d)))
        if not np.all(np.isfinite(vals)):
            continue
        ax.plot(x, vals, color=model_color(m), lw=2.0,
                linestyle=_model_linestyle(m), marker="o", ms=4.5,
                label=model_label(m), zorder=3)
        endpoints.append((m, float(vals[1])))
    if endpoints:
        endpoints = sorted(endpoints, key=lambda item: item[1])
        label_y = [v for _, v in endpoints]
        min_sep = 0.055
        for i in range(1, len(label_y)):
            if label_y[i] - label_y[i - 1] < min_sep:
                label_y[i] = label_y[i - 1] + min_sep
        overflow = label_y[-1] - 0.94
        if overflow > 0:
            label_y = [y - overflow for y in label_y]
        underflow = 0.06 - label_y[0]
        if underflow > 0:
            label_y = [y + underflow for y in label_y]
        for (m, actual_y), ylab in zip(endpoints, label_y):
            ax.plot([1.0, 1.035], [actual_y, ylab], color=model_color(m),
                    lw=0.55, alpha=0.85, zorder=3)
            ax.text(1.045, ylab, model_label(m), color=model_color(m),
                    fontsize=6.3, va="center", ha="left")
    ax.set_xlim(-0.08, 1.42)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Temporal test\n$n$={metrics[metrics['split'].eq('test')]['ID'].nunique()}",
                        f"Spatial holdout\n$n$={metrics[metrics['split'].eq('spatial_test')]['ID'].nunique()}"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Median KGE")
    ax.set_title("Transferability slopegraph", fontweight="semibold")
    _grid(ax, axis="y")
    _panel_label(ax, "a", x=-0.12, y=1.02)

    # (b) Generalisation delta heatmap.
    ax = axes[0, 1]
    metric_specs = [("kge", "KGE"), ("nse", "NSE"), ("log_nse", "log-NSE"), ("abs_pbias", "|PBIAS|")]
    rows = []
    for m in models:
        vals = []
        for metric, _ in metric_specs:
            if metric == "abs_pbias":
                t = pd.to_numeric(metrics.loc[(metrics["split"].eq("test")) & (metrics["model"].eq(m)), "pbias"],
                                  errors="coerce").abs().median()
                s = pd.to_numeric(metrics.loc[(metrics["split"].eq("spatial_test")) & (metrics["model"].eq(m)), "pbias"],
                                  errors="coerce").abs().median()
                vals.append(float(s - t))
            else:
                t = pd.to_numeric(metrics.loc[(metrics["split"].eq("test")) & (metrics["model"].eq(m)), metric],
                                  errors="coerce").median()
                s = pd.to_numeric(metrics.loc[(metrics["split"].eq("spatial_test")) & (metrics["model"].eq(m)), metric],
                                  errors="coerce").median()
                vals.append(float(s - t))
        rows.append(vals)
    grid = np.asarray(rows, dtype=float)
    lim = max(0.15, float(np.nanpercentile(np.abs(grid[:, :3]), 95)) if np.isfinite(grid[:, :3]).any() else 0.25)
    # Put |PBIAS| on a comparable visual scale by clipping to ±20 percentage points.
    plot_grid = grid.copy()
    plot_grid[:, 3] = np.clip(plot_grid[:, 3] / 100.0, -0.2, 0.2)
    norm = TwoSlopeNorm(vmin=-max(lim, 0.2), vcenter=0.0, vmax=max(lim, 0.2))
    im = ax.imshow(plot_grid, cmap="RdBu_r", norm=norm, aspect="auto")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            txt = f"{v:+.2f}" if c < 3 else f"{v:+.1f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=6.6,
                    color="white" if abs(plot_grid[r, c]) > 0.12 else "#111111")
    ax.set_xticks(range(len(metric_specs)))
    ax.set_xticklabels([lab for _, lab in metric_specs], fontsize=7.0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_label(m) for m in models], fontsize=7.0)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Spatial − temporal generalisation delta", fontweight="semibold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
    cbar.set_label("Scaled delta", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.5)
    _panel_label(ax, "b", x=-0.12, y=1.02)

    # (c) Reliability thresholds.
    ax = axes[1, 0]
    threshold_cols = [("KGE ≥ 0.5", lambda d: d["kge"] >= 0.5),
                      ("KGE ≥ 0.7", lambda d: d["kge"] >= 0.7),
                      ("|PBIAS| ≤ 10%", lambda d: d["pbias"].abs() <= 10)]
    rows = []
    row_labels = []
    for m in models:
        for split in ("test", "spatial_test"):
            d = metrics[(metrics["model"].eq(m)) & (metrics["split"].eq(split))].copy()
            if d.empty:
                continue
            for c in ["kge", "pbias"]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            rows.append([float(rule(d).mean() * 100.0) for _, rule in threshold_cols])
            row_labels.append(f"{model_label(m)} · {SPLIT_DISPLAY.get(split, split)}")
    reli = np.asarray(rows, dtype=float)
    im = ax.imshow(reli, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for r in range(reli.shape[0]):
        for c in range(reli.shape[1]):
            ax.text(c, r, f"{reli[r, c]:.0f}", ha="center", va="center",
                    fontsize=6.2, color="white" if reli[r, c] > 58 else "#111111")
    ax.set_xticks(range(len(threshold_cols)))
    ax.set_xticklabels([lab for lab, _ in threshold_cols], fontsize=6.8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.8)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Transfer reliability (% basins)", fontweight="semibold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
    cbar.set_label("%", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.5)
    _panel_label(ax, "c", x=-0.12, y=1.02)

    # (d) Evaluation-set process composition.
    ax = axes[1, 1]
    basin_splits = metrics[metrics["split"].isin(["test", "spatial_test"])][["split", "ID"]].drop_duplicates()
    comp = basin_splits.merge(attrs_geo, on="ID", how="left")
    zones = [("snow_class", "Snow"), ("aridity_class", "Aridity"), ("storage_class", "Storage")]
    y = 0
    ytick_pos = []
    ytick_lab = []
    for zone, label in zones:
        values = [v for v in ZONE_VALUE_ORDER.get(zone, [])
                  if v in set(comp[zone].dropna().astype(str))]
        if not values:
            continue
        colors = [REGIME_PALETTES.get(zone, {}).get(v, MUTED_COLOR) for v in values]
        for split in ("test", "spatial_test"):
            d = comp[comp["split"].eq(split)]
            total = max(int(d["ID"].nunique()), 1)
            left = 0.0
            for val, color in zip(values, colors):
                pct = 100.0 * d[d[zone].astype(str).eq(val)]["ID"].nunique() / total
                ax.barh(y, pct, left=left, color=color, edgecolor="white",
                        linewidth=0.35, height=0.7, zorder=3)
                if pct >= 13:
                    label_color = "white" if color in {"#0E2A55", "#B5302A"} else "#111111"
                    ax.text(left + pct / 2, y, zone_value_pretty(val),
                            ha="center", va="center", fontsize=4.8,
                            color=label_color, zorder=4)
                left += pct
            ytick_pos.append(y)
            ytick_lab.append(f"{label} · {SPLIT_DISPLAY.get(split, split)}")
            y += 1
        y += 0.45
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of evaluation basins (%)")
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=6.4)
    ax.set_title("Temporal/spatial process coverage", fontweight="semibold")
    _grid(ax, axis="x")
    _panel_label(ax, "d", x=-0.12, y=1.02)

    fig.tight_layout()
    _save(fig, fig_dir / "fig34_transferability_dashboard.png")


# ─── Figure 35. COSERO transfer benchmark maps ─────────────────────────────

def plot_cosero_transfer_maps(metrics: pd.DataFrame,
                              attrs_geo: pd.DataFrame,
                              fig_dir: Path):
    if metrics.empty or attrs_geo.empty:
        return
    required = {"ID", "lon_deg", "lat_deg"}
    if not required.issubset(attrs_geo.columns):
        return
    needed = {"cosero", "xlstm"}
    if not needed.issubset(set(metrics["model"].dropna().astype(str))):
        return
    splits = [s for s in ("test", "spatial_test") if s in set(metrics["split"].dropna().astype(str))]
    if not splits:
        return

    geo = attrs_geo[["ID", "lon_deg", "lat_deg"]].copy()
    skill_norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.5, vmax=1.0)
    delta_norm = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.35)
    n_rows = len(splits)
    use_cartopy, proj, data_crs, feats = _init_cartopy()

    lon_min, lon_max = geo["lon_deg"].min(), geo["lon_deg"].max()
    lat_min, lat_max = geo["lat_deg"].min(), geo["lat_deg"].max()
    lon_pad = max((lon_max - lon_min) * 0.06, 0.2)
    lat_pad = max((lat_max - lat_min) * 0.06, 0.2)
    extent = (max(lon_min - lon_pad, 5.5), min(lon_max + lon_pad, 22.0),
              max(lat_min - lat_pad, 44.5), min(lat_max + lat_pad, 52.5))

    fig = plt.figure(figsize=(COL_DOUBLE, 4.85))
    gs = fig.add_gridspec(
        n_rows, 5,
        width_ratios=[1.0, 1.0, 1.0, 0.04, 0.04],
        wspace=0.10,
        hspace=0.20,
    )
    axes_grid = []
    for r in range(n_rows):
        row = []
        for c in range(3):
            if use_cartopy:
                ax = fig.add_subplot(gs[r, c], projection=proj)
                _add_basemap(ax, feats, extent=extent, data_crs=data_crs,
                             gridlines=False)
            else:
                ax = fig.add_subplot(gs[r, c])
                ax.set_facecolor(LAND_COLOR)
            row.append(ax)
        axes_grid.append(row)
    axes = np.asarray(axes_grid, dtype=object)
    skill_cax = fig.add_subplot(gs[:, 3])
    delta_cax = fig.add_subplot(gs[:, 4])
    sc_skill = None
    sc_delta = None
    panel_i = 0

    for r, split in enumerate(splits):
        wide = metrics[metrics["split"].eq(split)].pivot_table(
            index="ID", columns="model", values="kge", aggfunc="median"
        )
        if not {"cosero", "xlstm"}.issubset(wide.columns):
            continue
        wide = wide[["cosero", "xlstm"]].dropna()
        wide["xlstm_minus_cosero"] = wide["xlstm"] - wide["cosero"]
        plot_df = wide.reset_index().merge(geo, on="ID", how="left").dropna(
            subset=["lon_deg", "lat_deg"]
        )
        if plot_df.empty:
            continue
        panels = [
            ("cosero", "COSERO benchmark", "skill"),
            ("xlstm", "xLSTM regional model", "skill"),
            ("xlstm_minus_cosero", "xLSTM − COSERO", "delta"),
        ]
        for c, (col, title, kind) in enumerate(panels):
            ax = axes[r, c]
            sub = plot_df.sort_values(col)
            scatter_kw = dict(edgecolor="white", linewidth=0.35,
                              alpha=0.96, zorder=3)
            if use_cartopy:
                scatter_kw["transform"] = data_crs
            if kind == "skill":
                sc_skill = ax.scatter(sub["lon_deg"], sub["lat_deg"], c=sub[col],
                                      cmap=SKILL_CMAP, norm=skill_norm,
                                      s=16 if split == "test" else 28,
                                      **scatter_kw)
                med_txt = f"median KGE={sub[col].median():.2f}"
            else:
                sc_delta = ax.scatter(sub["lon_deg"], sub["lat_deg"], c=sub[col],
                                      cmap="RdBu", norm=delta_norm,
                                      s=18 if split == "test" else 30,
                                      **scatter_kw)
                med_txt = (
                    f"median ΔKGE={sub[col].median():+.2f}\n"
                    f"wins={(sub[col] > 0).mean() * 100:.0f}%"
                )
            if not use_cartopy:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xticks([])
                ax.set_yticks([])
                _despine(ax, keep=())
            if r == 0:
                ax.set_title(title, fontsize=8.6, fontweight="semibold", pad=6)
            if c == 0:
                ax.text(-0.05, 0.50, SPLIT_DISPLAY.get(split, split),
                        transform=ax.transAxes, rotation=90,
                        fontsize=8.2, fontweight="semibold",
                        ha="center", va="center", clip_on=False)
            ax.text(0.03, 0.97,
                    f"{med_txt}\n$n$={plot_df['ID'].nunique()}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.3, color=ANNOT_COLOR,
                    bbox=dict(facecolor="white", edgecolor="0.85",
                              alpha=0.9, pad=1.6,
                              boxstyle="round,pad=0.25"), zorder=20)
            ax.text(0.97, 0.04, f"({_panel_letter(panel_i)})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.6, fontweight="bold", color=AXIS_COLOR,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.82, pad=1.0), zorder=20)
            panel_i += 1

    if sc_skill is not None:
        cbar = fig.colorbar(sc_skill, cax=skill_cax)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.set_title("KGE", fontsize=6.4, pad=4)
        cbar.ax.tick_params(labelsize=6.0, pad=1)
    else:
        skill_cax.set_visible(False)
    if sc_delta is not None:
        cbar = fig.colorbar(sc_delta, cax=delta_cax)
        cbar.set_ticks([-0.3, 0.0, 0.3])
        cbar.ax.set_title("ΔKGE", fontsize=6.4, pad=4)
        cbar.ax.tick_params(labelsize=6.0, pad=1)
    else:
        delta_cax.set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.04,
                        wspace=0.10, hspace=0.20)
    _save(fig, fig_dir / "fig35_cosero_transfer_maps.png")


# ─── Experiment-audit figures ────────────────────────────────────────────────

def _read_sequence_history(out_dir: Path) -> pd.DataFrame:
    tables = out_dir / "tables"
    combined = tables / "sequence_training_history.csv"
    if combined.exists():
        return _read(combined)
    parts = [_read(path) for path in sorted(tables.glob("sequence_training_history_*.csv"))]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_training_and_tuning_audit(out_dir: Path, fig_dir: Path):
    """Visual audit of model selection: sequence convergence + tabular validation
    search. Panels are laid out dynamically from the data that actually exists;
    no placeholder panels and no figure-level title."""
    hist = _read_sequence_history(out_dir)
    tuning = _read(out_dir / "tables" / "tuning_log.csv")
    if hist.empty and tuning.empty:
        return

    seq_models = [m for m in ["xlstm", "transformer"]
                  if not hist.empty and hist["model"].astype(str).eq(m).any()]
    tab_models = [m for m in ["random_forest", "xgboost"]
                  if not tuning.empty and "model" in tuning.columns
                  and {"candidate", "score"}.issubset(tuning.columns)
                  and tuning["model"].astype(str).eq(m).any()]
    n_panels = len(seq_models) + len(tab_models)
    if n_panels == 0:
        return
    nrows = 2 if n_panels > 2 else 1
    ncols = 2 if n_panels > 1 else 1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL_DOUBLE, 2.85 if nrows == 1 else 4.85),
                             squeeze=False)
    axes = axes.ravel()
    for extra in axes[n_panels:]:
        extra.set_visible(False)
    for i, model in enumerate(seq_models):
        ax = axes[i]
        sub = hist[hist["model"].astype(str).eq(model)].copy()
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub["train_loss"], color=model_color(model), lw=1.15,
                ls=(0, (3, 2)), label="train")
        ax.plot(sub["epoch"], sub["val_loss"], color=model_color(model), lw=1.9,
                label="validation")
        best = sub.loc[sub["val_loss"].idxmin()]
        ax.scatter([best["epoch"]], [best["val_loss"]], s=32, color="white",
                   edgecolor=model_color(model), linewidth=1.2, zorder=5)
        _annot(
            ax,
            f"best epoch {int(best['epoch'])}\nlast epoch {int(sub['epoch'].max())}",
            x=0.98, y=0.96, ha="right",
            bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98, pad=1.8),
        )
        ax.set_title(f"{model_label(model)} convergence", fontweight="semibold", loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NSE-loss" if str(sub["loss"].iloc[-1]).lower() == "nse" else "Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _grid(ax, axis="both")
        _panel_label(ax, _panel_letter(i))
        if i == 0:
            ax.legend(loc="upper right", fontsize=6.6, frameon=False,
                      handletextpad=0.4, borderaxespad=0.2)

    for j, model in enumerate(tab_models, start=len(seq_models)):
        ax = axes[j]
        sub = tuning[tuning["model"].astype(str).eq(model)].copy()
        sub = sub.sort_values("candidate")
        ax.plot(sub["candidate"], sub["score"], marker="o", ms=4.2, lw=1.5,
                color=model_color(model), label="candidate score")
        best = sub.loc[sub["score"].idxmin()]
        ax.scatter([best["candidate"]], [best["score"]], marker="*", s=110,
                   color=model_color(model), edgecolor="white", linewidth=0.6,
                   zorder=5, label="selected")
        _annot(
            ax,
            f"selected candidate {int(best['candidate'])}\nmedian KGE {best['median_kge']:.3f}",
            x=0.98, y=0.96, ha="right",
            bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98, pad=1.8),
        )
        ax.set_title(f"{model_label(model)} validation search", fontweight="semibold", loc="left")
        ax.set_xlabel("Candidate")
        ax.set_ylabel("Composite validation score\n(lower is better)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _grid(ax, axis="both")
        _panel_label(ax, _panel_letter(j))
        if j == len(seq_models):
            ax.legend(loc="upper right", fontsize=6.6, frameon=False,
                      handletextpad=0.4, borderaxespad=0.2)

    fig.tight_layout()
    _save(fig, fig_dir / "fig36_training_tuning_audit.png")


def plot_model_setup_audit(out_dir: Path, fig_dir: Path):
    """Compact visual of actual fitted model setup and convergence controls."""
    tables = out_dir / "tables"
    selected_path = tables / "selected_hyperparameters.json"
    seq_setups = []
    for path in sorted(tables.glob("sequence_model_setup_*.json")):
        try:
            seq_setups.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    selected = {}
    if selected_path.exists():
        try:
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
        except Exception:
            selected = {}
    if not seq_setups and not selected:
        return

    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 3.55))
    ax = axes[0]
    if seq_setups:
        seq = pd.DataFrame(seq_setups)
        x = np.arange(len(seq))
        ax.bar(x, seq["parameters"].astype(float) / 1000.0,
               color=[model_color(m) for m in seq["model"]], width=0.58, zorder=3)
        for xi, (_, row) in zip(x, seq.iterrows()):
            label = f"{int(row.get('sequence_length', 0))} d\nbest {int(row.get('best_epoch', 0) or 0)}/{int(row.get('last_epoch', 0) or 0)}"
            ax.text(xi, float(row["parameters"]) / 1000.0, label,
                    ha="center", va="bottom", fontsize=6.4, color=ANNOT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([model_label(m) for m in seq["model"]])
        ax.set_ylabel("Trainable parameters (thousand)")
        ax.set_title("Sequence-model setup", fontweight="semibold", loc="left")
        _grid(ax, axis="y")
    else:
        ax.text(0.5, 0.5, "sequence setup\nnot available", ha="center", va="center",
                transform=ax.transAxes, color=MUTED_COLOR)
        ax.set_axis_off()
    _panel_label(ax, "a")

    ax = axes[1]
    rows = []
    for model in ["random_forest", "xgboost"]:
        params = selected.get(model, {})
        if not params:
            continue
        rows.append({
            "model": model_label(model),
            "n_estimators": params.get("n_estimators", np.nan),
            "max_depth": params.get("max_depth", np.nan),
            "learning_rate": params.get("learning_rate", np.nan),
            "min_samples_leaf": params.get("min_samples_leaf", np.nan),
        })
    ax.axis("off")
    if rows:
        text_lines = ["Validation-selected tabular setup"]
        for row in rows:
            vals = [f"{k}={v}" for k, v in row.items() if k != "model" and pd.notna(v)]
            text_lines.append(f"{row['model']}: " + ", ".join(vals[:4]))
        text_lines += ["", "Selection rule: validation split only",
                       "Final temporal/spatial test metrics are locked out"]
        ax.text(0.02, 0.96, "\n".join(text_lines), ha="left", va="top",
                transform=ax.transAxes, fontsize=7.2,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98, pad=4.0))
    else:
        ax.text(0.5, 0.5, "tabular setup\nnot available", ha="center", va="center",
                transform=ax.transAxes, color=MUTED_COLOR)
    ax.set_title("Tabular-model setup", fontweight="semibold", loc="left")
    _panel_label(ax, "b")
    fig.tight_layout()
    _save(fig, fig_dir / "fig37_model_setup_audit.png")


def plot_uncertainty_dashboard(out_dir: Path, fig_dir: Path):
    """Two clean panels: (a) median KGE with bootstrap CIs per model and split,
    (b) forest plot of all paired contrasts. Split semantics: filled = temporal
    test, open = spatial test; colour always identifies the model. Panels with
    no data are omitted entirely — never drawn as placeholders."""
    metric_ci = _read(out_dir / "tables" / "metric_uncertainty.csv")
    contrast_ci = _read(out_dir / "tables" / "model_contrast_uncertainty.csv")
    fold_summary = _read(out_dir / "tables" / "spatial_fold_summary.csv")

    kge = pd.DataFrame()
    if not metric_ci.empty and "metric" in metric_ci.columns:
        kge = metric_ci[metric_ci["metric"].eq("kge")
                        & metric_ci["split"].isin(["test", "spatial_test"])].copy()
    contrasts = pd.DataFrame()
    if not contrast_ci.empty:
        keep = ["memory_gain_xlstm_vs_rf", "transformer_gain_vs_rf",
                "sequence_gain_vs_tree", "xlstm_minus_transformer",
                "xlstm_gain_vs_cosero", "xgboost_gain_vs_cosero",
                "transformer_gain_vs_cosero"]
        contrasts = contrast_ci[contrast_ci["split"].isin(["test", "spatial_test"])
                                & contrast_ci["contrast"].isin(keep)].copy()
        contrasts["order"] = contrasts["contrast"].map({c: i for i, c in enumerate(keep)})
        contrasts = contrasts.sort_values(["order", "split"], ascending=[True, True])

    panels = [p for p, ok in (("skill", not kge.empty),
                              ("contrast", not contrasts.empty),
                              ("folds", not fold_summary.empty)) if ok]
    if not panels:
        return
    widths = {"skill": 1.0, "contrast": 1.35, "folds": 1.0}
    fig, axes = plt.subplots(
        1, len(panels), figsize=(COL_DOUBLE, 3.15),
        gridspec_kw={"width_ratios": [widths[p] for p in panels]})
    axes = np.atleast_1d(axes)
    split_marker = {"test": dict(fillstyle="full"), "spatial_test": dict(fillstyle="none")}

    for p_i, panel in enumerate(panels):
        ax = axes[p_i]
        if panel == "skill":
            models = [m for m in MODEL_ORDER if m in set(kge["model"].astype(str))]
            width = 0.30
            for s_i, split in enumerate(["test", "spatial_test"]):
                sub = kge[kge["split"].eq(split)].set_index("model").reindex(models)
                if sub["median"].isna().all():
                    continue
                x = np.arange(len(models)) + (s_i - 0.5) * width
                y = sub["median"].astype(float).to_numpy()
                yerr = np.vstack([y - sub["ci_low"].astype(float).to_numpy(),
                                  sub["ci_high"].astype(float).to_numpy() - y])
                for m_i, m in enumerate(models):
                    ax.errorbar(x[m_i], y[m_i], yerr=yerr[:, m_i:m_i + 1], fmt="o",
                                ms=4.6, lw=1.1, capsize=2.4, color=model_color(m),
                                markeredgecolor=model_color(m), zorder=3,
                                **split_marker[split])
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels([model_label(m).replace(" ", "\n") for m in models],
                               fontsize=6.6)
            ax.set_ylabel("Median KGE (95 % CI)")
            hdl = [Line2D([], [], marker="o", ls="", color=AXIS_COLOR,
                          fillstyle="full", label="temporal test"),
                   Line2D([], [], marker="o", ls="", color=AXIS_COLOR,
                          fillstyle="none", label="spatial test")]
            ax.legend(handles=hdl, loc="lower left", fontsize=6.4, frameon=False,
                      handletextpad=0.3, borderaxespad=0.2)
            ax.set_title("Skill uncertainty", fontweight="semibold", loc="left")
            _grid(ax, axis="y")
        elif panel == "contrast":
            y_pos, labels_out = [], []
            y_cur = 0.0
            for c in contrasts["contrast"].drop_duplicates():
                grp = contrasts[contrasts["contrast"].eq(c)]
                for _, row in grp.iterrows():
                    x = float(row["median_delta"])
                    lo, hi = float(row["ci_low"]), float(row["ci_high"])
                    is_test = row["split"] == "test"
                    ax.errorbar(x, y_cur, xerr=[[x - lo], [hi - x]], fmt="o", ms=4.2,
                                lw=1.0, capsize=2.2, color=AXIS_COLOR, zorder=3,
                                fillstyle="full" if is_test else "none")
                    y_pos.append(y_cur)
                    lbl = str(row.get("label", c)).replace(" KGE", "")
                    labels_out.append(f"{lbl}  ({'T' if is_test else 'S'})")
                    y_cur += 1.0
                y_cur += 0.55
            ax.axvline(0, color=RULE_COLOR, lw=0.8, ls="--", zorder=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_out, fontsize=6.0)
            ax.invert_yaxis()
            ax.set_xlabel("Paired median KGE difference")
            ax.set_title("Paired contrasts (T = temporal, S = spatial)",
                         fontweight="semibold", loc="left")
            _grid(ax, axis="x")
        else:
            sub = fold_summary[fold_summary["split"].eq("spatial_test")].copy()
            models = [m for m in MODEL_ORDER if m in set(sub["model"].astype(str))]
            sub = sub.set_index("model").reindex(models)
            x = np.arange(len(models))
            y = sub["fold_mean_median_kge"].astype(float).to_numpy()
            sd = sub["fold_sd_median_kge"].astype(float).to_numpy()
            ax.bar(x, y, yerr=sd, capsize=2.5, color=[model_color(m) for m in models],
                   width=0.58, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([model_label(m).replace(" ", "\n") for m in models],
                               fontsize=6.6)
            ax.set_ylabel("Mean fold-median KGE ± SD")
            ax.set_title("Spatial-fold stability", fontweight="semibold", loc="left")
            _grid(ax, axis="y")
        _despine(ax)
        _panel_label(ax, _panel_letter(p_i))

    fig.tight_layout()
    _save(fig, fig_dir / "fig38_uncertainty_dashboard.png")


# ─── Figure 2. Spatial map ───────────────────────────────────────────────────

def plot_basin_map(metrics: pd.DataFrame, attrs_geo: pd.DataFrame, fig_dir: Path):
    if metrics.empty or attrs_geo.empty or "lon_deg" not in attrs_geo.columns:
        return
    split = _pick_eval_split(metrics)
    if split is None:
        return
    test = metrics[metrics["split"].eq(split)].merge(
        attrs_geo[["ID", "lon_deg", "lat_deg"]], on="ID", how="left"
    ).dropna(subset=["lon_deg", "lat_deg", "kge"])
    if test.empty:
        return
    models = _present_models(test)
    if not models:
        return

    use_cartopy, proj, data_crs, cartopy_features = _init_cartopy()

    n_models = len(models)
    # One row when the model count fits — keeps every panel the same size and
    # avoids a blank slot in 2x2 grids when there are only 3 models.
    if n_models <= 3:
        cols, rows = n_models, 1
    elif n_models == 4:
        cols, rows = 4, 1
    elif n_models <= 6:
        cols, rows = 3, 2
    else:
        cols, rows = 3, int(np.ceil(n_models / 3))
    # Maps in degrees (without cartopy) need plenty of vertical space because
    # the equal-aspect ratio compresses them otherwise. Single-row layouts get
    # extra height so the panels read as proper maps, not strips. The export
    # uses bbox_inches="tight", which trims unused canvas around the panels.
    panel_h = 3.85 if rows == 1 else 2.55
    fig_h = panel_h * rows + 0.6
    fig = plt.figure(figsize=(COL_DOUBLE, fig_h))
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.5, vmax=1.0)
    cmap = plt.cm.get_cmap(SKILL_CMAP)

    gs = fig.add_gridspec(rows, cols + 1,
                          width_ratios=[1.0] * cols + [0.045],
                          wspace=0.12, hspace=0.32)

    last_sc = None
    letters = "abcdefgh"
    last_row_index = (len(models) - 1) // cols
    for k, model in enumerate(models):
        r, c = divmod(k, cols)
        # If this column has no panel below me, treat me as a bottom panel so
        # the longitude label and ticks render (handles 5-on-3x2 layouts).
        is_bottom = (r == last_row_index) or (k + cols >= len(models))
        if use_cartopy:
            ax = fig.add_subplot(gs[r, c], projection=proj)
            _add_basemap(ax, cartopy_features, extent=(7.5, 18.5, 45.0, 51.5),
                         data_crs=data_crs, gridlines=True, label_size=6.4,
                         label_left=(c == 0),
                         label_bottom=is_bottom)
        else:
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(LAND_COLOR)
        g = test[test["model"].eq(model)].sort_values("kge")
        scatter_kwargs = dict(c=g["kge"], cmap=cmap, norm=norm, s=15,
                              edgecolor="white", linewidth=0.25, zorder=4)
        if use_cartopy:
            scatter_kwargs["transform"] = data_crs
        last_sc = ax.scatter(g["lon_deg"], g["lat_deg"], **scatter_kwargs)
        med = g["kge"].median(); q25 = g["kge"].quantile(0.25); q75 = g["kge"].quantile(0.75)
        # Combined panel-letter + model title — single line, single anchor — so
        # nothing in row r overlaps row r-1's gridline labels.
        ax.set_title(f"({letters[k]})  {model_label(model)}",
                     fontsize=9.0, fontweight="semibold",
                     loc="left", pad=4)
        # Stats tile in upper-RIGHT — left-side cartopy gridline labels
        # ("51°N", "50°N", …) sit in the upper-left and were colliding with
        # the box previously.
        stat = (f"$n$ = {len(g)}\n"
                f"median KGE = {med:.2f}\n"
                f"IQR [{q25:.2f}, {q75:.2f}]")
        ax.text(0.98, 0.97, stat, transform=ax.transAxes,
                fontsize=6.6, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98,
                          pad=2.0, boxstyle="round,pad=0.28"), zorder=20)
        if not use_cartopy:
            # Use a fixed lat:lon ratio (~1.5 ≈ cos(48°)·1) so panels don't
            # collapse to letterbox strips, but still look spatially honest.
            ax.set_aspect(1.5)
            ax.set_xlim(7.5, 18.5)
            ax.set_ylim(45.0, 51.5)
            if is_bottom:
                ax.set_xlabel("Longitude (°E)", fontsize=7.5)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel("Latitude (°N)", fontsize=7.5)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    # Empty slots are intentionally NOT added — when 5 models land on a 3x2
    # grid, simply leaving slot (1,2) absent lets the tight-bbox export drop
    # the unused area instead of preserving it as whitespace.

    # Single shared colorbar on the right gutter — replaces the floating
    # add_axes block, which collided with the title row when fig height
    # changed. The Europe inset is dropped because it crowded panel (a) and
    # the inset position is hard to keep stable across N-panel layouts.
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_sc, cax=cbar_ax, extend="both")
    cbar.set_label(f"{SPLIT_DISPLAY.get(split, split)} KGE", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.94, bottom=0.08,
                        wspace=0.12, hspace=0.30)
    _save(fig, fig_dir / "fig02_basin_map_kge.png")


# ─── Figure 3. Signature error grouped bars ──────────────────────────────────

def plot_signature_errors(sig: pd.DataFrame, fig_dir: Path):
    if sig.empty:
        return
    split = _pick_eval_split(sig)
    if split is None:
        return
    test = sig[sig["split"].eq(split)].copy()
    if test.empty:
        return
    models = _present_models(test)
    if not models:
        return
    sig_cols = [c for c in SIG_LABELS if c in test.columns]
    sig_cols = [c for c in sig_cols
                if pd.to_numeric(test[c], errors="coerce").abs().median() > 1e-6
                or test[c].notna().sum() > 0]
    if not sig_cols:
        return
    medians = test.groupby("model")[sig_cols].median().reindex(models)

    # Three unit groups so no signature dominates the others. COSERO can have
    # ~9 d low-flow duration error while xLSTM is ~1.7 d — putting them on
    # one shared y-axis with mean-discharge errors of ~0.1 hides the signal.
    days_cols  = [c for c in sig_cols if c in {"abs_err_hfd_mean",
                                                "abs_err_high_q_dur",
                                                "abs_err_low_q_dur"}]
    freq_cols  = [c for c in sig_cols if c in {"abs_err_high_q_freq",
                                                "abs_err_low_q_freq",
                                                "abs_err_zero_q_freq"}]
    unit_cols  = [c for c in sig_cols if c not in days_cols + freq_cols]
    pretty = lambda cs: [SIG_LABELS[c] for c in cs]

    # Drop the frequency panel if every model's median is essentially zero
    # — using max() across all rows kept the panel even when only stray
    # outliers were non-zero, which produced one stub bar over an empty axis.
    freq_cols_kept = []
    for c in freq_cols:
        col_med = medians[c].abs().max() if c in medians.columns else 0.0
        if col_med > 5e-3:
            freq_cols_kept.append(c)
    panels = [(unit_cols, "Dimensionless signatures",
               "Median |signature error|"),
              (days_cols, "Duration / timing signatures",
               "Median |error| (days)"),
              (freq_cols_kept, "Frequency signatures",
               "Median |error| (events yr$^{-1}$)")]
    panels = [(cols, title, ylab) for cols, title, ylab in panels if cols]
    if not panels:
        return

    width_ratios = [max(len(c), 1) + 0.4 for c, _, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=PAPER_FIGSIZE,
                             gridspec_kw={"width_ratios": width_ratios,
                                          "wspace": 0.32})
    if len(panels) == 1:
        axes = [axes]

    width = 0.78 / max(len(models), 1)

    def _draw(ax, cols, title, ylabel, panel_letter):
        x = np.arange(len(cols))
        for i, m in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, medians.loc[m, cols].values, width=width,
                   color=model_color(m), edgecolor="white", linewidth=0.5,
                   label=model_label(m), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(pretty(cols), rotation=28, ha="right", fontsize=7.4)
        ax.set_ylabel(ylabel, fontsize=8.0)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_title(f"({panel_letter})  {title}", fontsize=8.6,
                     fontweight="semibold", loc="left", pad=4)
        _grid(ax, axis="y")

    letters = "abcdef"
    for k, ((cols, title, ylab), ax) in enumerate(zip(panels, axes)):
        _draw(ax, cols, title, ylab, letters[k])

    # One shared legend at the very bottom — keeps it out of every panel's
    # value space and avoids the previous overlap with panel (a)'s title.
    handles = _model_legend_handles(models, lw=2.4)
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=7.6)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.20)
    _save(fig, fig_dir / "fig03_signature_errors.png")


# ─── Figure 4. Regime violins with quartile markers ──────────────────────────

def plot_regime_violins(skill_attrs: pd.DataFrame, fig_dir: Path):
    if skill_attrs.empty:
        return
    models = _present_models(skill_attrs)
    if not models:
        return
    panels = [p for p in ["snow_class", "aridity_class", "storage_class", "impact_class"]
              if p in skill_attrs.columns]
    if not panels:
        return
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=PAPER_FIGSIZE,
                             squeeze=False)
    letters = "abcdefgh"
    for k, zone in enumerate(panels):
        ax = axes[k // cols, k % cols]
        order = ZONE_VALUE_ORDER.get(zone, sorted(skill_attrs[zone].dropna().unique().tolist()))
        order = [z for z in order if z in skill_attrs[zone].astype(str).unique()
                 and z != "unknown"]
        x_pos = np.arange(len(order))
        width = 0.78 / max(len(models), 1)
        # n-counts above panel
        ax.set_xlim(-0.6, len(order) - 0.4)
        for i, m in enumerate(models):
            offsets = (i - (len(models) - 1) / 2) * width
            for j, val in enumerate(order):
                vals = pd.to_numeric(
                    skill_attrs.loc[(skill_attrs["model"].eq(m)) & (skill_attrs[zone].astype(str).eq(val)), "kge"],
                    errors="coerce",
                ).dropna().to_numpy()
                if vals.size < 3:
                    continue
                pos = x_pos[j] + offsets
                vp = ax.violinplot([vals], positions=[pos], widths=width * 0.95,
                                   showmeans=False, showextrema=False, showmedians=False)
                for b in vp["bodies"]:
                    b.set_facecolor(model_color(m))
                    b.set_edgecolor(model_color(m))
                    b.set_alpha(0.45)
                    b.set_linewidth(0.4)
                # Quartile bar (Q25–Q75) + median tick — Tukey-style overlay
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                ax.plot([pos, pos], [q25, q75], color=model_color(m),
                        lw=2.0, alpha=0.95, solid_capstyle="butt", zorder=4)
                ax.plot([pos - width * 0.32, pos + width * 0.32], [q50, q50],
                        color="white", lw=1.0, zorder=5)
        ax.axhline(0.5, color=RULE_COLOR, lw=0.5, ls=":", zorder=1)
        ax.axhline(0.0, color=SLATE_COLOR, lw=0.5, zorder=1)
        ax.set_xticks(x_pos)
        # Basin counts go on the SECOND line of the x-tick label — keeps the
        # upper plotting area clean and the count stays glued to its category.
        tick_labels = []
        for j, val in enumerate(order):
            n_bas = int(skill_attrs.loc[(skill_attrs["model"].eq(models[0])) & (skill_attrs[zone].astype(str).eq(val))].shape[0])
            tick_labels.append(f"{zone_value_pretty(val)}\n$n$={n_bas}")
        ax.set_xticklabels(tick_labels, fontsize=7.5, linespacing=1.05)
        ax.set_ylim(-0.25, 1.0)
        ax.set_ylabel("Test KGE")
        ax.set_title(ZONE_LABELS.get(zone, zone), fontsize=9.5, fontweight="semibold")
        _grid(ax, axis="y")
        _panel_label(ax, _panel_letter(k), x=-0.10, y=1.02)
    fig.legend(handles=_model_legend_handles(models), loc="lower center",
               ncol=len(models), bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    _save(fig, fig_dir / "fig04_regime_violins.png")


# ─── Figure 5. Skill conditioned on attribute (LOWESS) ───────────────────────

def plot_skill_vs_attribute(skill_attrs: pd.DataFrame, fig_dir: Path):
    if skill_attrs.empty:
        return
    models = _present_models(skill_attrs)
    if not models:
        return
    panels = [
        ("frac_snow", "Fraction of precipitation as snow", False),
        ("aridity",   "Aridity index (PET / P)",           False),
        ("bfi",       "Baseflow index (LamaH static)",     False),
        ("area_calc", "Catchment area (km²)",              True),
    ]
    panels = [p for p in panels if p[0] in skill_attrs.columns]
    fig, axes = plt.subplots(2, 2, figsize=PAPER_FIGSIZE, squeeze=False)
    for k, (col, label, log_x) in enumerate(panels):
        ax = axes[k // 2, k % 2]
        rho_lines: list[tuple[str, float]] = []
        for m in models:
            sub = skill_attrs[skill_attrs["model"].eq(m)].dropna(subset=[col, "kge"])
            if sub.empty:
                continue
            ax.scatter(sub[col], sub["kge"], s=7, color=model_color(m),
                       alpha=0.22, edgecolor="none", zorder=2, label=None)
            x_data = sub[col].to_numpy()
            y_data = sub["kge"].to_numpy()
            if log_x:
                m_pos = x_data > 0
                x_data = np.log10(x_data[m_pos])
                y_data = y_data[m_pos]
            z = _lowess(x_data, y_data, frac=0.45)
            if z is not None:
                # Clip LOWESS to the central 95% of data so the smoother does
                # not extrapolate wildly past the last observation.
                if len(x_data):
                    lo = float(np.nanquantile(x_data, 0.025))
                    hi = float(np.nanquantile(x_data, 0.975))
                    mask = (z[:, 0] >= lo) & (z[:, 0] <= hi)
                    z = z[mask]
                if z.size:
                    xs = z[:, 0]
                    if log_x:
                        xs = 10 ** xs
                    ax.plot(xs, z[:, 1], color=model_color(m), lw=1.9, zorder=4,
                            label=model_label(m), solid_capstyle="round")
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(sub[col], sub["kge"])
            except Exception:
                rho = np.nan
            rho_lines.append((model_label(m), rho))
        if log_x:
            ax.set_xscale("log")
        ax.axhline(0.5, color=RULE_COLOR, lw=0.5, ls=":", zorder=1)
        ax.set_xlabel(label)
        ax.set_ylabel("Test KGE")
        ax.set_ylim(-0.25, 1.0)
        _grid(ax, axis="both")
        # Spearman ρ box in the lower-right where points are sparse and curves
        # have already passed — does not collide with LOWESS data anymore.
        if rho_lines:
            text = "\n".join(f"{tag}: $\\rho$={rho:+.2f}" for tag, rho in rho_lines)
            _annot(ax, text, x=0.97, y=0.04, fontsize=6.6, ha="right", va="bottom",
                   bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98,
                             pad=2.2, boxstyle="round,pad=0.22"))
        _panel_label(ax, _panel_letter(k), x=-0.10, y=1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles=handles, labels=labels, loc="lower center",
                   ncol=len(models), bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1.0))
    _save(fig, fig_dir / "fig05_skill_vs_attribute.png")


# ─── Figure 6. Twin-axis hydrographs with precipitation ──────────────────────

def plot_hydrographs(hydrographs: pd.DataFrame, examples: pd.DataFrame, raw: pd.DataFrame, fig_dir: Path):
    if hydrographs.empty or examples.empty:
        return
    test = hydrographs[hydrographs["split"].eq("test")].copy()
    if test.empty:
        test = hydrographs[hydrographs["split"].eq("val")].copy()
    if test.empty:
        return
    test["date"] = pd.to_datetime(test["date"])
    # Six basins gives a legible double-column paper figure. The larger atlas
    # below carries more examples.
    show = _balanced_examples(examples, n_per_zone=2, max_total=6)
    if show.empty:
        return
    cols = 2
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=PAPER_FIGSIZE,
                             squeeze=False)
    axes = axes.ravel()
    models = _present_models(test)
    letters = "abcdefghij"
    for i, (_, basin) in enumerate(show.iterrows()):
        ax = axes[i]
        bid = int(basin["ID"])
        sub = test[test["ID"].eq(bid)].sort_values("date")
        if sub.empty:
            ax.axis("off"); continue
        years = _peak_years(sub, n_years=2)
        if years:
            sub = sub[sub["date"].dt.year.isin(years)]
        _add_precip_axis(ax, raw, bid, sub["date"].min(), sub["date"].max(),
                         label=(i // cols == rows - 1), compact=True)
        # Observed: thicker dark line behind everything
        obs = sub.drop_duplicates(subset=["date"]).sort_values("date")
        ax.fill_between(obs["date"], obs["q_obs"], 0, color=OBS_COLOR,
                        alpha=0.06, lw=0, zorder=2)
        ax.plot(obs["date"], obs["q_obs"], color=OBS_COLOR, lw=1.25,
                label="Observed", zorder=4)
        # Per-model simulation
        for m in models:
            ms = sub[sub["model"].eq(m)]
            if ms.empty:
                continue
            ax.plot(ms["date"], ms["q_sim"], color=model_color(m), lw=0.95,
                    linestyle=_model_linestyle(m),
                    alpha=0.95, label=model_label(m), zorder=3)
        regime_zone = str(basin.get("regime_zone", "")).replace("_class", "")
        regime_val  = zone_value_pretty(str(basin.get("regime_value", "")))
        regime = f"{regime_zone}: {regime_val}" if regime_zone else regime_val
        # Per-model KGE on this window
        kge_lines = []
        for m in models:
            mvals = sub[sub["model"].eq(m)].dropna(subset=["q_obs", "q_sim"])
            if mvals.empty: continue
            kge = _kge_from_arrays(mvals["q_obs"].to_numpy(), mvals["q_sim"].to_numpy())
            if not np.isfinite(kge): continue
            kge_lines.append((m, kge))
        ax.set_title(f"({letters[i]}) ID {bid} · {regime}",
                     fontsize=8.0, loc="left", fontweight="semibold")
        if kge_lines:
            text = "\n".join(f"{model_label(m)} {kge:.2f}" for m, kge in kge_lines)
            _annot(ax, text, x=0.985, y=0.96, fontsize=5.9, ha="right", va="top",
                   bbox=dict(facecolor="white", edgecolor="0.85",
                             alpha=0.98, pad=1.7, boxstyle="round,pad=0.18"))
        if i % cols == 0:
            ax.set_ylabel("$Q$ (mm d$^{-1}$)")
        _grid(ax, axis="y")
        _format_date_axis(ax, span_years=2)
        ax.margins(x=0.005)
    for k in range(len(show), rows * cols):
        axes[k].axis("off")
    # Single shared legend at the very bottom — keeps all panels uncluttered.
    precip_handle = Patch(facecolor=PRECIP_COLOR, alpha=0.38, edgecolor="none",
                          label="Precipitation")
    handles = [Line2D([0], [0], color=OBS_COLOR, lw=1.4, label="Observed")]
    handles += _model_legend_handles(models, lw=1.8) + [precip_handle]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), fontsize=7.3,
               ncol=min(len(handles), 6), handlelength=1.6,
               columnspacing=1.4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, fig_dir / "fig06_hydrographs.png")


# ─── Figure 7. Flow-duration curves at example basins ────────────────────────

def plot_flow_duration_curves(hydrographs: pd.DataFrame, examples: pd.DataFrame, fig_dir: Path):
    if hydrographs.empty or examples.empty:
        return
    test = hydrographs[hydrographs["split"].eq("test")]
    if test.empty:
        test = hydrographs[hydrographs["split"].eq("val")]
    if test.empty:
        return
    show = _balanced_examples(examples, n_per_zone=2, max_total=6)
    cols = 3
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=PAPER_FIGSIZE,
                             squeeze=False, sharey=False)
    models = _present_models(test)
    letters = "abcdefghij"
    for k, (_, basin) in enumerate(show.iterrows()):
        ax = axes[k // cols, k % cols]
        bid = int(basin["ID"])
        sub = test[test["ID"].eq(bid)]
        if sub.empty:
            ax.axis("off"); continue
        obs = sub.drop_duplicates(subset=["date"])["q_obs"].dropna().to_numpy()
        if obs.size == 0:
            ax.axis("off"); continue
        obs_sorted = np.sort(obs)[::-1]
        ex_obs = (np.arange(1, len(obs_sorted) + 1) - 0.5) / len(obs_sorted) * 100
        # Shade the high-flow (<5%) and low-flow (>95%) zones
        ax.axvspan(0, 5,  color=GRID_COLOR, alpha=0.55, zorder=0)
        ax.axvspan(95, 100, color=GRID_COLOR, alpha=0.55, zorder=0)
        ax.plot(ex_obs, np.maximum(obs_sorted, 1e-4), color=OBS_COLOR, lw=1.4,
                label="Observed", zorder=5)
        for m in models:
            sim = sub.loc[sub["model"].eq(m), "q_sim"].dropna().to_numpy()
            if sim.size == 0: continue
            sim_sorted = np.sort(sim)[::-1]
            ex_sim = (np.arange(1, len(sim_sorted) + 1) - 0.5) / len(sim_sorted) * 100
            ax.plot(ex_sim, np.maximum(sim_sorted, 1e-4), color=model_color(m),
                    lw=1.05, linestyle=_model_linestyle(m),
                    alpha=0.95, label=model_label(m))
        for ref_p, ref_lbl in [(5, "$Q_{05}$"), (95, "$Q_{95}$")]:
            ax.axvline(ref_p, color=RULE_COLOR, lw=0.45, ls=":", zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        if k // cols == rows - 1:
            ax.set_xlabel("Exceedance probability (%)")
        if k % cols == 0:
            ax.set_ylabel("$Q$ (mm d$^{-1}$, log)")
        regime_zone = str(basin.get("regime_zone", "")).replace("_class", "")
        regime_val  = zone_value_pretty(str(basin.get("regime_value", "")))
        # Two-line title so the regime label never overflows into the
        # neighbouring panel's letter at narrow column widths.
        ax.set_title(f"ID {bid}\n{regime_zone}: {regime_val}",
                     fontsize=8.0, fontweight="semibold", loc="left",
                     linespacing=1.05, pad=4)
        _grid(ax, axis="both")
        _panel_label(ax, _panel_letter(k), x=-0.14, y=1.04)
    for k in range(len(show), rows * cols):
        axes[k // cols, k % cols].axis("off")
    handles = [Line2D([0], [0], color=OBS_COLOR, lw=1.4, label="Observed")]
    handles += _model_legend_handles(models, lw=1.6)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.99), h_pad=1.6)
    _save(fig, fig_dir / "fig07_flow_duration_curves.png")


# ─── Figure 8. Signature obs vs sim with R², 1:1 line, ±20% bands ────────────

def plot_signature_obs_vs_sim(sig_long: pd.DataFrame, fig_dir: Path):
    if sig_long.empty:
        return
    split = _pick_eval_split(sig_long)
    if split is None:
        return
    sub = sig_long[sig_long["split"].eq(split)].copy()
    if sub.empty:
        return
    models = _present_models(sub)
    if not models:
        return
    plot_sigs = ["q_mean", "runoff_ratio", "fdc_slope", "baseflow_index",
                 "q05", "q95"]
    plot_sigs = [s for s in plot_sigs if s in sub["signature"].unique()]
    log_sigs = {"q_mean", "q05", "q95"}
    cols = 3
    rows = int(np.ceil(len(plot_sigs) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=PAPER_FIGSIZE,
                             squeeze=False)
    letters = "abcdefghij"
    for k, sig_name in enumerate(plot_sigs):
        ax = axes[k // cols, k % cols]
        chunk = sub[sub["signature"].eq(sig_name)].dropna(subset=["obs", "sim"])
        if chunk.empty:
            ax.axis("off"); continue
        all_v = pd.concat([chunk["obs"], chunk["sim"]])
        lo = float(np.nanpercentile(all_v, 1))
        hi = float(np.nanpercentile(all_v, 99))
        if hi <= lo:
            hi = lo + 1
        if sig_name in log_sigs:
            lo = max(lo, 1e-3)
            chunk = chunk[(chunk["obs"] > 0) & (chunk["sim"] > 0)]
        # ±20% envelope band (relative for log sigs, absolute for the rest)
        xs_line = np.linspace(lo, hi, 200)
        if sig_name in log_sigs:
            ax.fill_between(xs_line, xs_line * 0.8, xs_line * 1.2,
                            color=GRID_COLOR, alpha=0.7, zorder=1, lw=0)
        else:
            band = 0.2 * (hi - lo)
            ax.fill_between(xs_line, xs_line - band, xs_line + band,
                            color=GRID_COLOR, alpha=0.7, zorder=1, lw=0)
        rho_lines: list[tuple[str, str, float]] = []
        for m in models:
            mod = chunk[chunk["model"].eq(m)]
            if mod.empty:
                continue
            ax.scatter(mod["obs"], mod["sim"], s=8, color=model_color(m),
                       alpha=0.45, edgecolor="none", zorder=3,
                       label=model_label(m))
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(mod["obs"], mod["sim"])
            except Exception:
                rho = np.nan
            rho_lines.append((model_label(m), model_color(m), rho))
        ax.plot([lo, hi], [lo, hi], color=AXIS_COLOR, lw=0.7, ls="--", zorder=2)
        if sig_name in log_sigs:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        pretty = signature_pretty(sig_name)
        ax.set_xlabel(f"observed {pretty}")
        ax.set_ylabel(f"simulated {pretty}")
        _grid(ax, axis="both")
        _panel_label(ax, _panel_letter(k), x=-0.18, y=1.02)
        # Spearman ρ table — lower-right, neutral colour, single tight box.
        if rho_lines:
            text = "\n".join(f"{tag}: $\\rho$={rho:+.2f}" for tag, _, rho in rho_lines)
            _annot(ax, text, x=0.97, y=0.04, fontsize=6.2,
                   ha="right", va="bottom",
                   bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98,
                             pad=2.0, boxstyle="round,pad=0.20"))
    for k in range(len(plot_sigs), rows * cols):
        axes[k // cols, k % cols].axis("off")
    # Figure-level legend at the bottom — replaces the panel-0 inset that used
    # to collide with the ρ statistics box.
    legend_handles = [
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor=model_color(m), markeredgecolor="white",
               markersize=6, lw=0, label=model_label(m))
        for m in models
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(models), bbox_to_anchor=(0.5, -0.005),
               frameon=False, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.035, 1, 1.0))
    _save(fig, fig_dir / "fig08_signature_obs_vs_sim.png")


# ─── Figure 9. Storm-event zoom ──────────────────────────────────────────────

def plot_storm_zoom(hydrographs: pd.DataFrame, examples: pd.DataFrame, raw: pd.DataFrame, fig_dir: Path):
    if hydrographs.empty or examples.empty:
        return
    test = hydrographs[hydrographs["split"].eq("test")].copy()
    if test.empty:
        return
    test["date"] = pd.to_datetime(test["date"])
    show = _balanced_examples(examples, n_per_zone=1, max_total=4)
    if show.empty:
        return
    cols = 2
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=PAPER_FIGSIZE,
                             squeeze=False)
    axes = axes.ravel()
    models = _present_models(test)
    letters = "abcdefgh"
    for i, (_, basin) in enumerate(show.iterrows()):
        ax = axes[i]
        bid = int(basin["ID"])
        sub = test[test["ID"].eq(bid)].sort_values("date")
        if sub.empty:
            ax.axis("off"); continue
        obs = sub.drop_duplicates(subset=["date"]).sort_values("date")
        if obs.empty:
            ax.axis("off"); continue
        peak_date = obs.loc[obs["q_obs"].idxmax(), "date"]
        d0 = peak_date - pd.Timedelta(days=60)
        d1 = peak_date + pd.Timedelta(days=60)
        zoom = sub[(sub["date"] >= d0) & (sub["date"] <= d1)]
        zobs = obs[(obs["date"] >= d0) & (obs["date"] <= d1)]
        if zobs.empty: ax.axis("off"); continue
        _add_precip_axis(ax, raw, bid, d0, d1, label=(i // cols == rows - 1),
                         compact=True)
        # Mark peak day
        ax.axvline(peak_date, color=RULE_COLOR, lw=0.55, ls=":", zorder=1)
        ax.fill_between(zobs["date"], zobs["q_obs"], 0, color=OBS_COLOR,
                        alpha=0.06, lw=0, zorder=2)
        ax.plot(zobs["date"], zobs["q_obs"], color=OBS_COLOR, lw=1.3,
                label="Observed", zorder=4)
        for m in models:
            ms = zoom[zoom["model"].eq(m)]
            if ms.empty: continue
            ax.plot(ms["date"], ms["q_sim"], color=model_color(m), lw=1.05,
                    linestyle=_model_linestyle(m),
                    alpha=0.95, label=model_label(m), zorder=3)
        if i % cols == 0:
            ax.set_ylabel("$Q$ (mm d$^{-1}$)")
        regime_zone = str(basin.get("regime_zone", "")).replace("_class", "")
        regime_val  = zone_value_pretty(str(basin.get("regime_value", "")))
        ax.set_title(
            f"({letters[i]})  ID {bid} — {peak_date.strftime('%b %Y')} peak\n"
            f"{regime_zone}: {regime_val}",
            fontsize=8.5, loc="left", fontweight="semibold",
            linespacing=1.05, pad=4)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
        _grid(ax, axis="y")
        ax.margins(x=0.005)
    for k in range(len(show), rows * cols):
        axes[k].axis("off")
    # Shared legend at the bottom — keeps every panel's title clear.
    precip_handle = Patch(facecolor=PRECIP_COLOR, alpha=0.38, edgecolor="none",
                          label="Precipitation")
    handles = [Line2D([0], [0], color=OBS_COLOR, lw=1.4, label="Observed")]
    handles += _model_legend_handles(models, lw=1.8) + [precip_handle]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), fontsize=7.3,
               ncol=min(len(handles), 6), handlelength=1.6,
               columnspacing=1.4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, fig_dir / "fig09_storm_event_zoom.png")


# ─── Figure 10. Top feature importances ──────────────────────────────────────

def plot_feature_importance(imp: pd.DataFrame, fig_dir: Path):
    if imp.empty:
        return
    models = [m for m in ["random_forest", "xgboost"] if m in imp["model"].astype(str).unique()]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models),
                             figsize=PAPER_FIGSIZE, squeeze=False)
    letters = "abcd"
    for i, m in enumerate(models):
        ax = axes[0, i]
        sub = imp[imp["model"].eq(m)].head(12).iloc[::-1]
        bars = ax.barh(sub["feature"], sub["importance"], color=model_color(m),
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(model_label(m), fontsize=9.5, fontweight="semibold")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=7.5)
        # Inline value at the end of each bar
        xmax = float(sub["importance"].max()) if len(sub) else 1.0
        for bar, v in zip(bars, sub["importance"].to_numpy()):
            ax.text(v + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=6.8, color=ANNOT_COLOR)
        ax.set_xlim(0, xmax * 1.18)
        _grid(ax, axis="x")
        _panel_label(ax, _panel_letter(i), x=-0.32, y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig10_feature_importance.png")


# ─── Figure 11. Signature winners ────────────────────────────────────────────

def plot_signature_winners(ranks: pd.DataFrame, fig_dir: Path):
    if ranks.empty:
        return
    split = _pick_eval_split(ranks)
    if split is None:
        return
    test = ranks[(ranks["split"].eq(split)) & (ranks["rank"].eq(1))].copy()
    if test.empty:
        return
    models = _present_models(test)
    counts = test["model"].value_counts().reindex(models, fill_value=0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    bars = ax.bar([model_label(m) for m in models], counts.values,
                  color=palette_for(models), edgecolor="white", linewidth=0.5,
                  zorder=3)
    ax.set_ylabel("Number of signatures best preserved")
    for i, v in enumerate(counts.values):
        pct = 100 * v / total if total else 0
        ax.text(i, v + 0.05, f"{int(v)}\n({pct:.0f}%)", ha="center", va="bottom",
                fontsize=8.0, color=ANNOT_COLOR)
    if len(counts):
        ax.set_ylim(0, max(counts.values) + 1.6)
    _grid(ax, axis="y")
    fig.tight_layout()
    _save(fig, fig_dir / "fig11_signature_winners.png")


# ─── Figure 12. Human-impact PBIAS with ±10 % reference band ─────────────────

def plot_human_impact_pbias(skill_attrs: pd.DataFrame, fig_dir: Path):
    if skill_attrs.empty or "impact_class" not in skill_attrs.columns:
        return
    models = _present_models(skill_attrs)
    if not models:
        return
    order = [z for z in ZONE_VALUE_ORDER["impact_class"]
             if z in skill_attrs["impact_class"].astype(str).unique()
             and z != "impact_unknown"]
    if not order:
        return
    width = 0.78 / max(len(models), 1)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    # ±10 % reference band (HESS-standard "good fit" envelope)
    ax.axhspan(-10, 10, color=GRID_COLOR, alpha=0.55, zorder=1, lw=0)
    ax.axhline(0, color=AXIS_COLOR, lw=0.6, zorder=2)
    ax.axhline(10, color=RULE_COLOR, lw=0.4, ls=":", zorder=2)
    ax.axhline(-10, color=RULE_COLOR, lw=0.4, ls=":", zorder=2)
    for i, m in enumerate(models):
        meds = []
        for z in order:
            v = pd.to_numeric(
                skill_attrs.loc[(skill_attrs["model"].eq(m)) & (skill_attrs["impact_class"].astype(str).eq(z)), "pbias"],
                errors="coerce",
            ).dropna()
            meds.append(v.median() if len(v) else np.nan)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, meds, width=width, color=model_color(m),
               edgecolor="white", linewidth=0.5, label=model_label(m), zorder=3)
    labels = []
    for z in order:
        n_bas = int(skill_attrs.loc[(skill_attrs["model"].eq(models[0])) & (skill_attrs["impact_class"].astype(str).eq(z))].shape[0])
        labels.append(f"{zone_value_pretty(z)}\n$n$ = {n_bas}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median PBIAS (%)")
    # Legend includes ±10 % band
    band_handle = Patch(facecolor=GRID_COLOR, alpha=0.7, label="±10 % reference")
    handles = _model_legend_handles(models) + [band_handle]
    ax.legend(handles=handles, ncol=min(len(models) + 1, 5),
              loc="upper right",
              frameon=True, facecolor="white", edgecolor="0.85",
              framealpha=0.98).set_zorder(20)
    _grid(ax, axis="y")
    fig.tight_layout()
    _save(fig, fig_dir / "fig12_human_impact_pbias.png")


# ─── Figure 0. Study domain (basin distribution by regime) ───────────────────

def plot_study_domain(attrs_geo: pd.DataFrame, fig_dir: Path):
    if attrs_geo.empty or "lon_deg" not in attrs_geo.columns:
        return
    panels = [
        ("snow_class",     "Snow regime"),
        ("aridity_class",  "Aridity regime"),
        ("impact_class",   "Human alteration"),
    ]
    panels = [(c, t) for (c, t) in panels if c in attrs_geo.columns]
    if not panels:
        return

    use_cartopy, proj, data_crs, feats = _init_cartopy()

    n_maps = len(panels)
    n_total = n_maps + 1  # plus a histogram panel
    rows = int(np.ceil(n_total / 2))
    cols = 2
    # Slightly taller than PAPER_FIGSIZE so the panel-letter row doesn't
    # collide with the gridline labels above.
    # Generous canvas — equal-aspect maps need room or they collapse to strips
    # when cartopy isn't available.
    fig = plt.figure(figsize=(COL_DOUBLE, 3.5 * rows + 0.3))
    gs = fig.add_gridspec(rows, cols, hspace=0.42, wspace=0.18)
    letters = "abcdefgh"

    for k, (col, title) in enumerate(panels):
        r, c = divmod(k, cols)
        if use_cartopy:
            ax = fig.add_subplot(gs[r, c], projection=proj)
            _add_basemap(ax, feats, extent=(7.5, 18.5, 45.0, 51.5),
                         data_crs=data_crs, gridlines=True, label_size=6.0,
                         label_left=(c == 0),
                         label_bottom=True)
        else:
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(LAND_COLOR)
            ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
            ax.set_aspect(1.5)
            ax.set_xlim(7.5, 18.5)
            ax.set_ylim(45.0, 51.5)

        palette = REGIME_PALETTES.get(col, {})
        order = ZONE_VALUE_ORDER.get(col, [])
        order = [v for v in order if v in attrs_geo[col].astype(str).unique()
                 and v != "unknown" and v != "impact_unknown"]
        legend_h = []
        for val in order:
            sub = attrs_geo[attrs_geo[col].astype(str).eq(val)]
            if sub.empty:
                continue
            color = palette.get(val, MUTED_COLOR)
            kw = dict(c=color, s=12, edgecolor="white", linewidth=0.2,
                      zorder=4, label=ZONE_VALUE_PRETTY.get(val, val))
            if use_cartopy:
                kw["transform"] = data_crs
            ax.scatter(sub["lon_deg"], sub["lat_deg"], **kw)
            legend_h.append(Line2D([0], [0], marker="o", linestyle="",
                                    markerfacecolor=color, markeredgecolor="white",
                                    markersize=6,
                                    label=f"{ZONE_VALUE_PRETTY.get(val, val)} ($n$={len(sub)})"))
        # Combined panel-letter + title — one anchor, leaves room for top
        # gridline labels without overlapping them.
        ax.set_title(f"({letters[k]})  {title}", fontsize=9.0,
                     fontweight="semibold", loc="left", pad=4)
        ax.legend(handles=legend_h, loc="lower left", fontsize=6.5,
                  frameon=True, facecolor="white", edgecolor="0.85",
                  framealpha=0.98)

    # Final panel — catchment area distribution (log)
    r, c = divmod(n_maps, cols)
    ax_h = fig.add_subplot(gs[r, c])
    areas = pd.to_numeric(attrs_geo.get("area_calc"), errors="coerce").dropna()
    p_mean = pd.to_numeric(attrs_geo.get("p_mean"), errors="coerce").dropna()
    if len(areas):
        ax_h.hist(areas, bins=np.logspace(np.log10(max(areas.min(), 0.1)),
                                          np.log10(areas.max()), 32),
                  color=SLATE_COLOR, edgecolor="white", linewidth=0.5, alpha=0.95)
        ax_h.set_xscale("log")
        ax_h.set_xlabel("Catchment area (km²)")
        ax_h.set_ylabel("Number of catchments")
        ax_h.set_title(f"({letters[n_maps]})  Catchment-area distribution",
                       fontsize=9.0, fontweight="semibold", loc="left", pad=4)
        med = areas.median()
        ax_h.axvline(med, color=HESS_RED, lw=1.0, ls="--", zorder=4)
        ax_h.text(med * 1.05, ax_h.get_ylim()[1] * 0.92, f"median = {med:.0f} km²",
                  color=HESS_RED, fontsize=7.0, ha="left", va="top")
    _grid(ax_h, axis="y")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.07,
                        hspace=0.42, wspace=0.18)
    _save(fig, fig_dir / "fig00_study_domain.png")


# ─── Figure 13. Hydrograph mosaic — many basins, compact ─────────────────────

def plot_hydrograph_mosaic(hydrographs: pd.DataFrame, examples: pd.DataFrame,
                           raw: pd.DataFrame, fig_dir: Path):
    if hydrographs.empty or examples.empty:
        return
    test = hydrographs[hydrographs["split"].eq("test")].copy()
    if test.empty:
        test = hydrographs[hydrographs["split"].eq("val")].copy()
    if test.empty:
        return
    test["date"] = pd.to_datetime(test["date"])
    # 9-panel 3×3 mosaic (was 12-panel 4×3 — too cramped). Wider columns let the
    # subtitles render without colliding with the neighbour's panel title.
    show = _balanced_examples(examples, n_per_zone=2, max_total=9)
    if show.empty:
        return
    cols = 3
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=PAPER_FIGSIZE,
                             squeeze=False)
    models = _present_models(test)
    letters = "abcdefghijklmnop"
    for k, (_, basin) in enumerate(show.iterrows()):
        ax = axes[k // cols, k % cols]
        bid = int(basin["ID"])
        sub = test[test["ID"].eq(bid)].sort_values("date")
        if sub.empty:
            ax.axis("off"); continue
        years = _peak_years(sub, n_years=1)
        if not years:
            ax.axis("off"); continue
        sub = sub[sub["date"].dt.year.eq(years[0])]
        if sub.empty:
            ax.axis("off"); continue
        _add_precip_axis(ax, raw, bid, sub["date"].min(), sub["date"].max(),
                         label=False, compact=True)
        obs = sub.drop_duplicates(subset=["date"]).sort_values("date")
        ax.fill_between(obs["date"], obs["q_obs"], 0, color=OBS_COLOR,
                        alpha=0.06, lw=0, zorder=2)
        ax.plot(obs["date"], obs["q_obs"], color=OBS_COLOR, lw=1.0,
                label="Observed", zorder=4)
        for m in models:
            ms = sub[sub["model"].eq(m)]
            if ms.empty:
                continue
            ax.plot(ms["date"], ms["q_sim"], color=model_color(m), lw=0.85,
                    linestyle=_model_linestyle(m),
                    alpha=0.95, zorder=3)
        regime_val  = zone_value_pretty(str(basin.get("regime_value", "")))
        # Two-line title: bold panel letter + ID on line 1, regime + year on
        # line 2. Stays inside the panel and never wraps into a neighbour.
        ax.set_title(f"({letters[k]}) ID {bid}\n{regime_val} · {years[0]}",
                     fontsize=8.0, loc="left", fontweight="semibold",
                     linespacing=1.05, pad=4)
        ax.set_ylabel("$Q$ (mm d$^{-1}$)", fontsize=7.2)
        ax.tick_params(axis="x", labelsize=6.6)
        ax.tick_params(axis="y", labelsize=6.6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.margins(x=0.01)
        _grid(ax, axis="y")
    for k in range(len(show), rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.legend(handles=[Line2D([0], [0], color=OBS_COLOR, lw=1.4, label="Observed")] +
               _model_legend_handles(models),
               loc="lower center", ncol=len(models) + 1,
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    _save(fig, fig_dir / "fig13_hydrograph_mosaic.png")


# ─── Figure 14. KGE–PBIAS bivariate skill landscape ──────────────────────────

def plot_kge_pbias_landscape(metrics: pd.DataFrame, fig_dir: Path):
    if metrics.empty or "kge" not in metrics.columns or "pbias" not in metrics.columns:
        return
    split = _pick_eval_split(metrics)
    if split is None:
        return
    sub = metrics[metrics["split"].eq(split)].copy()
    models = _present_models(sub)
    if not models:
        return
    n_models = len(models)
    if n_models <= 3:
        cols, rows = n_models, 1
    elif n_models == 4:
        cols, rows = 4, 1
    else:
        cols = 3
        rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(COL_DOUBLE, 2.65 * rows + 0.3),
                             squeeze=False, sharex=True, sharey=True)
    letters = "abcdefgh"
    for k, m in enumerate(models):
        ax = axes[k // cols, k % cols]
        d = sub[sub["model"].eq(m)].dropna(subset=["kge", "pbias"])
        if d.empty:
            ax.axis("off"); continue
        # ±10 % PBIAS reference band
        ax.axhspan(-10, 10, color=GRID_COLOR, alpha=0.55, zorder=1, lw=0)
        ax.axhline(0, color=AXIS_COLOR, lw=0.6, zorder=2)
        ax.axvline(0.5, color=RULE_COLOR, lw=0.6, ls=":", zorder=2)
        # Plain scatter cloud — keeps the figure flat-coloured and consistent
        # with the rest of the manuscript (no white→colour gauge gradients).
        kge = d["kge"].clip(-1.0, 1.0).to_numpy()
        pb  = d["pbias"].clip(-60, 60).to_numpy()
        ax.scatter(kge, pb, s=8, color=model_color(m), alpha=0.45,
                   edgecolor="none", zorder=3)
        # Median crosshair
        kmed, pmed = float(np.median(kge)), float(np.median(pb))
        ax.scatter([kmed], [pmed], s=42, marker="P", color=model_color(m),
                   edgecolor="white", linewidth=0.8, zorder=6,
                   label=f"median ({kmed:.2f}, {pmed:.1f}%)")
        # Quadrant counts
        n_total = len(d)
        good = ((d["kge"] >= 0.5) & (d["pbias"].between(-10, 10))).sum()
        ax.text(0.97, 0.02, f"$n$ = {n_total}  ·  KGE ≥ 0.5 & |PBIAS| ≤ 10 %: {good}\n"
                            f"({100 * good / n_total:.0f} % of basins)",
                transform=ax.transAxes, fontsize=6.6, ha="right", va="bottom",
                color=ANNOT_COLOR,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98,
                          pad=2.2, boxstyle="round,pad=0.25"))
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-50, 50)
        ax.set_title(f"({letters[k]})  {model_label(m)}",
                     fontsize=9.0, fontweight="semibold", loc="left", pad=4)
        if k % cols == 0:
            ax.set_ylabel("PBIAS (%)")
        below = k + cols
        is_bottom = (k // cols == rows - 1) or (below >= len(models))
        if is_bottom:
            ax.set_xlabel("KGE")
            ax.tick_params(labelbottom=True)
        _grid(ax, axis="both")
        leg = ax.legend(loc="upper left", fontsize=6.5,
                        frameon=True, facecolor="white",
                        edgecolor="0.85", framealpha=0.98)
        leg.set_zorder(20)
    # Hide any leftover slots
    for k_extra in range(len(models), rows * cols):
        axes[k_extra // cols, k_extra % cols].axis("off")
    fig.tight_layout(h_pad=1.6)
    _save(fig, fig_dir / "fig14_kge_pbias_landscape.png")


# ─── Figure 15. Regime × model KGE heatmap ───────────────────────────────────

def plot_regime_kge_heatmap(skill_attrs: pd.DataFrame, fig_dir: Path):
    if skill_attrs.empty:
        return
    models = _present_models(skill_attrs)
    if not models:
        return
    zones = [z for z in ["snow_class", "aridity_class", "storage_class", "impact_class"]
             if z in skill_attrs.columns]
    if not zones:
        return
    rows_meta = []  # (zone_label, value, value_pretty)
    for z in zones:
        order = [v for v in ZONE_VALUE_ORDER.get(z, [])
                 if v in skill_attrs[z].astype(str).unique()
                 and v not in ("unknown", "impact_unknown")]
        for v in order:
            rows_meta.append((ZONE_LABELS.get(z, z), v,
                              ZONE_VALUE_PRETTY.get(v, v), z))

    if not rows_meta:
        return
    grid = np.full((len(rows_meta), len(models)), np.nan)
    counts = np.zeros((len(rows_meta), len(models)), dtype=int)
    for r, (_, val, _, z) in enumerate(rows_meta):
        for c, m in enumerate(models):
            d = pd.to_numeric(
                skill_attrs.loc[(skill_attrs["model"].eq(m)) & (skill_attrs[z].astype(str).eq(val)), "kge"],
                errors="coerce",
            ).dropna()
            if len(d):
                grid[r, c] = float(d.median())
                counts[r, c] = int(len(d))

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.5, vmax=0.9)
    im = ax.imshow(grid, cmap=SKILL_CMAP, norm=norm, aspect="auto")
    # Cell annotations: median KGE
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            if not np.isfinite(v):
                ax.text(c, r, "—", ha="center", va="center", fontsize=7.5, color="0.4")
                continue
            txtcolor = "white" if abs(v - 0.5) > 0.28 else "#111111"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=txtcolor, fontweight="semibold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models], rotation=20, ha="right")
    # Row labels with regime zone groupings + count
    rowlabels = []
    last_zone = None
    for (zlbl, val, vpretty, z), row_counts in zip(rows_meta, counts):
        n_max = int(row_counts.max()) if row_counts.size else 0
        prefix = f"{zlbl}  ·  " if zlbl != last_zone else " " * (len(zlbl) + 4)
        rowlabels.append(f"{prefix}{vpretty}  ($n$={n_max})")
        last_zone = zlbl
    ax.set_yticks(range(len(rows_meta)))
    ax.set_yticklabels(rowlabels, fontsize=7.5)
    # Faint horizontal separators between regime groups
    prev_zone = None
    for r, (zlbl, *_rest) in enumerate(rows_meta):
        if prev_zone is not None and zlbl != prev_zone:
            ax.axhline(r - 0.5, color="white", lw=2.0, zorder=4)
            ax.axhline(r - 0.5, color="0.55", lw=0.5, zorder=5)
        prev_zone = zlbl
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cbar.set_label("Median test KGE", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout()
    _save(fig, fig_dir / "fig15_regime_kge_heatmap.png")


# ─── Figure 16. Runoff coefficient (Q / P) — observed vs simulated ───────────

def plot_runoff_coefficient(sig_long: pd.DataFrame, attrs_geo: pd.DataFrame, fig_dir: Path):
    if sig_long.empty or attrs_geo.empty or "p_mean" not in attrs_geo.columns:
        return
    split = _pick_eval_split(sig_long)
    if split is None:
        return
    sub = sig_long[(sig_long["split"].eq(split)) & (sig_long["signature"].eq("q_mean"))].copy()
    if sub.empty:
        return
    p = attrs_geo[["ID", "p_mean"]].copy()
    p["p_mean"] = pd.to_numeric(p["p_mean"], errors="coerce")
    sub = sub.merge(p, on="ID", how="left").dropna(subset=["p_mean", "obs", "sim"])
    if sub.empty:
        return
    sub = sub[sub["p_mean"] > 0]
    sub["rc_obs"] = sub["obs"] / sub["p_mean"]
    sub["rc_sim"] = sub["sim"] / sub["p_mean"]
    models = _present_models(sub)
    if not models:
        return
    n_models = len(models)
    if n_models <= 3:
        cols, rows = n_models, 1
    elif n_models == 4:
        cols, rows = 4, 1
    else:
        cols = 3
        rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(COL_DOUBLE, 2.55 * rows + 0.3),
                             squeeze=False, sharex=True, sharey=True)
    for k_extra in range(len(models), rows * cols):
        axes[k_extra // cols, k_extra % cols].axis("off")
    lo = 0.0
    hi = float(np.nanpercentile(pd.concat([sub["rc_obs"], sub["rc_sim"]]), 99))
    hi = max(hi, 1.0) * 1.05
    letters = "abcdefgh"
    for k, m in enumerate(models):
        ax = axes[k // cols, k % cols]
        d = sub[sub["model"].eq(m)]
        # ±20 % envelope
        xs = np.linspace(lo, hi, 200)
        ax.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID_COLOR, alpha=0.7, lw=0, zorder=1)
        ax.plot([lo, hi], [lo, hi], color=AXIS_COLOR, lw=0.7, ls="--", zorder=2)
        ax.scatter(d["rc_obs"], d["rc_sim"], s=10, color=model_color(m),
                   alpha=0.55, edgecolor="none", zorder=3)
        # Spearman ρ + median bias on Q/P
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(d["rc_obs"], d["rc_sim"])
        except Exception:
            rho = float("nan")
        med_bias = float((d["rc_sim"] - d["rc_obs"]).median())
        ax.text(0.04, 0.96,
                f"$\\rho$ = {rho:.2f}\nmedian $\\Delta$ = {med_bias:+.03f}\n$n$ = {len(d)}",
                transform=ax.transAxes, fontsize=7.2, ha="left", va="top",
                color=ANNOT_COLOR,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.98,
                          pad=2.5, boxstyle="round,pad=0.25"))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        # x-label on every panel whose column has no panel below it (so the
        # 5-on-6-grid case still labels the lone bottom panels in row 0).
        below = k + cols
        is_bottom = (k // cols == rows - 1) or (below >= len(models))
        if is_bottom:
            ax.set_xlabel("observed runoff ratio  $Q_{obs}/P$")
            ax.tick_params(labelbottom=True)
        if k % cols == 0:
            ax.set_ylabel("simulated  $Q_{sim}/P$")
        ax.set_title(f"({letters[k]})  {model_label(m)}",
                     fontsize=9.0, fontweight="semibold", loc="left", pad=4)
        _grid(ax, axis="both")
    fig.tight_layout(h_pad=1.6)
    _save(fig, fig_dir / "fig16_runoff_coefficient.png")


# ─── Figure 17. Rainfall-runoff atlas across regimes ────────────────────────

def _render_hydrograph_atlas_part(show: pd.DataFrame, test: pd.DataFrame,
                                  raw: pd.DataFrame, models: list[str],
                                  fig_path: Path, letters_offset: int = 0,
                                  panels_total: int = 16):
    """Render an 8-panel atlas page (4 rows × 2 cols). Called twice — atlas A and
    atlas B — so each panel has enough horizontal width for a single-line title."""
    if show.empty:
        return
    cols = 2
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=PAPER_FIGSIZE,
                             squeeze=False)
    letters = "abcdefghijklmnop"
    for k, (_, basin) in enumerate(show.iterrows()):
        ax = axes[k // cols, k % cols]
        bid = int(basin["ID"])
        sub = test[test["ID"].eq(bid)].sort_values("date")
        if sub.empty:
            ax.axis("off"); continue
        years = _peak_years(sub, n_years=1)
        if not years:
            ax.axis("off"); continue
        sub = sub[sub["date"].dt.year.eq(years[0])]
        if sub.empty:
            ax.axis("off"); continue
        _add_precip_axis(ax, raw, bid, sub["date"].min(), sub["date"].max(),
                         label=False, compact=True)
        obs = sub.drop_duplicates(subset=["date"]).sort_values("date")
        ax.plot(obs["date"], obs["q_obs"], color=OBS_COLOR, lw=1.05, zorder=5)
        ax.fill_between(obs["date"], obs["q_obs"], 0, color=OBS_COLOR,
                        alpha=0.055, lw=0, zorder=2)
        for m in models:
            ms = sub[sub["model"].eq(m)]
            if ms.empty:
                continue
            ax.plot(ms["date"], ms["q_sim"], color=model_color(m), lw=0.85,
                    linestyle=_model_linestyle(m), alpha=0.95, zorder=4)
        ymax = pd.concat([
            pd.to_numeric(obs["q_obs"], errors="coerce"),
            pd.to_numeric(sub["q_sim"], errors="coerce"),
        ]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ymax):
            ax.set_ylim(0, max(float(np.nanpercentile(ymax, 99.5)) * 1.18, 0.05))
        regime_val = zone_value_pretty(str(basin.get("regime_value", "")))
        letter_idx = letters_offset + k
        if letter_idx < len(letters):
            letter = letters[letter_idx]
        else:
            letter = "x"
        # Two-line title prevents collisions between adjacent panel headers
        # at narrow column widths.
        ax.set_title(f"({letter}) ID {bid}\n{regime_val} · {years[0]}",
                     fontsize=7.8, loc="left", fontweight="semibold",
                     linespacing=1.05, pad=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="both", labelsize=6.6)
        if k % cols == 0:
            ax.set_ylabel("$Q$ (mm d$^{-1}$)", fontsize=7.2)
        if k // cols == rows - 1:
            ax.set_xlabel("Month", fontsize=7.2)
        _grid(ax, axis="y")
        ax.margins(x=0.01)
    for k in range(len(show), rows * cols):
        axes[k // cols, k % cols].axis("off")
    precip_handle = Patch(facecolor=PRECIP_COLOR, alpha=0.36, edgecolor="none",
                          label="Precipitation")
    handles = [Line2D([0], [0], color=OBS_COLOR, lw=1.4, label="Observed")]
    handles += _model_legend_handles(models, lw=1.7) + [precip_handle]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.004), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    _save(fig, fig_path)


def plot_hydrograph_atlas(hydrographs: pd.DataFrame, examples: pd.DataFrame,
                          raw: pd.DataFrame, fig_dir: Path):
    """Two-part hydrograph atlas — fig17a (process zones A) + fig17b (zones B).

    Splitting 16 panels into 2×(4×2) instead of one 4×4 mosaic gives each panel
    twice the horizontal width, so single-line titles render without overlap.
    """
    if hydrographs.empty or examples.empty:
        return
    test = hydrographs[hydrographs["split"].eq("test")].copy()
    if test.empty:
        test = hydrographs[hydrographs["split"].eq("val")].copy()
    if test.empty:
        return
    test["date"] = pd.to_datetime(test["date"])
    show = _balanced_examples(examples, n_per_zone=4, max_total=16).reset_index(drop=True)
    if show.empty:
        return
    models = _present_models(test)
    half = (len(show) + 1) // 2
    show_a = show.iloc[:half]
    show_b = show.iloc[half:]
    _render_hydrograph_atlas_part(show_a, test, raw, models,
                                  fig_dir / "fig17a_hydrograph_atlas.png",
                                  letters_offset=0)
    if not show_b.empty:
        _render_hydrograph_atlas_part(show_b, test, raw, models,
                                      fig_dir / "fig17b_hydrograph_atlas.png",
                                      letters_offset=half)


# ─── Figure 18. Signed hydrological ratio bias ───────────────────────────────

def plot_signature_bias_ratios(sig_long: pd.DataFrame, fig_dir: Path):
    if sig_long.empty:
        return
    split = _pick_eval_split(sig_long)
    if split is None:
        return
    sub = sig_long[sig_long["split"].eq(split)].copy()
    models = _present_models(sub)
    if not models:
        return
    sigs = [
        ("q_mean", "mean Q"),
        ("runoff_ratio", "Q/P"),
        ("q05", "low-flow Q$_{05}$"),
        ("q95", "high-flow Q$_{95}$"),
        ("high_q_dur", "high-flow duration"),
        ("low_q_dur", "low-flow duration"),
    ]
    sigs = [(s, lab) for s, lab in sigs if s in set(sub["signature"].astype(str))]
    if not sigs:
        return
    rows = []
    for sig_name, label in sigs:
        chunk = sub[sub["signature"].eq(sig_name)].copy()
        chunk["obs"] = pd.to_numeric(chunk["obs"], errors="coerce")
        chunk["sim"] = pd.to_numeric(chunk["sim"], errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=["obs", "sim"])
        chunk = chunk[chunk["obs"].abs() > 1e-9]
        if chunk.empty:
            continue
        chunk["rel_bias"] = 100.0 * (chunk["sim"] - chunk["obs"]) / chunk["obs"].abs()
        chunk["signature_label"] = label
        rows.append(chunk[["model", "ID", "signature", "signature_label", "rel_bias"]])
    if not rows:
        return
    bias = pd.concat(rows, ignore_index=True)
    bias["rel_bias_clipped"] = bias["rel_bias"].clip(-120, 120)

    summary = []
    for sig_name, label in sigs:
        for m in models:
            v = bias.loc[(bias["signature"].eq(sig_name)) & (bias["model"].eq(m)), "rel_bias"]
            v = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not len(v):
                continue
            summary.append({
                "signature": sig_name,
                "signature_label": label,
                "model": m,
                "median": float(v.median()),
                "q25": float(v.quantile(0.25)),
                "q75": float(v.quantile(0.75)),
                "within20": float((v.abs() <= 20).mean() * 100),
                "n": int(len(v)),
            })
    stats = pd.DataFrame(summary)
    if stats.empty:
        return

    # Reserve a thin right-hand gutter so panel (a) and panel (b) end at the
    # same x-position despite (b)'s colorbar — keeps the figure width consistent.
    # constrained_layout was previously used; it conflicted with the bottom-
    # anchored model legend, which led the colorbar to be silently collapsed.
    fig = plt.figure(figsize=(COL_DOUBLE, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          width_ratios=[1.0, 0.030], hspace=0.55, wspace=0.04)
    ax = fig.add_subplot(gs[0, :])
    labels = [lab for _, lab in sigs]
    x = np.arange(len(labels))
    width = min(0.72 / max(len(models), 1), 0.18)
    ax.axhspan(-20, 20, color=GRID_COLOR, alpha=0.65, lw=0, zorder=0)
    ax.axhline(0, color=AXIS_COLOR, lw=0.7, zorder=1)
    ax.axhline(20, color=RULE_COLOR, lw=0.45, ls=":", zorder=1)
    ax.axhline(-20, color=RULE_COLOR, lw=0.45, ls=":", zorder=1)
    for i, m in enumerate(models):
        d = stats[stats["model"].eq(m)].set_index("signature")
        meds = []
        lows = []
        highs = []
        xpos = []
        for j, (sig_name, _label) in enumerate(sigs):
            if sig_name not in d.index:
                continue
            row = d.loc[sig_name]
            med = float(row["median"])
            meds.append(np.clip(med, -120, 120))
            lows.append(max(0.0, med - float(row["q25"])))
            highs.append(max(0.0, float(row["q75"]) - med))
            xpos.append(x[j] + (i - (len(models) - 1) / 2) * width)
        if not xpos:
            continue
        ax.errorbar(xpos, meds, yerr=[lows, highs], fmt="o",
                    color=model_color(m), ecolor=model_color(m),
                    elinewidth=1.0, capsize=2.0, markersize=4.2,
                    label=model_label(m), zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Signed relative bias (%)")
    ax.set_ylim(-110, 110)
    ax.set_title("(a) Basin-wise signature bias, median and IQR",
                 fontsize=9.4, fontweight="semibold", loc="left", pad=4)
    _grid(ax, axis="y")

    axh = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    heat = np.full((len(sigs), len(models)), np.nan)
    for r, (sig_name, _label) in enumerate(sigs):
        for c, m in enumerate(models):
            hit = stats[(stats["signature"].eq(sig_name)) & (stats["model"].eq(m))]
            if len(hit):
                heat[r, c] = float(hit["within20"].iloc[0])
    im = axh.imshow(heat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            v = heat[r, c]
            if not np.isfinite(v):
                axh.text(c, r, "-", ha="center", va="center", fontsize=7.3, color="0.35")
                continue
            txt = "white" if v > 62 else "#111111"
            axh.text(c, r, f"{v:.0f}%", ha="center", va="center",
                     fontsize=7.3, color=txt, fontweight="semibold")
    axh.set_xticks(range(len(models)))
    axh.set_xticklabels([model_label(m) for m in models], rotation=18, ha="right")
    axh.set_yticks(range(len(sigs)))
    axh.set_yticklabels(labels)
    axh.set_title("(b) Catchments within +/-20% signature bias",
                  fontsize=9.4, fontweight="semibold")
    axh.tick_params(axis="both", length=0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Basins (%)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.outline.set_linewidth(0.4)
    # Single shared model legend at the figure bottom — keeps it out of every
    # panel's data area and replaces the per-axis legend that previously
    # collided with panel (a)'s upper whiskers.  Bottom margin is generous so
    # the rotated x-tick labels of (b)'s heatmap don't kiss the legend.
    handles = _model_legend_handles(models, lw=2.4)
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=7.6)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.16,
                        hspace=0.65, wspace=0.04)
    _save(fig, fig_dir / "fig18_signature_bias_ratios.png")


# ─── Figure 19. Flood typology seasonality and climate fingerprint ───────────

def plot_flood_typology_seasonality(flood_events: pd.DataFrame, fig_dir: Path):
    if flood_events.empty or "event_type" not in flood_events.columns:
        return
    split = _pick_eval_split(flood_events)
    if split is None:
        return
    sub = flood_events[flood_events["split"].eq(split)].copy()
    if sub.empty:
        return
    sub["event_type"] = sub["event_type"].astype(str)
    sub["season"] = sub["season"].astype(str)
    for c in ["p7", "temp7", "swe_before", "swe_delta_7"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    event_types = [t for t in FLOOD_TYPE_ORDER if t in set(sub["event_type"])]
    if not event_types:
        event_types = list(sub["event_type"].value_counts().index)
    seasons = ["DJF", "MAM", "JJA", "SON"]

    fig, axes = plt.subplots(1, 2, figsize=PAPER_FIGSIZE,
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    counts = (
        sub.groupby(["season", "event_type"])["event_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=seasons, columns=event_types, fill_value=0)
    )
    bottom = np.zeros(len(seasons))
    for event_type in event_types:
        vals = counts[event_type].to_numpy(dtype=float)
        ax.bar(seasons, vals, bottom=bottom, color=_flood_color(event_type),
               edgecolor="white", linewidth=0.45, label=_flood_label(event_type),
               zorder=3)
        bottom += vals
    ax.set_ylabel("Observed flood events")
    ax.set_title("(a) Seasonal flood-type composition", fontsize=9.4,
                 fontweight="semibold")
    _grid(ax, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2,
              fontsize=6.7, frameon=False, handlelength=1.0, columnspacing=0.9)

    ax = axes[1]
    stats = (
        sub.groupby("event_type")
        .agg(
            events=("event_id", "nunique"),
            median_temp7=("temp7", "median"),
            median_p7=("p7", "median"),
            median_swe_before=("swe_before", "median"),
            median_swe_delta_7=("swe_delta_7", "median"),
        )
        .reindex(event_types)
        .dropna(subset=["median_temp7", "median_p7"], how="all")
    )
    if not stats.empty:
        max_n = max(float(stats["events"].max()), 1.0)
        # Generous right-hand headroom so every label can left-anchor at its
        # bubble and extend rightward without colliding with neighbours.
        x_min = float(stats["median_temp7"].min())
        x_max = float(stats["median_temp7"].max())
        y_min = float(stats["median_p7"].min())
        y_max = float(stats["median_p7"].max())
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.55 * x_span)
        ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.18 * y_span)
        # Spread y-offsets where two bubbles share a similar precipitation level
        # (otherwise labels stack on the same line).
        ordered = stats.sort_values("median_p7")
        recent_y = []
        y_offsets: dict[str, int] = {}
        for event_type, row in ordered.iterrows():
            yv = float(row["median_p7"])
            collisions = sum(1 for prev in recent_y if abs(prev - yv) < 0.06 * y_span)
            y_offsets[event_type] = 4 + 10 * collisions if collisions else 4
            recent_y.append(yv)
        for event_type, row in stats.iterrows():
            n = float(row["events"])
            size = 34 + 170 * np.sqrt(n / max_n)
            edge = HESS_RED if event_type in {"rain_on_snow", "snowmelt"} else "white"
            ax.scatter(row["median_temp7"], row["median_p7"], s=size,
                       color=_flood_color(event_type), edgecolor=edge,
                       linewidth=0.9, alpha=0.92, zorder=4)
            ax.annotate(_flood_label(event_type),
                        (row["median_temp7"], row["median_p7"]),
                        xytext=(6, y_offsets.get(event_type, 4)),
                        textcoords="offset points", fontsize=6.8,
                        ha="left", va="bottom", color=ANNOT_COLOR)
        ax.axvline(0, color=RULE_COLOR, lw=0.65, ls=":", zorder=1)
        ax.set_xlabel("Median 7-day temperature at peak (deg C)")
        ax.set_ylabel("Median 7-day precipitation at peak (mm)")
        ax.set_title("(b) Hydroclimate fingerprint of flood types", fontsize=9.4,
                     fontweight="semibold")
        # Footnote moved to figure-level caption space below panel (b) so it
        # never overlaps low-precipitation bubbles (snowmelt / storage release).
        ax.text(0.5, -0.30,
                "Bubble area scales with event count; red outline marks snow-transition types.",
                transform=ax.transAxes, fontsize=6.5, ha="center", va="top",
                color=ANNOT_COLOR)
        _grid(ax, axis="both")
    else:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, fig_dir / "fig19_flood_typology_seasonality.png")


# ─── Figure 20. Model performance by observed flood type ─────────────────────

def plot_flood_typology_model_skill(flood_model_summary: pd.DataFrame, fig_dir: Path):
    if flood_model_summary.empty or "event_type" not in flood_model_summary.columns:
        return
    split = _pick_eval_split(flood_model_summary)
    if split is None:
        return
    sub = flood_model_summary[flood_model_summary["split"].eq(split)].copy()
    if sub.empty:
        return
    models = _present_models(sub)
    if not models:
        return
    event_types = [t for t in FLOOD_TYPE_ORDER if t in set(sub["event_type"].astype(str))]
    event_types = [t for t in event_types if sub[sub["event_type"].eq(t)]["events"].max() >= 3]
    if not event_types:
        event_types = list(sub["event_type"].value_counts().index)
    x = np.arange(len(event_types))
    width = min(0.70 / max(len(models), 1), 0.18)

    fig, axes = plt.subplots(2, 1, figsize=PAPER_FIGSIZE, sharex=True)
    ax = axes[0]
    ax.axhline(1.0, color=AXIS_COLOR, lw=0.7, zorder=1)
    ax.axhspan(0.8, 1.2, color=GRID_COLOR, alpha=0.62, lw=0, zorder=0)
    for i, m in enumerate(models):
        d = sub[sub["model"].eq(m)].set_index("event_type")
        vals = [float(d.loc[t, "median_peak_ratio"]) if t in d.index else np.nan for t in event_types]
        xpos = x + (i - (len(models) - 1) / 2) * width
        ax.vlines(xpos, 1.0, vals, color=model_color(m), lw=1.4, alpha=0.75, zorder=3)
        ax.scatter(xpos, vals, s=30, color=model_color(m), edgecolor="white",
                   linewidth=0.5, label=model_label(m), zorder=4)
    ax.set_ylabel("Median peak ratio\n$Q_{sim,peak}/Q_{obs,peak}$")
    ax.set_ylim(0, 2.05)
    ax.set_title("(a) Flood-peak magnitude reproduction", fontsize=9.4,
                 fontweight="semibold", loc="left", pad=4)
    # Single shared legend for the figure — written at the bottom in the
    # _save block below; per-axis legend was overlapping the title text.
    _grid(ax, axis="y")

    ax = axes[1]
    for i, m in enumerate(models):
        d = sub[sub["model"].eq(m)].set_index("event_type")
        vals = [float(d.loc[t, "median_abs_peak_timing_error_days"]) if t in d.index else np.nan
                for t in event_types]
        xpos = x + (i - (len(models) - 1) / 2) * width
        ax.bar(xpos, vals, width=width * 0.88, color=model_color(m),
               edgecolor=AXIS_COLOR, linewidth=0.3, zorder=3)
    ax.set_ylabel("Median absolute peak\ntiming error (days)")
    ax.set_title("(b) Flood-peak timing error", fontsize=9.4,
                 fontweight="semibold", loc="left", pad=4)
    ax.set_xticks(x)
    labels = []
    for t in event_types:
        count = int(sub[sub["event_type"].eq(t)]["events"].max())
        labels.append(f"{_flood_label(t)}\n$n$={count}")
    ax.set_xticklabels(labels, rotation=18, ha="right")
    _grid(ax, axis="y")
    handles = _model_legend_handles(models, lw=2.4)
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=7.6)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.18,
                        hspace=0.35)
    _save(fig, fig_dir / "fig20_flood_typology_model_skill.png")


# ─── Figure 21. Snow-timing error (half-flow date) ───────────────────────────

def plot_snow_timing_error(sig_long: pd.DataFrame, attrs_geo: pd.DataFrame, fig_dir: Path):
    """Diagnose spring snowmelt-timing failures by model.

    Half-flow date (HFD) is the day of the water year by which 50% of the
    annual flow has been delivered. A negative bias means simulated flow is
    delivered too early, which in snow-influenced basins fingerprints a
    premature snowmelt response.
    """
    if sig_long.empty or attrs_geo.empty:
        return
    if "signature" not in sig_long.columns:
        return
    hfd = sig_long[(sig_long["signature"].astype(str).eq("hfd_mean")) &
                   (sig_long["split"].astype(str).eq("test"))].copy()
    if hfd.empty:
        return
    hfd["signed_err"] = pd.to_numeric(hfd["sim"], errors="coerce") - \
                        pd.to_numeric(hfd["obs"], errors="coerce")
    hfd = hfd.merge(attrs_geo[["ID", "frac_snow", "snow_class"]], on="ID", how="left")
    hfd = hfd.dropna(subset=["signed_err", "frac_snow"])
    if hfd.empty:
        return
    models = _present_models(hfd)
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=PAPER_FIGSIZE,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) Signed HFD error vs frac_snow with LOWESS
    ax = axes[0]
    ax.axhline(0.0, color=AXIS_COLOR, lw=0.7, zorder=2)
    for m in models:
        sub = hfd[hfd["model"].eq(m)]
        if sub.empty:
            continue
        ax.scatter(sub["frac_snow"], sub["signed_err"], s=7,
                   color=model_color(m), alpha=0.18, edgecolor="none", zorder=2)
        z = _lowess(sub["frac_snow"].to_numpy(), sub["signed_err"].to_numpy(), frac=0.45)
        if z is not None:
            ax.plot(z[:, 0], z[:, 1], color=model_color(m), lw=1.9,
                    label=model_label(m), zorder=4, solid_capstyle="round")
    ax.set_xlabel("Fraction of precipitation as snow")
    ax.set_ylabel("HFD error (sim − obs, days)")
    ax.set_ylim(-55, 55)
    _grid(ax, axis="both")
    _panel_label(ax, "a", x=-0.10, y=1.02)
    # Single shared legend at the figure bottom (was previously inside (a)
    # at 'lower left' and overlapped the x-axis label).

    # (b) Absolute HFD error by snow regime class — boxplot per model
    ax = axes[1]
    classes = [c for c in ZONE_VALUE_ORDER["snow_class"]
               if c in set(hfd["snow_class"].dropna().astype(str))]
    if not classes:
        classes = list(hfd["snow_class"].dropna().astype(str).unique())
    width = 0.78 / max(len(models), 1)
    x = np.arange(len(classes))
    for i, m in enumerate(models):
        data = []
        counts = []
        for cls in classes:
            d = hfd[(hfd["model"].eq(m)) & (hfd["snow_class"].astype(str).eq(cls))]
            arr = d["abs_err_hfd_mean"].dropna().to_numpy() if "abs_err_hfd_mean" in d.columns \
                  else np.abs(d["signed_err"].dropna().to_numpy())
            data.append(arr if arr.size else np.array([np.nan]))
            counts.append(len(arr))
        positions = x + (i - (len(models) - 1) / 2) * width
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False, whis=(10, 90),
                        medianprops=dict(color="white", lw=1.0))
        for patch in bp["boxes"]:
            patch.set_facecolor(model_color(m))
            patch.set_edgecolor(model_color(m))
            patch.set_alpha(0.85)
        for whisker in bp["whiskers"] + bp["caps"]:
            whisker.set_color(model_color(m))
            whisker.set_lw(0.55)
    ax.set_xticks(x)
    nlabels = []
    for cls in classes:
        n = int(hfd[hfd["snow_class"].astype(str).eq(cls)]["ID"].nunique())
        nlabels.append(f"{ZONE_VALUE_PRETTY.get(cls, cls)}\n$n$={n}")
    # Slight rotation prevents adjacent labels (e.g. "mixed-snow" /
    # "snow-influenced") from running into each other in the narrow panel.
    ax.set_xticklabels(nlabels, fontsize=7.2, rotation=12, ha="center")
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylabel("|HFD error| (days)")
    ax.set_ylim(0, None)
    _grid(ax, axis="y")
    _panel_label(ax, "b", x=-0.12, y=1.02)

    handles = _model_legend_handles(models, lw=2.4)
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=7.6)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.18,
                        wspace=0.22)
    _save(fig, fig_dir / "fig21_snow_timing_error.png")


# ─── Figure 22. Predictability-zones synthesis map ───────────────────────────

def plot_predictability_zones(skill_attrs: pd.DataFrame, attrs_geo: pd.DataFrame,
                              fig_dir: Path):
    """Map LamaH-CE basins into hydrological predictability zones.

    Hierarchical assignment using static catchment attributes already in the
    LamaH-CE database, intersected with per-basin best-model identity:
      1. human-altered          — `is_impacted == 1` and `impact_class` ∈ {impact_m, impact_l}
      2. snow-timing controlled — `frac_snow ≥ 0.4`
      3. storage-memory contr.  — `bfi ≥ 0.55`
      4. climate-partitioning   — `aridity_class == dry`
      5. threshold/event        — remainder
    Best model per basin is shown as a marker shape so the map shows zone
    boundary AND which architecture wins in each zone.
    """
    if skill_attrs.empty or attrs_geo.empty:
        return
    if "lon_deg" not in attrs_geo.columns:
        return
    sk = skill_attrs.copy()
    sk["kge"] = pd.to_numeric(sk["kge"], errors="coerce")
    sk = sk.dropna(subset=["kge"])
    if sk.empty:
        return
    best = sk.sort_values("kge", ascending=False).drop_duplicates("ID")[["ID", "model", "kge"]]
    best = best.rename(columns={"model": "best_model", "kge": "best_kge"})

    geo = attrs_geo.copy()
    for col in ("frac_snow", "bfi", "is_impacted"):
        if col in geo.columns:
            geo[col] = pd.to_numeric(geo[col], errors="coerce")

    df = geo.merge(best, on="ID", how="inner").dropna(subset=["lon_deg", "lat_deg"])
    if df.empty:
        return

    def assign_zone(row) -> str:
        impact_cls = str(row.get("impact_class", "")).strip()
        if int(row.get("is_impacted", 0) or 0) == 1 and impact_cls in {"impact_m", "impact_l"}:
            return "human_altered"
        if float(row.get("frac_snow", 0.0) or 0.0) >= 0.40:
            return "snow_timing"
        if float(row.get("bfi", 0.0) or 0.0) >= 0.55:
            return "storage_memory"
        if str(row.get("aridity_class", "")) == "dry":
            return "climate_partitioning"
        return "threshold_event"

    df["zone"] = df.apply(assign_zone, axis=1)

    ZONE_ORDER = ["climate_partitioning", "storage_memory", "snow_timing",
                  "threshold_event", "human_altered"]
    ZONE_LABEL = {
        "climate_partitioning": "Climate-partitioning",
        "storage_memory":       "Storage-memory",
        "snow_timing":          "Snow-timing",
        "threshold_event":      "Threshold/event",
        "human_altered":        "Human-altered",
    }
    ZONE_COLOR = {
        "climate_partitioning": "#B5302A",
        "storage_memory":       "#0E2A55",
        "snow_timing":          "#1F77BD",
        "threshold_event":      "#586471",
        "human_altered":        "#9CA3AB",
    }
    BEST_MARKER = {
        "random_forest": "s",
        "xgboost":       "D",
        "xlstm":         "o",
        "transformer":   "^",
    }

    use_cartopy, proj, data_crs, cartopy_features = _init_cartopy()

    fig = plt.figure(figsize=(COL_DOUBLE, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.3], wspace=0.40)
    if use_cartopy:
        ax = fig.add_subplot(gs[0, 0], projection=proj)
        _add_basemap(ax, cartopy_features, extent=(7.5, 18.5, 45.0, 51.5),
                     data_crs=data_crs, gridlines=True, label_size=6.4)
    else:
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor(LAND_COLOR)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect(1.5)
        ax.set_xlim(7.5, 18.5)
        ax.set_ylim(45.0, 51.5)

    for zone in ZONE_ORDER:
        zg = df[df["zone"].eq(zone)]
        if zg.empty:
            continue
        for m in _present_models(zg.rename(columns={"best_model": "model"})):
            mg = zg[zg["best_model"].eq(m)]
            if mg.empty:
                continue
            scatter_kwargs = dict(
                color=ZONE_COLOR[zone], marker=BEST_MARKER.get(m, "o"),
                s=20, edgecolor="white", linewidth=0.4, zorder=4, alpha=0.92,
            )
            if use_cartopy:
                scatter_kwargs["transform"] = data_crs
            ax.scatter(mg["lon_deg"], mg["lat_deg"], **scatter_kwargs)

    # (b) Zone composition stacked horizontal bar
    bx = fig.add_subplot(gs[0, 1])
    counts = df["zone"].value_counts().reindex(ZONE_ORDER, fill_value=0)
    yvals = np.arange(len(ZONE_ORDER))
    bx.barh(yvals, counts.values, color=[ZONE_COLOR[z] for z in ZONE_ORDER],
            edgecolor="white", linewidth=0.5, zorder=3)
    bar_max = float(counts.values.max()) if counts.values.size else 1.0
    # Uniform neutral-colour label to the right of every bar (never on top).
    for y, z in zip(yvals, ZONE_ORDER):
        n = int(counts.loc[z])
        med_kge = df[df["zone"].eq(z)]["best_kge"].median()
        if n > 0 and np.isfinite(med_kge):
            txt = f"$n$={n}  ·  KGE {med_kge:.2f}"
        else:
            txt = f"$n$={n}"
        bx.text(n + bar_max * 0.03, y, txt,
                va="center", ha="left", fontsize=6.9,
                color=ANNOT_COLOR)
    bx.set_yticks(yvals)
    bx.set_yticklabels([ZONE_LABEL[z] for z in ZONE_ORDER], fontsize=7.6)
    bx.invert_yaxis()
    bx.set_xlabel("Catchments")
    _grid(bx, axis="x")
    bx.set_xlim(0, bar_max * 1.85)
    _panel_label(bx, "b", x=-0.42, y=1.02)

    # Two stacked legends below the map: zones (color) on top row, best-model
    # marker (shape) on the row below.  The previous one-row layout collided
    # zone colours with marker shapes and ran off the right margin.
    zone_handles = [Patch(facecolor=ZONE_COLOR[z], edgecolor="white",
                          label=ZONE_LABEL[z]) for z in ZONE_ORDER]
    best_models_present = [m for m in ["random_forest", "xgboost", "xlstm",
                                        "transformer"]
                           if m in set(df["best_model"].astype(str))]
    model_handles = [Line2D([0], [0], marker=BEST_MARKER.get(m, "o"),
                            color="white", markerfacecolor="#444",
                            markeredgecolor="white", markersize=6, lw=0,
                            label=f"best: {model_label(m)}")
                     for m in best_models_present]
    # Anchor both stacked legends at the FIGURE level, not the axis.  An axis
    # legend with bbox_to_anchor=(0.5, …) is centred on the (a) axis only,
    # which clipped "Climate-partitioning" off the left margin in 2x2 layout.
    leg_zone = fig.legend(handles=zone_handles, loc="lower center",
                          bbox_to_anchor=(0.5, 0.10),
                          fontsize=6.9, frameon=False,
                          ncol=len(zone_handles),
                          title="Predictability zone",
                          title_fontsize=7.0,
                          columnspacing=1.4, handlelength=1.4,
                          handletextpad=0.45)
    if model_handles:
        fig.legend(handles=model_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.005),
                   fontsize=6.9, frameon=False,
                   ncol=len(model_handles),
                   title="Best model (marker shape)",
                   title_fontsize=7.0,
                   columnspacing=1.4, handlelength=1.4,
                   handletextpad=0.45)
    _panel_label(ax, "a", x=-0.02, y=1.04)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.22,
                        wspace=0.40)
    _save(fig, fig_dir / "fig22_predictability_zones.png")


# ─── Figure 23. Conceptual framework of diagnostic models ────────────────────

def plot_conceptual_framework(fig_dir: Path):
    """Schematic placing each diagnostic model in process-control space.

    Pure-schematic, no run data — a visual reference for the introduction.
    Maps the four model architectures onto two hydrological axes:
      x: temporal-memory requirement (none → multi-month)
      y: process abstraction (threshold/regime → continuous storage)
    """
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor(LAND_COLOR)

    # Axis arrows
    ax.annotate("", xy=(9.7, 0.6), xytext=(0.3, 0.6),
                arrowprops=dict(arrowstyle="->", color=AXIS_COLOR, lw=1.0))
    ax.annotate("", xy=(0.3, 9.7), xytext=(0.3, 0.6),
                arrowprops=dict(arrowstyle="->", color=AXIS_COLOR, lw=1.0))
    ax.text(5.0, 0.05, "Temporal-memory requirement",
            ha="center", va="bottom", fontsize=8.6, color=AXIS_COLOR)
    ax.text(0.10, 5.0, "Process abstraction",
            ha="center", va="center", rotation=90, fontsize=8.6, color=AXIS_COLOR)
    ax.text(0.65, 0.30, "none", fontsize=7.0, color="#555555")
    ax.text(9.55, 0.30, "multi-month", ha="right", fontsize=7.0, color="#555555")
    ax.text(0.55, 1.20, "thresholds /\nregimes", fontsize=7.0, color="#555555", va="center")
    ax.text(0.55, 9.05, "continuous\nstorage", fontsize=7.0, color="#555555", va="center")

    # Model boxes with role text
    boxes = [
        ("random_forest", 2.0, 2.9, "Random Forest",
         "Static regime partitioning\nNonlinear baseline\nPermutation importance"),
        ("xgboost",       4.3, 4.1, "XGBoost",
         "Hydrological thresholds\nFeature interactions\nSHAP / partial dep."),
        ("transformer",   6.3, 5.9, "Transformer",
         "Event-sequence logic\nAttention to forcing lags\nHigh-flow timing"),
        ("xlstm",         8.4, 7.6, "xLSTM",
         "Long hydrological memory\nStorage, snow, baseflow\nDelayed release"),
    ]
    box_w, box_h = 1.20, 0.95
    for model, cx, cy, title, body in boxes:
        color = model_color(model)
        rect = Rectangle((cx - box_w, cy - box_h), 2 * box_w, 2 * box_h,
                         facecolor=color, alpha=0.92, edgecolor="white",
                         linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy + 0.55, title, ha="center", va="center",
                fontsize=8.8, color="white", fontweight="semibold", zorder=4)
        ax.text(cx, cy - 0.18, body, ha="center", va="center",
                fontsize=6.5, color="white", zorder=4, linespacing=1.18)

    # AutoResearch loop badge
    ax.text(5.0, 9.30,
            "AutoResearch loop  ·  hydrology-constrained experiments",
            ha="center", va="center", fontsize=7.8, color=AXIS_COLOR,
            bbox=dict(facecolor="white", edgecolor="0.7",
                      boxstyle="round,pad=0.32", lw=0.6))

    # Diagonal "diagnostic spectrum" guide arrow (dashed, behind boxes)
    ax.annotate("", xy=(7.8, 7.2), xytext=(2.5, 2.7),
                arrowprops=dict(arrowstyle="->", color="#7A8590",
                                lw=0.7, ls=(0, (3, 2))), zorder=2)
    ax.text(5.10, 4.75, "diagnostic spectrum",
            ha="center", va="center",
            fontsize=6.8, color="#7A8590", fontstyle="italic", rotation=42,
            zorder=2,
            bbox=dict(facecolor=LAND_COLOR, edgecolor="none", pad=0.6))

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)
    _save(fig, fig_dir / "fig23_conceptual_framework.png")


# ─── Orchestrator ────────────────────────────────────────────────────────────

# Convenience colour aliases used by figures that need raw palette tones.
HESS_NAVY = "#0E2A55"
HESS_RED  = "#B5302A"

CANONICAL_FIGURES = {
    "fig00_study_domain",
    "fig01_skill_cdf",
    "fig02_transfer_skill",
    "fig02_basin_map_kge",
    "fig03_signature_errors",
    "fig04_regime_violins",
    "fig05_skill_vs_attribute",
    "fig06_hydrographs",
    "fig07_flow_duration_curves",
    "fig08_signature_obs_vs_sim",
    "fig09_storm_event_zoom",
    "fig10_feature_importance",
    "fig11_signature_winners",
    "fig12_human_impact_pbias",
    "fig13_hydrograph_mosaic",
    "fig14_kge_pbias_landscape",
    "fig15_regime_kge_heatmap",
    "fig16_runoff_coefficient",
    "fig17a_hydrograph_atlas",
    "fig17b_hydrograph_atlas",
    "fig18_signature_bias_ratios",
    "fig19_flood_typology_seasonality",
    "fig20_flood_typology_model_skill",
    "fig21_snow_timing_error",
    "fig22_predictability_zones",
    "fig23_conceptual_framework",
    "fig25_transfer_kge_maps",
    "fig26_memory_gain_maps",
    "fig27_transfer_regime_delta",
    "fig28_seasonal_residual_cycle",
    "fig29_budyko_aridity_partitioning",
    "fig30_memory_gradient_diagnostics",
    "fig31_spatial_novelty_transfer",
    "fig32_climate_sensitivity_flood_errors",
    "fig33_cosero_benchmark_dashboard",
    "fig34_transferability_dashboard",
    "fig35_cosero_transfer_maps",
    "fig36_training_tuning_audit",
    "fig37_model_setup_audit",
    "fig38_uncertainty_dashboard",
}


def make_plots(out_dir: Path):
    apply_style()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    export_plot_data(out_dir)
    pd_dir = out_dir / "plot_data"

    metrics = _read(pd_dir / "basin_skill_long.parquet")
    if metrics.empty:
        metrics = _read(out_dir / "tables" / "metrics_by_basin.csv")
    sig_long = _read(pd_dir / "signature_obs_sim_long.parquet")
    skill_attrs = _read(pd_dir / "skill_attributes_test.parquet")
    attrs_geo = _read(pd_dir / "basin_attributes_geo.parquet")
    examples = _read(pd_dir / "example_basins.parquet")
    hydrographs = _read(pd_dir / "example_hydrographs_long.parquet")
    sig_summary = _read(out_dir / "tables" / "signature_errors.csv")
    imp = _read(out_dir / "tables" / "feature_importance.csv")
    ranks = _read(out_dir / "tables" / "signature_model_ranking.csv")
    flood_events = _read(pd_dir / "flood_events.parquet")
    flood_model_summary = _read(pd_dir / "flood_typology_model_summary.parquet")
    flood_event_skill = _read(out_dir / "tables" / "flood_event_model_skill.csv")
    model_delta = _read(out_dir / "tables" / "model_delta_diagnostics.csv")
    zone_summary = _read(out_dir / "tables" / "process_zone_summary.csv")

    raw = pd.DataFrame()
    raw_path = pd_dir / "example_raw_long.parquet"
    if raw_path.exists():
        raw = _read(raw_path)
        if not raw.empty and "date" in raw.columns:
            raw["date"] = pd.to_datetime(raw["date"])

    # Normalise legacy column names: some runs persisted projected `lon`/`lat`
    # (metres, ETRS89 / LAEA). Plotting code expects geodetic `lon_deg`/`lat_deg`.
    # When projected metres are detected (values >> 360) attempt a one-shot
    # back-transform to WGS84 degrees so basin maps render in the correct frame.
    if not attrs_geo.empty:
        if "lon_deg" not in attrs_geo.columns and "lon" in attrs_geo.columns:
            try:
                _maxabs = float(pd.to_numeric(attrs_geo["lon"], errors="coerce").abs().max())
            except Exception:
                _maxabs = 0.0
            if _maxabs > 360:
                try:
                    from pyproj import Transformer
                    tr = Transformer.from_crs(3035, 4326, always_xy=True)
                    lon_deg, lat_deg = tr.transform(
                        attrs_geo["lon"].to_numpy(), attrs_geo["lat"].to_numpy())
                    attrs_geo = attrs_geo.assign(lon_deg=lon_deg, lat_deg=lat_deg)
                except Exception as exc:
                    print(f"[plots] could not back-project lon/lat ({exc.__class__.__name__})")
            else:
                attrs_geo = attrs_geo.assign(lon_deg=attrs_geo["lon"],
                                             lat_deg=attrs_geo["lat"])

    for old in list(fig_dir.glob("*.png")) + list(fig_dir.glob("*.pdf")):
        if old.stem not in CANONICAL_FIGURES:
            old.unlink()

    plot_study_domain(attrs_geo, fig_dir)
    plot_skill_cdfs(metrics, fig_dir)
    plot_temporal_spatial_transfer(metrics, fig_dir)
    plot_transfer_kge_maps(metrics, attrs_geo, fig_dir)
    plot_memory_gain_maps(model_delta, attrs_geo, fig_dir)
    plot_transfer_regime_delta(zone_summary, fig_dir)
    plot_seasonal_residual_cycle(out_dir, fig_dir)
    plot_budyko_aridity_partitioning(sig_long, attrs_geo, fig_dir)
    plot_memory_gradient_diagnostics(model_delta, fig_dir)
    plot_spatial_novelty_transfer(model_delta, fig_dir)
    plot_climate_sensitivity_flood_errors(flood_event_skill, flood_events, fig_dir)
    plot_cosero_benchmark_dashboard(metrics, sig_summary, fig_dir)
    plot_transferability_dashboard(metrics, attrs_geo, fig_dir)
    plot_cosero_transfer_maps(metrics, attrs_geo, fig_dir)
    plot_training_and_tuning_audit(out_dir, fig_dir)
    plot_model_setup_audit(out_dir, fig_dir)
    plot_uncertainty_dashboard(out_dir, fig_dir)
    plot_basin_map(metrics, attrs_geo, fig_dir)
    plot_signature_errors(sig_summary, fig_dir)
    plot_regime_violins(skill_attrs, fig_dir)
    plot_skill_vs_attribute(skill_attrs, fig_dir)
    plot_hydrographs(hydrographs, examples, raw, fig_dir)
    plot_flow_duration_curves(hydrographs, examples, fig_dir)
    plot_signature_obs_vs_sim(sig_long, fig_dir)
    plot_storm_zoom(hydrographs, examples, raw, fig_dir)
    plot_feature_importance(imp, fig_dir)
    plot_signature_winners(ranks, fig_dir)
    plot_human_impact_pbias(skill_attrs, fig_dir)
    plot_hydrograph_mosaic(hydrographs, examples, raw, fig_dir)
    plot_kge_pbias_landscape(metrics, fig_dir)
    plot_regime_kge_heatmap(skill_attrs, fig_dir)
    plot_runoff_coefficient(sig_long, attrs_geo, fig_dir)
    plot_hydrograph_atlas(hydrographs, examples, raw, fig_dir)
    plot_signature_bias_ratios(sig_long, fig_dir)
    plot_flood_typology_seasonality(flood_events, fig_dir)
    plot_flood_typology_model_skill(flood_model_summary, fig_dir)
    plot_snow_timing_error(sig_long, attrs_geo, fig_dir)
    plot_predictability_zones(skill_attrs, attrs_geo, fig_dir)
    plot_conceptual_framework(fig_dir)
