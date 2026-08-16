#!/usr/bin/env python3
"""Figure 6, rebuilt from the actual agent-selected confirmation runs.

The previous version was carried over from an earlier pipeline run and its panel
medians did not match Table 4. This builds the maps directly from
agent_confirm_selected_<family>/tables/metrics_by_basin.csv, so the figure and
the table are guaranteed to agree.

Layout is 4 rows (families) x 2 columns (temporal / held-out) so each map is
substantially larger on a portrait page than the previous 2 x 4 arrangement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper4_pipeline/src"))

# the pipeline caches Natural Earth data locally at 50 m; the default user data
# directory is not writable here, so point cartopy at the cache before import
_CDIR = ROOT / "paper4_pipeline/outputs/.cartopy_data"
os.environ.setdefault("CARTOPY_DATA_DIR", str(_CDIR))
import cartopy

cartopy.config["data_dir"] = _CDIR
cartopy.config["pre_existing_data_dir"] = _CDIR

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from paper4.style import apply_style, COL_DOUBLE

AR = ROOT / "paper4_pipeline/outputs/autoresearch/runs"
FIG = ROOT / "paper4_pipeline/outputs/hess_100train_50test_final/figures"
IDX = ROOT / "paper4_pipeline/outputs/hess_100train_50test_final/tables/catchment_index.csv"

FAMS = [("xgboost", "XGBoost"), ("random_forest", "Random Forest"),
        ("xlstm", "xLSTM"), ("transformer", "Transformer")]
SPLITS = [("test", "Temporal confirmation\n(100 development catchments, 2014-2017)"),
          ("spatial_test", "Held-out catchments\n(50 catchments, 2014-2017)")]
CMAP = "RdYlBu"
EXTENT = (8.0, 18.3, 46.2, 50.2)


def load(fam: str) -> pd.DataFrame:
    d = pd.read_csv(AR / f"agent_confirm_selected_{fam}" / "tables" / "metrics_by_basin.csv")
    return d[d.model.eq(fam)]


def coords() -> pd.DataFrame:
    """catchment_index stores gauge locations in EPSG:3035 metres, not degrees."""
    from pyproj import Transformer
    g = pd.read_csv(IDX)[["ID", "lon", "lat"]]
    tf = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    g["lon_deg"], g["lat_deg"] = tf.transform(g.lon.values, g.lat.values)
    return g[["ID", "lon_deg", "lat_deg"]]


def main() -> None:
    apply_style()
    geo = coords()
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.5, vmax=1.0)
    proj, crs = ccrs.PlateCarree(), ccrs.PlateCarree()

    # each map is ~10.3 deg wide by 4.0 deg tall; size the rows to that aspect so
    # the panels sit tight instead of being letterboxed inside taller cells
    fig = plt.figure(figsize=(COL_DOUBLE, 6.1))
    gs = fig.add_gridspec(len(FAMS), 3, width_ratios=[1, 1, 0.030],
                          wspace=0.06, hspace=0.10)
    sc = None
    for i, (fam, flabel) in enumerate(FAMS):
        m = load(fam).merge(geo, on="ID", how="left").dropna(subset=["lon_deg", "lat_deg", "kge"])
        for j, (split, slabel) in enumerate(SPLITS):
            ax = fig.add_subplot(gs[i, j], projection=proj)
            ax.set_extent(EXTENT, crs=crs)
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F2F1ED", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCE7EE", zorder=0)
            ax.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor="#AFC7D6",
                           lw=0.35, zorder=1)
            ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#DCE7EE",
                           edgecolor="#AFC7D6", lw=0.25, zorder=1)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#9AA5AC",
                           lw=0.45, zorder=2)
            ax.coastlines("50m", color="#9AA5AC", lw=0.45, zorder=2)

            sub = m[m.split.eq(split)]
            sc = ax.scatter(sub.lon_deg, sub.lat_deg, c=sub.kge, cmap=CMAP, norm=norm,
                            s=42 if split == "spatial_test" else 26,
                            edgecolor="#33393D", lw=0.35, transform=crs, zorder=4)
            ax.text(0.025, 0.955, f"median KGE = {sub.kge.median():.3f}   n = {len(sub)}",
                    transform=ax.transAxes, fontsize=6.0, va="top",
                    bbox=dict(facecolor="white", edgecolor="#CCCCCC", lw=0.4,
                              boxstyle="round,pad=0.25", alpha=0.9), zorder=5)
            if i == 0:
                ax.set_title(slabel, fontsize=7.2, pad=5)
            if j == 0:
                ax.text(-0.035, 0.5, flabel, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8.0, fontweight="bold")
            for s in ax.spines.values():
                s.set_edgecolor("#8A9AA6"); s.set_linewidth(0.6)

    cax = fig.add_subplot(gs[1:3, -1])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Kling-Gupta efficiency", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.5)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig06_transfer_maps.{ext}", dpi=600, bbox_inches="tight")
    print("wrote", FIG / "fig06_transfer_maps.png")
    for fam, lab in FAMS:
        m = load(fam)
        for sp, _ in SPLITS:
            print(f"  {lab:14s} {sp:13s} median KGE {m[m.split.eq(sp)].kge.median():.3f}")


if __name__ == "__main__":
    main()
