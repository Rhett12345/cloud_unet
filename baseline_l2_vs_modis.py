"""
baseline_l2_vs_modis.py
========================
Product-level cross-comparison: AGRI L2 CLP/CTH vs MODIS MYD06 CLP/CTH.

No model involved — this measures the inherent disagreement between two
official cloud products, serving as a baseline for model evaluation.

For each AGRI L2 scene with time-matched MODIS granules:
  1. Read AGRI L2 CLP/CTH .NC, remap CLP to 3 classes
  2. Read AGRI L1B FDI+GEO for geolocation (lat/lon grid)
  3. Spatially match MODIS MYD06 → AGRI grid via fusion engine
  4. Pixel-by-pixel comparison: CLP OA + confusion matrix, CTH RMSE/R/Bias
  5. Generate publication-quality figures

Usage:
  python baseline_l2_vs_modis.py --day 20190505 [--quality] [--summary]
"""

import argparse, logging, sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as cfg
from fusion_core import aggregate_modis_to_agri, check_modis_in_agri_disk
from fusion_io import (
    apply_quality_filter,
    parse_agri_datetime, parse_modis_datetime,
    find_matching_modis, find_matching_myd03,
    read_agri_scene, read_myd06, read_modis_geo_quick,
    _extract_timestamp_from_filename, _find_matching_l2_file,
)

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# matplotlib publication defaults
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "l2_vs_modis_baseline"


def save_figure(fig, stem: str):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# AGRI L2 reader (same as validate_agri_l2)
# ---------------------------------------------------------------------------

def read_l2_clp(nc_path: Path) -> Optional[np.ndarray]:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CLP"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.int16)
    ds.close()
    clp = np.full(raw.shape, np.nan, dtype=np.float32)
    for src, dst in cfg.AGRI_L2_CLP_PHASE_MAP.items():
        clp[raw == src] = float(dst)
    return clp


def read_l2_cth(nc_path: Path) -> Optional[np.ndarray]:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CTH"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.float32)
    ds.close()
    vmin, vmax = cfg.AGRI_L2_CTH_VALID_RANGE
    raw[(raw <= 0) | (raw >= 65500) | ~np.isfinite(raw)] = np.nan
    raw[raw < vmin] = np.nan
    raw[raw > vmax] = np.nan
    return raw


# ---------------------------------------------------------------------------
# MODIS → AGRI grid matching (same as validate_modis)
# ---------------------------------------------------------------------------

def match_modis_to_agri_grid(agri_lat: np.ndarray, agri_lon: np.ndarray,
                              agri_dt, modis_files: list,
                              myd03_files: list,
                              quality: bool = True) -> Optional[Dict[str, np.ndarray]]:
    if quality:
        cfg.MODIS_FILTER_WEAK_QUALITY = True
        cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 3)
        cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0,)

    modis_list = []
    for mf in modis_files:
        mf = Path(mf) if isinstance(mf, str) else mf
        myd03_file = find_matching_myd03(mf, myd03_files)
        geo = read_modis_geo_quick(mf, myd03_file=myd03_file)
        if geo is None:
            continue
        mlat = geo.get("lat_1km") if geo.get("lat_1km") is not None else geo.get("lat_5km")
        mlon = geo.get("lon_1km") if geo.get("lon_1km") is not None else geo.get("lon_5km")
        if mlat is None or mlon is None:
            continue
        if not check_modis_in_agri_disk(mlat, mlon, agri_lat, agri_lon):
            continue
        m = read_myd06(mf, agri_dt=agri_dt, myd03_file=myd03_file, geo_cache=geo)
        if m is None:
            continue
        mdt = parse_modis_datetime(mf.name)
        if mdt:
            m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
        modis_list.append(m)

    if not modis_list:
        return None

    labels = aggregate_modis_to_agri(agri_lat, agri_lon, modis_list)
    if labels is None:
        return None

    vza = np.zeros_like(agri_lat)
    sza = np.zeros_like(agri_lat)
    labels = apply_quality_filter({"VZA": vza, "SZA": sza}, labels)
    return labels


# ---------------------------------------------------------------------------
# Metrics (same as validate_agri_l2)
# ---------------------------------------------------------------------------

def compute_metrics(clp_a: np.ndarray, clp_b: np.ndarray,
                    cth_a: np.ndarray, cth_b: np.ndarray) -> dict:
    valid_clp = np.isfinite(clp_a) & np.isfinite(clp_b)
    n_clp = int(valid_clp.sum())
    if n_clp == 0:
        return {"n_clp": 0, "n_cth": 0}

    pred = clp_a[valid_clp].astype(np.int32)
    true = clp_b[valid_clp].astype(np.int32)

    oa = float((pred == true).mean()) * 100.0

    classes = sorted(set(pred) | set(true))
    per_class = {}
    for c in classes:
        mask = true == c
        per_class[f"cls{c}_acc"] = float((pred[mask] == c).mean()) * 100.0 if mask.any() else -1.0
    accs = [v for v in per_class.values() if v >= 0]
    macro_acc = float(np.mean(accs)) if accs else 0.0

    cm = np.zeros((3, 3), dtype=np.int64)
    for ti in range(3):
        for pi in range(3):
            cm[ti, pi] = int(((true == ti) & (pred == pi)).sum())

    valid_cth = (np.isfinite(cth_a) & np.isfinite(cth_b) &
                 np.isfinite(clp_a) & np.isfinite(clp_b) &
                 (clp_a > 0) & (clp_b > 0))
    n_cth = int(valid_cth.sum())
    cth_r = cth_rmse = cth_mae = cth_bias = 0.0
    if n_cth > 10:
        p, t = cth_a[valid_cth], cth_b[valid_cth]
        cth_rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        cth_mae  = float(np.mean(np.abs(p - t)))
        cth_bias = float(np.mean(p - t))
        pm, tm = p - p.mean(), t - t.mean()
        denom = np.sqrt((pm ** 2).sum() * (tm ** 2).sum())
        cth_r = float((pm * tm).sum() / max(denom, 1e-12))

    return {"n_clp": n_clp, "n_cth": n_cth, "oa": oa, "macro_acc": macro_acc,
            "confusion_matrix": cm, "cth_rmse": cth_rmse, "cth_mae": cth_mae,
            "cth_bias": cth_bias, "cth_r": cth_r, **per_class}


# ---------------------------------------------------------------------------
# Figures (reuse from validate_agri_l2)
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, class_names: list, scene_id: str):
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i,j]:.1f}%\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=6,
                    color="white" if cm_norm[i, j] > 50 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(class_names, fontsize=6.5)
    ax.set_yticklabels(class_names, fontsize=6.5)
    ax.set_xlabel("AGRI L2", fontsize=7)
    ax.set_ylabel("MODIS MYD06", fontsize=7)
    ax.set_title(f"AGRI L2 vs MODIS\n{scene_id}", fontsize=7, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Row %", fontsize=6)
    fig.tight_layout()
    return fig


def _plot_cth_scatter(cth_a: np.ndarray, cth_b: np.ndarray,
                      metrics: dict, label_a: str, label_b: str,
                      scene_id: str, max_samples: int = 50_000):
    valid = np.isfinite(cth_a) & np.isfinite(cth_b)
    p, t = cth_a[valid], cth_b[valid]
    if len(p) > max_samples:
        idx = np.random.choice(len(p), max_samples, replace=False)
        p, t = p[idx], t[idx]

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    bins = np.linspace(0, 18000, 120)
    h, xe, ye = np.histogram2d(t, p, bins=[bins, bins])
    h = h.T
    h_log = np.log10(h.clip(min=1))
    im = ax.pcolormesh(xe, ye, h_log, cmap="YlOrRd", rasterized=True, shading="auto")
    ax.plot([0, 18000], [0, 18000], "k--", lw=0.7, alpha=0.5)
    ax.text(0.03, 0.97,
            f"$R$ = {metrics['cth_r']:.4f}\nRMSE = {metrics['cth_rmse']:.0f} m\n"
            f"MAE = {metrics['cth_mae']:.0f} m\nBias = {metrics['cth_bias']:+.0f} m\n"
            f"$n$ = {metrics['n_cth']:,}",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax.set_xlabel(f"{label_b} CTH (m)", fontsize=7)
    ax.set_ylabel(f"{label_a} CTH (m)", fontsize=7)
    ax.set_title(f"CTH: {label_a} vs {label_b}\n{scene_id}", fontsize=7, fontweight="bold")
    ax.set_xlim(0, 18000); ax.set_ylim(0, 18000)
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("log$_{10}$(count)", fontsize=6)
    fig.tight_layout()
    return fig


def _plot_spatial_error(clp_a: np.ndarray, clp_b: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray, scene_id: str):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    valid = np.isfinite(clp_a) & np.isfinite(clp_b)
    agree = valid & (clp_a == clp_b)
    disagree = valid & (clp_a != clp_b)

    sub_lon = 104.7
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.PlateCarree(central_longitude=sub_lon)

    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw={"projection": map_crs})
    ax.add_feature(cfeature.COASTLINE, lw=0.4, alpha=0.5, zorder=2)
    ax.set_extent([sub_lon - 85, sub_lon + 85, -85, 85], crs=data_crs)

    ax.scatter(lon[agree][::10], lat[agree][::10], s=0.15, alpha=0.25,
               color="royalblue", rasterized=True, zorder=1,
               transform=data_crs, label="Agreement")
    ax.scatter(lon[disagree][::10], lat[disagree][::10], s=0.3, alpha=0.6,
               color="crimson", rasterized=True, zorder=2,
               transform=data_crs, label="Disagreement")

    gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle="--", linewidth=0.4)
    gl.top_labels = False; gl.right_labels = False

    oa = (agree.sum() / max(valid.sum(), 1)) * 100
    ax.set_title(f"AGRI L2 vs MODIS — {scene_id}\nOA = {oa:.1f}%  (blue=agree, red=disagree)",
                 fontsize=7, fontweight="bold")
    ax.legend(loc="lower left", fontsize=6, markerscale=6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def compare_one(clp_nc: Path, cth_nc: Path, agri_file: Path,
                out_dir: Path, scene_id: str = "", quality: bool = True) -> dict:
    """Compare one AGRI L2 scene against time-matched MODIS."""
    # 1. Read AGRI L2
    clp_l2 = read_l2_clp(clp_nc)
    cth_l2 = read_l2_cth(cth_nc)
    if clp_l2 is None or cth_l2 is None:
        raise RuntimeError(f"Failed to read L2 for {scene_id}")

    # 2. Read AGRI L1B for geolocation
    agri = read_agri_scene(agri_file)
    if agri is None:
        raise RuntimeError(f"Failed to read AGRI scene {agri_file}")
    agri_dt = parse_agri_datetime(agri_file.name)

    # 3. Find matching MODIS
    day_str = agri_dt.strftime("%Y%m%d")
    modis_dir = cfg.MODIS_ROOT / day_str
    myd03_dir = cfg.MYD03_ROOT / day_str
    modis_files = sorted(list(modis_dir.glob("MYD06*.hdf")) + list(modis_dir.glob("MYD06*.HDF"))) if modis_dir.is_dir() else []
    myd03_files = sorted(list(myd03_dir.glob("MYD03*.hdf")) + list(myd03_dir.glob("MYD03*.HDF"))) if myd03_dir.is_dir() else []

    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_matching_modis", "n_granules": 0}

    # 4. Match MODIS → AGRI grid
    labels = match_modis_to_agri_grid(agri["lat"], agri["lon"], agri_dt,
                                       matched, myd03_files, quality=quality)
    if labels is None:
        return {"scene_id": scene_id, "status": "modis_matching_failed",
                "n_granules": len(matched)}

    # 5. Remap MODIS CLP
    clp_modis = labels["CLP"].astype(np.float32)
    clp_modis[(clp_modis < 0) | (clp_modis >= 3)] = np.nan
    cth_modis = labels["CTH"]

    # 6. Compute metrics (AGRI L2 = a, MODIS = b)
    assert clp_l2.shape == clp_modis.shape, f"Shape mismatch: {clp_l2.shape} vs {clp_modis.shape}"
    metrics = compute_metrics(clp_l2, clp_modis, cth_l2, cth_modis)
    metrics["scene_id"] = scene_id
    metrics["status"] = "ok"
    metrics["n_granules"] = len(matched)
    metrics["n_valid_pct"] = metrics["n_clp"] / max(clp_l2.size, 1) * 100

    log.info("  %s  AGRI L2 vs MODIS  OA=%5.2f%%  CTH R=%+.4f  RMSE=%.0f m  n_px=%d",
             scene_id, metrics["oa"], metrics["cth_r"], metrics["cth_rmse"], metrics["n_clp"])

    # 7. Figures
    if out_dir and metrics["n_clp"] > 100:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = scene_id.replace(" ", "_")

        cm = metrics.get("confusion_matrix")
        if cm is not None:
            fig = _plot_confusion_matrix(cm, cfg.CLP_CLASS_NAMES, scene_id)
            save_figure(fig, f"{stem}_l2_vs_modis_confusion")
            plt.close(fig)

        if metrics["n_cth"] > 10:
            fig = _plot_cth_scatter(cth_l2, cth_modis, metrics,
                                    "AGRI L2", "MODIS", scene_id)
            save_figure(fig, f"{stem}_l2_vs_modis_cth")
            plt.close(fig)

        fig = _plot_spatial_error(clp_l2, clp_modis, agri["lat"], agri["lon"], scene_id)
        save_figure(fig, f"{stem}_l2_vs_modis_spatial")
        plt.close(fig)

    return metrics


def main():
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL),
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        stream=sys.stdout)
    parser = argparse.ArgumentParser(description="AGRI L2 vs MODIS MYD06 baseline comparison")
    parser.add_argument("--day", required=True)
    parser.add_argument("--quality", action="store_true", default=True, help="Enable MODIS quality filtering")
    parser.add_argument("--no-quality", dest="quality", action="store_false", help="Disable MODIS quality filtering")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    day = args.day
    agri_day_dir = cfg.AGRI_ROOT / day
    if not agri_day_dir.is_dir():
        log.error("AGRI L1B directory not found: %s", agri_day_dir)
        sys.exit(1)

    # Build AGRI FDI file list
    agri_files = sorted([f for f in agri_day_dir.glob("*_FDI-_*.HDF") if not f.name.endswith(".db")])

    log.info("Day %s: %d AGRI scenes, quality=%s", day, len(agri_files), args.quality)

    out_dir = Path(args.out_dir)
    all_metrics = []
    for agri_f in agri_files:
        ts = _extract_timestamp_from_filename(agri_f.name)
        if ts is None:
            continue
        scene_id = f"{ts[:8]}_{ts[8:]}"

        # Find matching L2 files
        dummy_fdi_name = f"FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_{ts}_x_4000M_V0001.HDF"
        clp_nc = _find_matching_l2_file(Path(dummy_fdi_name), "CLP")
        cth_nc = _find_matching_l2_file(Path(dummy_fdi_name), "CTH")
        if clp_nc is None or cth_nc is None:
            continue

        try:
            m = compare_one(clp_nc, cth_nc, agri_f, out_dir, scene_id, quality=args.quality)
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", scene_id, exc)

    if all_metrics and args.summary:
        import pandas as pd
        rows = [{k: v for k, v in m.items() if k != "confusion_matrix"} for m in all_metrics]
        df = pd.DataFrame(rows)
        csv_path = out_dir / f"baseline_l2_vs_modis_{day}.csv"
        df.to_csv(csv_path, index=False)

        ok = [m for m in all_metrics if m.get("status") == "ok"]
        if ok:
            oa_vals = [m["oa"] for m in ok if m["oa"] > 0]
            r_vals  = [m["cth_r"] for m in ok if m["n_cth"] > 10]
            rmse_vals = [m["cth_rmse"] for m in ok if m["n_cth"] > 10]
            bias_vals = [m["cth_bias"] for m in ok if m["n_cth"] > 10]
            pct_vals  = [m["n_valid_pct"] for m in ok]
            log.info("=== BASELINE: AGRI L2 vs MODIS  %s (n=%d/%d) ===", day, len(ok), len(all_metrics))
            log.info("CLP OA:     %.2f ± %.2f %%", np.mean(oa_vals), np.std(oa_vals))
            log.info("CTH R:      %+.4f ± %.4f", np.mean(r_vals), np.std(r_vals))
            log.info("CTH RMSE:   %.0f ± %.0f m", np.mean(rmse_vals), np.std(rmse_vals))
            log.info("CTH Bias:   %+.0f ± %.0f m", np.mean(bias_vals), np.std(bias_vals))
            log.info("Coverage:   %.1f ± %.1f %%", np.mean(pct_vals), np.std(pct_vals))


if __name__ == "__main__":
    main()
