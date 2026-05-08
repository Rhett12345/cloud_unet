"""
validate_modis_cloud.py
========================
Simplified MODIS cross-validation: binary cloud detection + stratified CTH.

Instead of 3-class CLP comparison (which suffers from AGRI L2 / MODIS
phase-definition mismatch), this script evaluates:

  1. Cloud masking (clear vs cloudy) — POD, FAR, HSS
  2. CTH stratified by cloud height (low / mid / high)

Only high-confidence MODIS pixels are used (Confident Cloudy / Clear).

Can validate either model predictions (.npz) or AGRI L2 products (.NC)
against MODIS — controlled by the --reference flag.

Usage:
  # Model vs MODIS
  python validate_modis_cloud.py --npz_dir /path/to/retrieval/ --day 20190505

  # AGRI L2 vs MODIS (baseline)
  python validate_modis_cloud.py --day 20190505 --reference l2
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

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "modis_cloud_validation"

# CTH strata (meters)
CTH_LAYERS = {"Low": (0, 3000), "Mid": (3000, 8000), "High": (8000, 20000)}


def save_figure(fig, stem: str):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        (OUT_DIR / f"{stem}.{ext}").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# AGRI L2 readers
# ---------------------------------------------------------------------------

def read_l2_clp_raw(nc_path: Path) -> Optional[np.ndarray]:
    """Read L2 CLP as raw int16 (no remap)."""
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CLP"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.int16)
    ds.close()
    return raw

def read_l2_cth(nc_path: Path) -> Optional[np.ndarray]:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CTH"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.float32)
    ds.close()
    vmin, vmax = cfg.AGRI_L2_CTH_VALID_RANGE
    raw[(raw <= 0) | (raw >= 65500) | ~np.isfinite(raw)] = np.nan
    raw[raw < vmin] = np.nan; raw[raw > vmax] = np.nan
    return raw


# ---------------------------------------------------------------------------
# Binary cloud mask from CLP
# ---------------------------------------------------------------------------

def to_cloud_mask(clp: np.ndarray, source: str = "agri_l2") -> np.ndarray:
    """
    Convert CLP array to binary cloud mask: 0=clear, 1=cloudy, NaN=invalid.
    AGRI L2 raw:  0=Clear, 1=Water, 2=Supercooled, 3=Mixed, 4=Ice → cloudy
    AGRI L2 remapped (model output): 0=Clear, 1=Water, 2=Ice → cloudy
    MODIS: 0=Clear, 1=Water, 2=Ice → cloudy
    """
    mask = np.full(clp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(clp)
    if source == "agri_l2_raw":
        # Raw values 0=clear, 1-4=cloudy, others invalid
        mask[valid & (clp == 0)] = 0
        mask[valid & (clp >= 1) & (clp <= 4)] = 1
    else:
        # Already remapped: 0=clear, >0=cloudy
        mask[valid & (clp == 0)] = 0
        mask[valid & (clp > 0)] = 1
    return mask


# ---------------------------------------------------------------------------
# MODIS matching (high-confidence only)
# ---------------------------------------------------------------------------

def match_modis_to_agri_grid(agri_lat, agri_lon, agri_dt, modis_files, myd03_files):
    """Match MODIS to AGRI grid with strict quality filtering."""
    cfg.MODIS_FILTER_WEAK_QUALITY = True
    cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 3)  # Confident only
    cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0,)

    modis_list = []
    for mf in modis_files:
        mf = Path(mf) if isinstance(mf, str) else mf
        m03 = find_matching_myd03(mf, myd03_files)
        geo = read_modis_geo_quick(mf, myd03_file=m03)
        if geo is None: continue
        mlat = geo.get("lat_1km") if geo.get("lat_1km") is not None else geo.get("lat_5km")
        mlon = geo.get("lon_1km") if geo.get("lon_1km") is not None else geo.get("lon_5km")
        if mlat is None or mlon is None: continue
        if not check_modis_in_agri_disk(mlat, mlon, agri_lat, agri_lon): continue
        m = read_myd06(mf, agri_dt=agri_dt, myd03_file=m03, geo_cache=geo)
        if m is None: continue
        mdt = parse_modis_datetime(mf.name)
        if mdt: m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
        modis_list.append(m)

    if not modis_list: return None
    labels = aggregate_modis_to_agri(agri_lat, agri_lon, modis_list)
    if labels is None: return None
    vza = np.zeros_like(agri_lat); sza = np.zeros_like(agri_lat)
    return apply_quality_filter({"VZA": vza, "SZA": sza}, labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cloud_detection_metrics(cmask_ref: np.ndarray, cmask_test: np.ndarray) -> dict:
    """
    Binary cloud detection metrics (ref = AGRI L2 or model, test = MODIS).
    """
    valid = np.isfinite(cmask_ref) & np.isfinite(cmask_test)
    ref = cmask_ref[valid]
    tst = cmask_test[valid]
    n = len(ref)
    if n == 0:
        return {"n_cloud": 0}

    tp = int(((ref == 1) & (tst == 1)).sum())
    tn = int(((ref == 0) & (tst == 0)).sum())
    fp = int(((ref == 0) & (tst == 1)).sum())
    fn = int(((ref == 1) & (tst == 0)).sum())

    pod = tp / max(tp + fn, 1)           # hit rate
    far = fp / max(tp + fp, 1)           # false alarm ratio
    oa  = (tp + tn) / max(n, 1) * 100    # overall accuracy
    # Heidke Skill Score
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / max(denom, 1)

    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "pod": pod, "far": far, "oa_cloud": oa, "hss": hss}


def cth_stratified_metrics(cth_ref: np.ndarray, cth_test: np.ndarray,
                           cmask_ref: np.ndarray, cmask_test: np.ndarray) -> dict:
    """CTH metrics stratified by reference cloud height."""
    cloudy = (np.isfinite(cmask_ref) & np.isfinite(cmask_test) &
              (cmask_ref == 1) & (cmask_test == 1) &
              np.isfinite(cth_ref) & np.isfinite(cth_test))
    p, t = cth_ref[cloudy], cth_test[cloudy]

    result = {"n_cth_total": int(cloudy.sum())}
    if result["n_cth_total"] < 10:
        return result

    # Overall
    result["cth_r_all"] = _pearson_r(p, t)
    result["cth_rmse_all"] = float(np.sqrt(np.mean((p - t) ** 2)))
    result["cth_bias_all"] = float(np.mean(t - p))

    for name, (lo, hi) in CTH_LAYERS.items():
        mask = (p >= lo) & (p < hi)
        n_layer = int(mask.sum())
        result[f"n_{name}"] = n_layer
        if n_layer > 10:
            pl, tl = p[mask], t[mask]
            result[f"cth_r_{name}"] = _pearson_r(pl, tl)
            result[f"cth_rmse_{name}"] = float(np.sqrt(np.mean((pl - tl) ** 2)))
            result[f"cth_bias_{name}"] = float(np.mean(tl - pl))
            result[f"cth_mean_ref_{name}"] = float(np.mean(pl))
            result[f"cth_mean_modis_{name}"] = float(np.mean(tl))
        else:
            for k in ["cth_r", "cth_rmse", "cth_bias", "cth_mean_ref", "cth_mean_modis"]:
                result[f"{k}_{name}"] = np.nan

    return result


def _pearson_r(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / max(denom, 1e-12))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_stratified_cth(cth_ref, cth_test, cmask_ref, cmask_test, metrics, scene_id):
    """Multi-panel figure: scatter per layer + summary table."""
    cloudy = (np.isfinite(cmask_ref) & np.isfinite(cmask_test) &
              (cmask_ref == 1) & (cmask_test == 1) &
              np.isfinite(cth_ref) & np.isfinite(cth_test))
    p_all, t_all = cth_ref[cloudy], cth_test[cloudy]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    colors = {"Low": "#2E86AB", "Mid": "#A23B72", "High": "#F18F01"}

    for ax, (name, (lo, hi)) in zip(axes[:3], CTH_LAYERS.items()):
        mask = (p_all >= lo) & (p_all < hi)
        if mask.sum() > 10:
            idx = np.random.choice(mask.sum(), min(5000, mask.sum()), replace=False)
            p, t = p_all[mask][idx], t_all[mask][idx]
        else:
            p, t = p_all[mask], t_all[mask]
        ax.scatter(t, p, s=0.4, alpha=0.35, color=colors[name], rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{name} ({lo/1000:.0f}–{hi/1000:.0f} km)", fontsize=6.5, fontweight="bold",
                     color=colors[name])
        ax.set_xlabel("MODIS CTH (m)", fontsize=6)
        if name == "Low": ax.set_ylabel("Ref CTH (m)", fontsize=6)

    # Summary table panel
    ax = axes[3]; ax.axis("off")
    lines = [f"  {scene_id}", "", "Layer    R      RMSE    Bias"]
    for name in ["Low", "Mid", "High"]:
        r = metrics.get(f"cth_r_{name}", np.nan)
        rmse = metrics.get(f"cth_rmse_{name}", np.nan)
        bias = metrics.get(f"cth_bias_{name}", np.nan)
        n = metrics.get(f"n_{name}", 0)
        lines.append(f"  {name:5s}  {r:+.3f}  {rmse:4.0f}m  {bias:+.0f}m  n={n}")
    lines += ["", f"  ALL   {metrics.get('cth_r_all',0):+.3f}  "
              f"{metrics.get('cth_rmse_all',0):4.0f}m  "
              f"{metrics.get('cth_bias_all',0):+.0f}m"]
    ax.text(0, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=5.5, va="top", fontfamily="monospace")

    fig.suptitle(f"CTH stratified validation — {scene_id}", fontsize=7, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------------

def validate_one(ref_clp: np.ndarray, ref_cth: np.ndarray,
                 agri_lat: np.ndarray, agri_lon: np.ndarray,
                 agri_dt, modis_files: list, myd03_files: list,
                 scene_id: str, ref_type: str = "model") -> dict:
    """Run cloud-mask + stratified-CTH validation for one scene."""

    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_modis", "n_granules": 0}

    labels = match_modis_to_agri_grid(agri_lat, agri_lon, agri_dt, matched, myd03_files)
    if labels is None:
        return {"scene_id": scene_id, "status": "match_failed", "n_granules": len(matched)}

    # MODIS cloud mask & CTH
    modis_clp = labels["CLP"].astype(np.float32)
    modis_clp[(modis_clp < 0) | (modis_clp >= 3)] = np.nan
    modis_cmask = to_cloud_mask(modis_clp, "modis")
    modis_cth = labels["CTH"]

    # Reference cloud mask
    if ref_type == "agri_l2_raw":
        ref_cmask = to_cloud_mask(ref_clp, "agri_l2_raw")
    else:
        ref_cmask = to_cloud_mask(ref_clp, "model")  # 0=clear, 1/2=cloudy

    assert ref_clp.shape == modis_clp.shape, f"Shape mismatch"

    # Compute metrics
    cloud_m = cloud_detection_metrics(ref_cmask, modis_cmask)
    cth_m = cth_stratified_metrics(ref_cth, modis_cth, ref_cmask, modis_cmask)
    metrics = {**cloud_m, **cth_m, "scene_id": scene_id, "status": "ok",
               "n_granules": len(matched)}

    layers_str = " ".join(
        f"{n}={metrics.get(f'cth_r_{n}',np.nan):+.2f}" for n in ["Low","Mid","High"]
    )
    log.info("  %s  POD=%.3f FAR=%.3f HSS=%.3f  CTH R: %s",
             scene_id, cloud_m.get("pod", 0), cloud_m.get("far", 0),
             cloud_m.get("hss", 0), layers_str)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL),
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Cloud-mask + stratified-CTH MODIS validation")
    parser.add_argument("--day", required=True)
    parser.add_argument("--npz_dir", default=None, help="Model inference .npz directory")
    parser.add_argument("--reference", choices=["model", "l2"], default="model",
                        help="model = compare model .npz vs MODIS; l2 = AGRI L2 vs MODIS baseline")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    day = args.day
    agri_day_dir = cfg.AGRI_ROOT / day
    if not agri_day_dir.is_dir():
        log.error("AGRI L1B dir not found: %s", agri_day_dir)
        sys.exit(1)

    agri_files = sorted([f for f in agri_day_dir.glob("*_FDI-_*.HDF")
                         if not f.name.endswith(".db")])

    # MODIS files
    modis_dir = cfg.MODIS_ROOT / day
    myd03_dir = cfg.MYD03_ROOT / day
    all_modis = sorted(list(modis_dir.glob("MYD06*.hdf")) + list(modis_dir.glob("MYD06*.HDF"))) if modis_dir.is_dir() else []
    all_myd03 = sorted(list(myd03_dir.glob("MYD03*.hdf")) + list(myd03_dir.glob("MYD03*.HDF"))) if myd03_dir.is_dir() else []

    # Model .npz index (if applicable)
    npz_index = {}
    if args.reference == "model" and args.npz_dir:
        for npz_p in Path(args.npz_dir).rglob(f"*{day}*.npz"):
            ts = _extract_timestamp_from_filename(npz_p.name)
            if ts: npz_index[ts] = npz_p

    log.info("Day %s: %d AGRI scenes, %d MODIS, %d MYD03, ref=%s",
             day, len(agri_files), len(all_modis), len(all_myd03), args.reference)

    OUTPUT_DIR = OUT_DIR
    all_metrics = []
    for agri_f in agri_files:
        ts = _extract_timestamp_from_filename(agri_f.name)
        if ts is None: continue
        scene_id = f"{ts[:8]}_{ts[8:]}"

        # Get reference CLP/CTH
        if args.reference == "model":
            if ts not in npz_index: continue
            data = np.load(npz_index[ts])
            ref_clp = data["CLP_pred"].astype(np.float32)
            ref_cth = data["CTH_pred"].astype(np.float32)
            data.close()
            ref_type = "model"
        else:  # l2
            dummy = Path(f"FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_{ts}_x_4000M_V0001.HDF")
            clp_nc = _find_matching_l2_file(dummy, "CLP")
            cth_nc = _find_matching_l2_file(dummy, "CTH")
            if clp_nc is None or cth_nc is None: continue
            ref_clp_raw = read_l2_clp_raw(clp_nc)
            ref_cth = read_l2_cth(cth_nc)
            if ref_clp_raw is None or ref_cth is None: continue
            ref_clp = ref_clp_raw  # keep raw for cloud mask
            ref_type = "agri_l2_raw"

        # Read AGRI geo
        agri = read_agri_scene(agri_f)
        if agri is None: continue
        agri_dt = parse_agri_datetime(agri_f.name)

        try:
            m = validate_one(ref_clp, ref_cth, agri["lat"], agri["lon"],
                           agri_dt, all_modis, all_myd03, scene_id, ref_type)
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", scene_id, exc)

    # Summary
    if all_metrics and args.summary:
        import pandas as pd
        rows = [{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                for m in all_metrics]
        df = pd.DataFrame(rows)
        tag = "model" if args.reference == "model" else "l2"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"cloud_validation_{tag}_{day}.csv", index=False)

        ok = [m for m in all_metrics if m.get("status") == "ok"]
        if ok:
            pod_vals = [m["pod"] for m in ok if m.get("pod", 0) > 0]
            far_vals = [m["far"] for m in ok]
            hss_vals = [m["hss"] for m in ok]
            log.info("=== Cloud detection: %s vs MODIS  %s (n=%d) ===", tag, day, len(ok))
            log.info("POD:   %.3f ± %.3f", np.mean(pod_vals), np.std(pod_vals))
            log.info("FAR:   %.3f ± %.3f", np.mean(far_vals), np.std(far_vals))
            log.info("HSS:   %.3f ± %.3f", np.mean(hss_vals), np.std(hss_vals))
            for layer in ["Low", "Mid", "High"]:
                r_vals = [m[f"cth_r_{layer}"] for m in ok
                          if m.get(f"n_{layer}", 0) > 10 and np.isfinite(m.get(f"cth_r_{layer}", np.nan))]
                rmse_vals = [m[f"cth_rmse_{layer}"] for m in ok
                             if m.get(f"n_{layer}", 0) > 10 and np.isfinite(m.get(f"cth_rmse_{layer}", np.nan))]
                if r_vals:
                    log.info("CTH %s: R=%+.3f±%.3f  RMSE=%.0f±%.0f m  (n_scenes=%d)",
                             layer, np.mean(r_vals), np.std(r_vals),
                             np.mean(rmse_vals), np.std(rmse_vals), len(r_vals))


if __name__ == "__main__":
    main()
