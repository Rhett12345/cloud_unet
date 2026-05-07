"""
tools/validate_modis.py
========================
Cross-validate model predictions against MODIS MYD06 (independent sensor).

For each model inference .npz, find time-matched MYD06 granules,
spatially match MODIS labels to the AGRI grid via the existing fusion engine,
then compare pixel-by-pixel.

This is a true external validation — MODIS and FY4A use completely independent
retrieval algorithms, so agreement implies high confidence.

Usage:
  python tools/validate_modis.py --npz_dir /path/to/retrieval/ --day 20190503

Requires: MODIS MYD06 + MYD03 data under cfg.MODIS_ROOT / cfg.MYD03_ROOT.
"""

import argparse, logging, sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as cfg
import fusion_config as fc
from fusion_core import aggregate_modis_to_agri, check_modis_in_agri_disk
from fusion_io import (
    apply_quality_filter,
    parse_agri_datetime, parse_modis_datetime,
    find_matching_modis, find_matching_myd03,
    read_agri_scene, read_myd06, read_modis_geo_quick,
    _extract_timestamp_from_filename,
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

OUT_DIR = cfg.ROOT / "eval" / "modis_validation"


def save_figure(fig, stem: str):
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# MODIS label matching to AGRI grid
# ---------------------------------------------------------------------------

def match_modis_to_agri_grid(agri_lat: np.ndarray, agri_lon: np.ndarray,
                              agri_dt, modis_files: list,
                              myd03_files: list,
                              quality: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """
    Run the full MODIS→AGRI spatial matching pipeline for validation.
    When quality=True, only confident MODIS pixels are kept
    (Cloud_Mask → Confident Cloudy / Confident Clear only + time + geometry).
    """
    # Enable quality filtering for validation
    if quality:
        cfg.MODIS_FILTER_WEAK_QUALITY = True
        # Only keep most confident pixels for validation
        cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 3)  # Confident Cloudy, Confident Clear
        cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0,)     # Only Confident Cloudy for CTH

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

    # Apply post-fusion quality filter (time, geometry, value range)
    vza = np.zeros_like(agri_lat)
    sza = np.zeros_like(agri_lat)
    agri_geo = {"VZA": vza, "SZA": sza}
    labels = apply_quality_filter(agri_geo, labels)
    return labels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_one_modis(npz_path: Path, agri_file: Path,
                        out_dir: Path, scene_id: str = "") -> dict:
    """Run MODIS cross-validation for one scene."""
    # Read model predictions
    data = np.load(npz_path)
    clp_pred = data["CLP_pred"].astype(np.float32)
    cth_pred = data["CTH_pred"].astype(np.float32)
    data.close()

    # Read AGRI for geo coordinates and time
    agri = read_agri_scene(agri_file)
    if agri is None:
        raise RuntimeError(f"Failed to read AGRI scene {agri_file}")

    agri_dt = parse_agri_datetime(agri_file.name)
    if agri_dt is None:
        raise RuntimeError(f"Cannot parse datetime from {agri_file.name}")

    # Find matching MODIS
    day_str = agri_dt.strftime("%Y%m%d")
    modis_day_dir = cfg.MODIS_ROOT / day_str
    myd03_day_dir = cfg.MYD03_ROOT / day_str

    modis_files = []
    if modis_day_dir.is_dir():
        modis_files = sorted(
            list(modis_day_dir.glob("MYD06*.hdf")) +
            list(modis_day_dir.glob("MYD06*.HDF"))
        )
    myd03_files = []
    if myd03_day_dir.is_dir():
        myd03_files = sorted(
            list(myd03_day_dir.glob("MYD03*.hdf")) +
            list(myd03_day_dir.glob("MYD03*.HDF"))
        )

    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_matching_modis"}

    # Spatial matching
    labels = match_modis_to_agri_grid(
        agri["lat"], agri["lon"], agri_dt, matched, myd03_files
    )
    if labels is None:
        return {"scene_id": scene_id, "status": "modis_matching_failed"}

    clp_true = labels["CLP"]
    cth_true = labels["CTH"]

    # Remap MODIS CLP: 0=clear, 1=water, 2=ice (already handled by MODIS_PHASE_MAP)
    clp_true_f = clp_true.astype(np.float32)
    clp_true_f[(clp_true_f < 0) | (clp_true_f >= 3)] = np.nan

    # Ensure same shape
    assert clp_pred.shape == clp_true_f.shape, "Shape mismatch"

    from validate_agri_l2 import compute_metrics
    metrics = compute_metrics(clp_pred, clp_true_f, cth_pred, cth_true)
    metrics["scene_id"] = scene_id
    metrics["status"] = "ok"

    log.info("  %s  vs MODIS  CLP OA=%5.2f%%  CTH R=%.4f  RMSE=%.0f m",
             scene_id, metrics["oa"], metrics.get("cth_r", 0), metrics["cth_rmse"])

    # Simple scatter figure
    if metrics["n_cth"] > 10:
        from validate_agri_l2 import _plot_cth_scatter
        fig = _plot_cth_scatter(cth_pred, cth_true, metrics,
                                f"{scene_id} vs MODIS")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = scene_id.replace(" ", "_")
        save_figure(fig, f"{stem}_modis_cth_scatter")
        plt.close(fig)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Cross-validate model against MODIS MYD06")
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    npz_files = sorted(Path(args.npz_dir).rglob(f"*{args.day}*.npz"))
    agri_day_dir = cfg.AGRI_ROOT / args.day
    agri_files = {f.stem.replace("_retrieval", ""): f
                  for f in agri_day_dir.glob("*_FDI-_*.HDF")} if agri_day_dir.is_dir() else {}

    out_dir = Path(args.out_dir)
    all_metrics = []
    for npz_p in npz_files:
        ts = _extract_timestamp_from_filename(npz_p.name)
        if ts is None:
            continue
        scene_id = f"{ts[:8]}_{ts[8:]}"
        # Find matching AGRI FDI file
        agri_match = None
        for stem, agri_p in agri_files.items():
            if ts in stem:
                agri_match = agri_p
                break
        if agri_match is None:
            log.warning("No AGRI FDI file for timestamp %s", ts)
            continue
        try:
            m = validate_one_modis(npz_p, agri_match, out_dir, scene_id)
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", scene_id, exc)

    if all_metrics and args.summary:
        import pandas as pd
        rows = [{k: v for k, v in m.items() if k != "confusion_matrix"}
                for m in all_metrics]
        pd.DataFrame(rows).to_csv(out_dir / f"summary_modis_{args.day}.csv", index=False)
        ok = [m for m in all_metrics if m.get("status") == "ok"]
        if ok:
            oa_vals = [m["oa"] for m in ok if m["oa"] > 0]
            log.info("MODIS cross-val: n=%d  OA mean=%.2f%%", len(ok), np.mean(oa_vals) if oa_vals else 0)


if __name__ == "__main__":
    main()
