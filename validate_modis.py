"""
validate_modis.py — MODIS 5km CTH cross-validation (simplified, v2)
====================================================================
Matches MODIS MYD06 5km Cloud_Top_Height to AGRI 4km grid using
simple KD-tree nearest-neighbour。Aligned with the original
MODISCOMPmatched.py approach: 5km SDS only, CTH only, no cloud detection.

All required functions are inlined — no dependency on fusion_io / fusion_core.

Usage:
  python validate_modis.py --npz_dir /path/to/retrieval/ --day 20190503
"""

import argparse, logging, re, sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pyhdf.SD import SD, SDC

import config as cfg

log = logging.getLogger(__name__)

# ── matplotlib rc ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "modis_validation_v2"
TIME_WINDOW_MIN = 5.0        # ±5 min temporal matching (tighter than original 15 min)
SEARCH_RADIUS_KM = 4.0       # 4 km search radius (aligned w/ original)
MAX_DIST_DEG = SEARCH_RADIUS_KM / 111.0  # approximate degree conversion
AGRI_SUB_LON = 104.7         # AGRI sub-satellite longitude (FY-4A)
AGRI_DISK_MAX_ANGLE = 75.0   # max angular distance from sub-satellite point (deg)


# ═════════════════════════════════════════════════════════════════════════════
# Inlined utility functions (from fusion_io / fusion_core — don't touch those)
# ═════════════════════════════════════════════════════════════════════════════

def parse_agri_datetime(filename: str) -> Optional[datetime]:
    """Extract datetime from AGRI FDI filename."""
    m = re.search(r"(\d{8})(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def parse_modis_datetime(filename: str) -> Optional[datetime]:
    """Extract datetime from MODIS filename: MYD06_L2.AYYYYDDD.HHMM.*"""
    m = re.search(r"\.A(\d{7})\.(\d{4})\.", filename)
    if m:
        try:
            year = int(m.group(1)[:4])
            doy  = int(m.group(1)[4:])
            hhmm = m.group(2)
            dt   = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return dt.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]))
        except (ValueError, IndexError):
            pass
    return None


def find_matching_modis(agri_dt: datetime, modis_files: list,
                         time_window_min: float = TIME_WINDOW_MIN) -> list:
    """Return MODIS files whose granule time is within ±time_window_min of agri_dt."""
    td = timedelta(minutes=time_window_min)
    candidates = []
    for f in modis_files:
        mdt = parse_modis_datetime(f.name)
        if mdt and abs(mdt - agri_dt) <= td:
            candidates.append((abs(mdt - agri_dt), f))
    return [f for _, f in sorted(candidates)]


def _extract_timestamp(filename: str) -> Optional[str]:
    """Extract YYYYMMDDHHMMSS from FY-4A filename."""
    m = re.search(r"_NOM_(\d{8})(\d{6})_", filename)
    if m:
        return m.group(1) + m.group(2)
    m = re.search(r"(\d{14})", filename)
    return m.group(1) if m else None


def _h5_first(hf: h5py.File, candidates: list) -> np.ndarray:
    """Return first matching dataset from HDF5 file."""
    for name in candidates:
        if name in hf:
            return hf[name][()]
    raise KeyError(f"None of {candidates} found")


def _attr_scalar(obj, key: str, default=None):
    v = obj.attrs.get(key, default)
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0] if v.size else default
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode()
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default


def _dataset_scaled(ds: h5py.Dataset) -> np.ndarray:
    arr = ds[()].astype(np.float64)
    fv = _attr_scalar(ds, "FillValue")
    if fv is not None:
        arr[arr == fv] = np.nan
    return arr * _attr_scalar(ds, "Slope", 1.0) + _attr_scalar(ds, "Intercept", 0.0)


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)


def _derive_latlon(gf: h5py.File):
    """Compute lat/lon from LineNumber/ColumnNumber + orbital params (AGRI GEO)."""
    line = _dataset_scaled(gf["LineNumber"])
    col  = _dataset_scaled(gf["ColumnNumber"])
    lon0_deg = _attr_scalar(gf, "NOMCenterLon")
    sat_h    = _attr_scalar(gf, "NOMSatHeight")
    ea_attr  = _attr_scalar(gf, "dEA")
    flat_inv = _attr_scalar(gf, "dObRecFlat")
    samp_ang = _attr_scalar(gf, "dSamplingAngle")
    step_ang = _attr_scalar(gf, "dSteppingAngle")

    if None in [lon0_deg, sat_h, ea_attr, flat_inv, samp_ang, step_ang]:
        raise KeyError("Missing GEO orbital params")

    ea_km    = ea_attr / 1000.0 if ea_attr > 1e5 else ea_attr
    sat_h_km = sat_h  / 1000.0 if sat_h   > 1e5 else sat_h
    H_sat    = sat_h_km + ea_km if sat_h_km < 40000 else sat_h_km
    eb_km    = ea_km * (1.0 - 1.0 / flat_inv)

    line[~np.isfinite(line) | (line < 0)] = np.nan
    col[ ~np.isfinite(col)  | (col  < 0)] = np.nan

    begin_pixel = _attr_scalar(gf, "Begin Pixel Number", 0.0)
    end_pixel   = _attr_scalar(gf, "End Pixel Number", 2747.0)
    begin_line  = _attr_scalar(gf, "Begin Line Number", 0.0)
    end_line    = _attr_scalar(gf, "End Line Number", 2747.0)
    coff = (begin_pixel + end_pixel) / 2.0
    loff = (begin_line + end_line) / 2.0

    H, W = line.shape
    mid_row = H // 2
    mid_col = W // 2
    col_vals  = col[mid_row, :][np.isfinite(col[mid_row, :])]
    line_vals = line[:, mid_col][np.isfinite(line[:, mid_col])]
    col_step  = float(np.nanmedian(np.diff(np.unique(col_vals))))  if len(col_vals)  > 1 else -1.0
    line_step = float(np.nanmedian(np.diff(np.unique(line_vals)))) if len(line_vals) > 1 else -1.0

    x_pix = samp_ang * 1e-6 / (col_step  if abs(col_step)  > 1.5 else 1.0)
    y_pix = step_ang * 1e-6 / (line_step if abs(line_step) > 1.5 else 1.0)

    x = (col - coff) * x_pix
    y = (loff - line) * y_pix
    lon0 = np.deg2rad(lon0_deg)

    ea2, eb2 = ea_km**2, eb_km**2
    cosx, sinx = np.cos(x), np.sin(x)
    cosy, siny = np.cos(y), np.sin(y)
    a = sinx**2 + cosx**2 * (cosy**2 + (ea2/eb2) * siny**2)
    b = -2.0 * H_sat * cosx * cosy
    c = H_sat**2 - ea2
    disc = b**2 - 4.0 * a * c

    lat = np.full(line.shape, np.nan, np.float64)
    lon = np.full(line.shape, np.nan, np.float64)
    valid = np.isfinite(disc) & (disc >= 0.0)
    if valid.any():
        sd  = np.sqrt(disc[valid])
        sn  = (-b[valid] - sd) / (2.0 * a[valid])
        s1  = H_sat - sn * cosx[valid] * cosy[valid]
        s2  = sn * sinx[valid] * cosy[valid]
        s3  = -sn * siny[valid]
        sxy = np.sqrt(s1**2 + s2**2)
        lat[valid] = np.rad2deg(np.arctan((ea2/eb2) * s3 / sxy))
        lon[valid] = np.rad2deg(np.arctan2(s2, s1) + lon0)

    bad = (lat < -90) | (lat > 90)
    lat[bad] = np.nan
    lon[bad] = np.nan
    return lat.astype(np.float32), _wrap_lon(lon)


def read_agri_geo(agri_file: Path) -> Optional[dict]:
    """
    Read AGRI lat/lon from paired GEO file.
    Falls back to _derive_latlon if precomputed lat/lon not found.
    """
    try:
        if "_FDI-_" not in agri_file.name:
            return None
        geo_file = Path(str(agri_file).replace("_FDI-_", "_GEO-_"))
        if not geo_file.exists():
            log.warning("GEO file missing for %s", agri_file.name)
            return None

        with h5py.File(geo_file, "r") as gf:
            try:
                lat = _h5_first(gf, ["Geolocation/NOMLatitude", "NOMLatitude", "Latitude"]).astype(np.float32)
                lon = _h5_first(gf, ["Geolocation/NOMLongitude", "NOMLongitude", "Longitude"]).astype(np.float32)
            except KeyError:
                lat, lon = _derive_latlon(gf)

        # basic cleaning
        for arr in [lat, lon]:
            arr[(arr > 1e4) | (arr < -1e4)] = np.nan
        lon = _wrap_lon(lon)

        return {"lat": lat, "lon": lon}
    except Exception as exc:
        log.warning("read_agri_geo failed %s: %s", agri_file, exc)
        return None


def read_myd06_cth_5km(modis_file: Path) -> Optional[dict]:
    """
    Read MODIS MYD06 5km CTH + geolocation.
    Reads three 5km SDS: Latitude, Longitude, Cloud_Top_Height.
    CTH is already in metres (scale_factor=1.0 in this product version).
    No MYD03, no 1km data, no cloud mask / phase.
    """
    try:
        sd = SD(str(modis_file), SDC.READ)

        lat = sd.select("Latitude")[:].astype(np.float32)
        lon = sd.select("Longitude")[:].astype(np.float32)
        cth_raw = sd.select("Cloud_Top_Height")[:].astype(np.float32)
        fv = sd.select("Cloud_Top_Height").attributes().get("_FillValue", -9999)
        sd.end()

        cth_raw[cth_raw == fv] = np.nan
        cth = cth_raw.astype(np.float32)
        cth[(cth <= 0) | (cth > 20000)] = np.nan

        # filter invalid geo
        lat[~np.isfinite(lat)] = np.nan
        lon[~np.isfinite(lon)] = np.nan

        return {"lat": lat, "lon": lon, "CTH": cth}
    except Exception as exc:
        log.warning("read_myd06_cth_5km failed %s: %s", modis_file, exc)
        return None


def _modis_in_agri_disk(modis_lat: np.ndarray, modis_lon: np.ndarray,
                       max_angle_deg: float = AGRI_DISK_MAX_ANGLE) -> float:
    """Return fraction of valid MODIS pixels within AGRI Earth disk."""
    valid = np.isfinite(modis_lat) & np.isfinite(modis_lon)
    if not valid.any():
        return 0.0
    lat_r, lon_r = np.deg2rad(modis_lat[valid]), np.deg2rad(modis_lon[valid])
    sub_lat_r = 0.0
    sub_lon_r = np.deg2rad(AGRI_SUB_LON)
    cos_angle = (np.sin(sub_lat_r) * np.sin(lat_r) +
                 np.cos(sub_lat_r) * np.cos(lat_r) * np.cos(lon_r - sub_lon_r))
    angle = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
    return float(np.mean(angle <= max_angle_deg))


def match_modis_cth_to_agri(
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    modis_files: list,
) -> Optional[dict]:
    """
    Match MODIS 5km CTH → AGRI 4km grid (k=1 nearest neighbour).
    Aligned with original MODISCOMPmatched.py approach.
    """
    # ── read & concat MODIS granules (only those fully within AGRI disk) ──
    all_lat, all_lon, all_cth = [], [], []
    n_granules, n_skipped = 0, 0
    for mf in modis_files:
        d = read_myd06_cth_5km(mf)
        if d is None:
            continue
        frac = _modis_in_agri_disk(d["lat"], d["lon"])
        if frac < 0.95:
            n_skipped += 1
            continue
        all_lat.append(d["lat"])
        all_lon.append(d["lon"])
        all_cth.append(d["CTH"])
        n_granules += 1
    if n_skipped:
        log.info("  Skipped %d MODIS granules not fully in AGRI disk", n_skipped)

    if not all_lat:
        return None

    modis_lat = np.concatenate(all_lat, axis=0)
    modis_lon = np.concatenate(all_lon, axis=0)
    modis_cth = np.concatenate(all_cth, axis=0)

    # ── filter valid MODIS pixels ──
    valid_m = (
        np.isfinite(modis_lat) & np.isfinite(modis_lon) &
        np.isfinite(modis_cth)
    )
    if not valid_m.any():
        return None

    # ── KD-tree: MODIS → AGRI (k=1) ──
    modis_pts = np.column_stack((
        modis_lat[valid_m].ravel(),
        modis_lon[valid_m].ravel(),
    ))
    tree = cKDTree(modis_pts)

    agri_shape = agri_lat.shape
    agri_valid = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    agri_pts = np.column_stack((
        agri_lat[agri_valid].ravel(),
        agri_lon[agri_valid].ravel(),
    ))

    distances, indices = tree.query(agri_pts, k=1, distance_upper_bound=MAX_DIST_DEG)
    valid_match = distances < MAX_DIST_DEG

    # ── build output ──
    out_cth = np.full(agri_shape, np.nan, dtype=np.float32)
    out_dist = np.full(agri_shape, np.nan, dtype=np.float32)

    agri_valid_flat = agri_valid.ravel()
    agri_valid_idx = np.where(agri_valid_flat)[0]
    matched_agri_idx = agri_valid_idx[valid_match]
    modis_matched = indices[valid_match]

    # map back through valid_m mask
    valid_m_idx = np.where(valid_m.ravel())[0]
    out_cth.ravel()[matched_agri_idx] = modis_cth.ravel()[valid_m_idx[modis_matched]]
    out_dist.ravel()[matched_agri_idx] = distances[valid_match] * 111.0  # deg → km

    n_matched = int(valid_match.sum())
    log.info("  MODIS match: %d AGRI pixels matched from %d granules  "
             "coverage=%.1f%%  dist_median=%.1f km",
             n_matched, n_granules,
             100.0 * n_matched / max(agri_valid.sum(), 1),
             float(np.median(out_dist[np.isfinite(out_dist)])) if n_matched > 0 else 0)

    return {"CTH": out_cth, "MATCH_DIST_KM": out_dist, "n_granules": n_granules}


# ═════════════════════════════════════════════════════════════════════════════
# CTH metrics
# ═════════════════════════════════════════════════════════════════════════════

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / max(denom, 1e-12))


def cth_metrics(cth_pred: np.ndarray, cth_true: np.ndarray) -> dict:
    """Compute CTH regression metrics."""
    valid = np.isfinite(cth_pred) & np.isfinite(cth_true)
    n = int(valid.sum())
    if n < 10:
        return {"n_cth": n, "cth_r": 0, "cth_rmse": 0, "cth_mae": 0, "cth_bias": 0}

    p, t = cth_pred[valid], cth_true[valid]
    cth_rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    cth_mae  = float(np.mean(np.abs(p - t)))
    cth_bias = float(np.mean(p - t))
    cth_r    = _pearson_r(p, t)
    return {"n_cth": n, "cth_r": cth_r, "cth_rmse": cth_rmse,
            "cth_mae": cth_mae, "cth_bias": cth_bias}


# ═════════════════════════════════════════════════════════════════════════════
# Figure
# ═════════════════════════════════════════════════════════════════════════════

def save_figure(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")


def _plot_cth_scatter(cth_pred, cth_true, metrics, scene_id):
    valid = np.isfinite(cth_pred) & np.isfinite(cth_true)
    p, t = cth_pred[valid], cth_true[valid]
    if len(p) > 5000:
        idx = np.random.choice(len(p), 5000, replace=False)
        p, t = p[idx], t[idx]

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(t, p, s=0.6, alpha=0.35, color="#2E86AB", rasterized=True)
    ax.plot([0, 20000], [0, 20000], "k--", lw=0.6, alpha=0.4)

    r    = metrics.get("cth_r", 0)
    rmse = metrics.get("cth_rmse", 0)
    bias = metrics.get("cth_bias", 0)
    n    = metrics.get("n_cth", 0)
    ax.text(0.03, 0.95,
            f"{scene_id}\n"
            f"R = {r:+.3f}  RMSE = {rmse:.0f} m  Bias = {bias:+.0f} m  n = {n}",
            transform=ax.transAxes, fontsize=6.5, va="top", fontfamily="monospace")

    ax.set_xlabel("MODIS CTH (m)", fontsize=7)
    ax.set_ylabel("Predicted CTH (m)", fontsize=7)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 20000)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Per-scene validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_one(
    npz_path: Path,
    out_dir: Path,
    scene_id: str = "",
) -> dict:
    """Run MODIS 5km CTH cross-validation for one scene."""
    # 1. Read model predictions + lat/lon from npz (same as original approach)
    data = np.load(npz_path)
    cth_pred = data["CTH_pred"].astype(np.float32)
    lat = data["latitude"].astype(np.float32)
    lon = data["longitude"].astype(np.float32)
    data.close()

    # 2. Parse datetime from npz filename
    agri_dt = parse_agri_datetime(npz_path.name)
    if agri_dt is None:
        return {"scene_id": scene_id, "status": "bad_datetime"}

    # 3. Find matching MODIS files
    day_str = agri_dt.strftime("%Y%m%d")
    modis_day_dir = cfg.MODIS_ROOT / day_str
    if not modis_day_dir.is_dir():
        return {"scene_id": scene_id, "status": "no_modis_dir"}

    modis_files = sorted(
        list(modis_day_dir.glob("MYD06*.hdf")) +
        list(modis_day_dir.glob("MYD06*.HDF"))
    )
    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_matching_modis"}

    # 4. Match MODIS 5km CTH → AGRI grid
    labels = match_modis_cth_to_agri(lat, lon, matched)
    if labels is None:
        return {"scene_id": scene_id, "status": "modis_matching_failed"}

    cth_true = labels["CTH"]

    # 5. Verify shape
    if cth_pred.shape != cth_true.shape:
        log.warning("Shape mismatch: pred=%s true=%s", cth_pred.shape, cth_true.shape)
        return {"scene_id": scene_id, "status": "shape_mismatch"}

    # 6. Metrics
    metrics = cth_metrics(cth_pred, cth_true)
    metrics["scene_id"] = scene_id
    metrics["status"] = "ok"
    metrics["n_granules"] = labels.get("n_granules", 0)

    log.info("  %s  CTH R=%+.3f  RMSE=%.0f m  Bias=%+.0f m  n=%d",
             scene_id, metrics["cth_r"], metrics["cth_rmse"],
             metrics["cth_bias"], metrics["n_cth"])

    # 7. Scatter plot
    if metrics["n_cth"] > 10:
        fig = _plot_cth_scatter(cth_pred, cth_true, metrics, scene_id)
        stem = scene_id.replace(" ", "_")
        save_figure(fig, f"{stem}_modis_cth_5km")
        plt.close(fig)

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="MODIS 5km CTH cross-validation (simplified)"
    )
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    npz_files = sorted(Path(args.npz_dir).rglob(f"*{args.day}*.npz"))
    if not npz_files:
        log.error("No npz files found for day %s in %s", args.day, args.npz_dir)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for npz_p in npz_files:
        ts = _extract_timestamp(npz_p.name)
        if ts is None:
            continue
        scene_id = f"{ts[:8]}_{ts[8:]}"

        try:
            m = validate_one(npz_p, out_dir, scene_id)
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", scene_id, exc)

    # ── Summary ──
    if all_metrics and args.summary:
        import pandas as pd
        rows = [
            {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
            for m in all_metrics
        ]
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"summary_modis_cth_5km_{args.day}.csv", index=False)

        ok = [m for m in all_metrics if m.get("status") == "ok" and m.get("n_cth", 0) > 10]
        if ok:
            r_vals    = [m["cth_r"] for m in ok]
            rmse_vals = [m["cth_rmse"] for m in ok]
            bias_vals = [m["cth_bias"] for m in ok]
            n_vals    = [m["n_cth"]  for m in ok]
            log.info("=" * 60)
            log.info("MODIS 5km CTH validation summary: %s (n=%d scenes)", args.day, len(ok))
            log.info("  CTH R:     %+.4f ± %.4f", np.mean(r_vals), np.std(r_vals))
            log.info("  CTH RMSE:  %.0f ± %.0f m", np.mean(rmse_vals), np.std(rmse_vals))
            log.info("  CTH Bias:  %+.0f ± %.0f m", np.mean(bias_vals), np.std(bias_vals))
            log.info("  Total matched pixels: %d", sum(n_vals))


if __name__ == "__main__":
    main()
