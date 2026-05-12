"""
validate_modis_cloud.py → validate_modis_cth_stratified.py (v2)
================================================================
Stratified CTH validation against MODIS MYD06 5km Cloud_Top_Height.
Matches Low (0-3km), Mid (3-8km), High (8-20km) layers.
Aligned with original MODISCOMPmatched.py: 5km SDS only, CTH only.

All required functions inlined — no dependency on fusion_io / fusion_core.

Usage:
  # Model vs MODIS
  python validate_modis_cloud.py --npz_dir /path/to/retrieval/ --day 20190505

  # AGRI L2 CTH vs MODIS (baseline)
  python validate_modis_cloud.py --day 20190505 --reference l2
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

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "modis_cth_stratified_v2"
TIME_WINDOW_MIN = 5.0
SEARCH_RADIUS_KM = 4.0
MAX_DIST_DEG = SEARCH_RADIUS_KM / 111.0
AGRI_SUB_LON = 104.7
AGRI_DISK_MAX_ANGLE = 75.0

CTH_LAYERS = {"Low": (0, 3000), "Mid": (3000, 8000), "High": (8000, 20000)}


# ═════════════════════════════════════════════════════════════════════════════
# Inlined utility functions
# ═════════════════════════════════════════════════════════════════════════════

def parse_agri_datetime(filename: str) -> Optional[datetime]:
    m = re.search(r"(\d{8})(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def parse_modis_datetime(filename: str) -> Optional[datetime]:
    m = re.search(r"\.A(\d{7})\.(\d{4})\.", filename)
    if m:
        try:
            year = int(m.group(1)[:4])
            doy  = int(m.group(1)[4:])
            dt   = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return dt.replace(hour=int(m.group(2)[:2]), minute=int(m.group(2)[2:]))
        except (ValueError, IndexError):
            pass
    return None


def find_matching_modis(agri_dt: datetime, modis_files: list,
                         time_window_min: float = TIME_WINDOW_MIN) -> list:
    td = timedelta(minutes=time_window_min)
    candidates = []
    for f in modis_files:
        mdt = parse_modis_datetime(f.name)
        if mdt and abs(mdt - agri_dt) <= td:
            candidates.append((abs(mdt - agri_dt), f))
    return [f for _, f in sorted(candidates)]


def _extract_timestamp(filename: str) -> Optional[str]:
    m = re.search(r"_NOM_(\d{8})(\d{6})_", filename)
    if m:
        return m.group(1) + m.group(2)
    m = re.search(r"(\d{14})", filename)
    return m.group(1) if m else None


def _h5_first(hf: h5py.File, candidates: list) -> np.ndarray:
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
        for arr in [lat, lon]:
            arr[(arr > 1e4) | (arr < -1e4)] = np.nan
        lon = _wrap_lon(lon)
        return {"lat": lat, "lon": lon}
    except Exception as exc:
        log.warning("read_agri_geo failed %s: %s", agri_file, exc)
        return None


def read_myd06_cth_5km(modis_file: Path) -> Optional[dict]:
    try:
        sd = SD(str(modis_file), SDC.READ)
        lat = sd.select("Latitude")[:].astype(np.float32)
        lon = sd.select("Longitude")[:].astype(np.float32)
        cth_raw = sd.select("Cloud_Top_Height")[:].astype(np.float32)
        fv = sd.select("Cloud_Top_Height").attributes().get("_FillValue", -9999)
        sd.end()
        cth_raw[cth_raw == fv] = np.nan
        cth = cth_raw.copy()
        cth[(cth <= 0) | (cth > 20000)] = np.nan
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

    valid_m = (np.isfinite(modis_lat) & np.isfinite(modis_lon) & np.isfinite(modis_cth))
    if not valid_m.any():
        return None

    modis_pts = np.column_stack((modis_lat[valid_m].ravel(), modis_lon[valid_m].ravel()))
    tree = cKDTree(modis_pts)

    agri_shape = agri_lat.shape
    agri_valid = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    agri_pts = np.column_stack((agri_lat[agri_valid].ravel(), agri_lon[agri_valid].ravel()))

    distances, indices = tree.query(agri_pts, k=1, distance_upper_bound=MAX_DIST_DEG)
    valid_match = distances < MAX_DIST_DEG

    out_cth = np.full(agri_shape, np.nan, dtype=np.float32)
    out_dist = np.full(agri_shape, np.nan, dtype=np.float32)

    agri_valid_idx = np.where(agri_valid.ravel())[0]
    matched_agri_idx = agri_valid_idx[valid_match]
    modis_matched = indices[valid_match]
    valid_m_idx = np.where(valid_m.ravel())[0]
    out_cth.ravel()[matched_agri_idx] = modis_cth.ravel()[valid_m_idx[modis_matched]]
    out_dist.ravel()[matched_agri_idx] = distances[valid_match] * 111.0

    n_matched = int(valid_match.sum())
    log.info("  MODIS match: %d pixels from %d granules  coverage=%.1f%%",
             n_matched, n_granules,
             100.0 * n_matched / max(agri_valid.sum(), 1))

    return {"CTH": out_cth, "MATCH_DIST_KM": out_dist, "n_granules": n_granules}


# ═════════════════════════════════════════════════════════════════════════════
# AGRI L2 CTH reader (for --reference l2 mode)
# ═════════════════════════════════════════════════════════════════════════════

def _find_l2_file(l1_fdi_path: Path, product: str) -> Optional[Path]:
    """Find matching L2 NetCDF file for a given L1B FDI file."""
    ts = _extract_timestamp(l1_fdi_path.name)
    if ts is None:
        return None
    date_str = ts[:8]
    l2_dir = cfg.FY4A_L2_ROOT / product / date_str
    if not l2_dir.is_dir():
        return None
    for f in sorted(l2_dir.iterdir()):
        if not f.name.endswith(".NC"):
            continue
        f_ts = _extract_timestamp(f.name)
        if f_ts == ts:
            return f
    return None


def read_agri_l2_cth(nc_path: Path) -> Optional[np.ndarray]:
    """Read AGRI L2 CTH NetCDF, return float32 2D array with fill-filtering."""
    import netCDF4 as nc
    try:
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
    except Exception as exc:
        log.warning("read_agri_l2_cth failed %s: %s", nc_path, exc)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def _pearson_r(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / max(denom, 1e-12))


def cth_stratified_metrics(cth_pred: np.ndarray, cth_true: np.ndarray) -> dict:
    """Stratified CTH metrics by cloud height layer."""
    valid = np.isfinite(cth_pred) & np.isfinite(cth_true)
    p, t = cth_pred[valid], cth_true[valid]
    n_total = len(p)

    result = {"n_cth_total": n_total}
    if n_total < 10:
        return result

    result["cth_r_all"] = _pearson_r(p, t)
    result["cth_rmse_all"] = float(np.sqrt(np.mean((p - t) ** 2)))
    result["cth_bias_all"] = float(np.mean(p - t))

    for name, (lo, hi) in CTH_LAYERS.items():
        mask = (t >= lo) & (t < hi)
        n_layer = int(mask.sum())
        result[f"n_{name}"] = n_layer
        if n_layer > 10:
            pl, tl = p[mask], t[mask]
            result[f"cth_r_{name}"]    = _pearson_r(pl, tl)
            result[f"cth_rmse_{name}"] = float(np.sqrt(np.mean((pl - tl) ** 2)))
            result[f"cth_bias_{name}"] = float(np.mean(pl - tl))
        else:
            for k in ["cth_r", "cth_rmse", "cth_bias"]:
                result[f"{k}_{name}"] = np.nan

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Figure
# ═════════════════════════════════════════════════════════════════════════════

def save_figure(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")


def _plot_stratified_cth(cth_pred, cth_true, metrics, scene_id):
    valid = np.isfinite(cth_pred) & np.isfinite(cth_true)
    p_all, t_all = cth_pred[valid], cth_true[valid]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    colors = {"Low": "#2E86AB", "Mid": "#A23B72", "High": "#F18F01"}

    for ax, (name, (lo, hi)) in zip(axes[:3], CTH_LAYERS.items()):
        mask = (t_all >= lo) & (t_all < hi)
        pts_p, pts_t = p_all[mask], t_all[mask]
        if mask.sum() > 5000:
            idx = np.random.choice(mask.sum(), 5000, replace=False)
            pts_p, pts_t = pts_p[idx], pts_t[idx]
        ax.scatter(pts_t, pts_p, s=0.4, alpha=0.35, color=colors[name], rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{name} ({lo//1000}–{hi//1000} km)",
                     fontsize=6.5, fontweight="bold", color=colors[name])
        ax.set_xlabel("MODIS CTH (m)", fontsize=6)
        if name == "Low":
            ax.set_ylabel("Ref CTH (m)", fontsize=6)

    # text panel
    ax = axes[3]; ax.axis("off")
    lines = [f"  {scene_id}", "", "Layer    R      RMSE    Bias"]
    for name in ["Low", "Mid", "High"]:
        r    = metrics.get(f"cth_r_{name}", np.nan)
        rmse = metrics.get(f"cth_rmse_{name}", np.nan)
        bias = metrics.get(f"cth_bias_{name}", np.nan)
        n    = metrics.get(f"n_{name}", 0)
        lines.append(f"  {name:5s}  {r:+.3f}  {rmse:4.0f}m  {bias:+.0f}m  n={n}")
    lines += ["", f"  ALL   {metrics.get('cth_r_all',0):+.3f}  "
              f"{metrics.get('cth_rmse_all',0):4.0f}m  "
              f"{metrics.get('cth_bias_all',0):+.0f}m"]
    ax.text(0, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=5.5, va="top", fontfamily="monospace")

    fig.suptitle(f"CTH stratified validation — {scene_id}",
                 fontsize=7, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Per-scene validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_one(
    ref_cth: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    agri_dt: datetime,
    modis_files: list,
    scene_id: str,
) -> dict:
    """Run stratified CTH validation for one scene."""
    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_modis", "n_granules": 0}

    labels = match_modis_cth_to_agri(agri_lat, agri_lon, matched)
    if labels is None:
        return {"scene_id": scene_id, "status": "match_failed", "n_granules": len(matched)}

    modis_cth = labels["CTH"]

    if ref_cth.shape != modis_cth.shape:
        log.warning("Shape mismatch: ref=%s modis=%s", ref_cth.shape, modis_cth.shape)
        return {"scene_id": scene_id, "status": "shape_mismatch"}

    metrics = cth_stratified_metrics(ref_cth, modis_cth)
    metrics["scene_id"] = scene_id
    metrics["status"] = "ok"
    metrics["n_granules"] = len(matched)

    layers_str = " ".join(
        f"{n}={metrics.get(f'cth_r_{n}', np.nan):+.2f}" for n in ["Low", "Mid", "High"]
    )
    log.info("  %s  CTH R: %s  n=%d",
             scene_id, layers_str, metrics.get("n_cth_total", 0))

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
        description="MODIS 5km stratified CTH validation"
    )
    parser.add_argument("--day", required=True)
    parser.add_argument("--npz_dir", default=None)
    parser.add_argument("--reference", choices=["model", "l2"], default="model")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    day = args.day
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # shared MODIS file listing
    modis_dir = cfg.MODIS_ROOT / day
    all_modis = sorted(
        list(modis_dir.glob("MYD06*.hdf")) + list(modis_dir.glob("MYD06*.HDF"))
    ) if modis_dir.is_dir() else []

    # model mode: iterate npz directly (use npz lat/lon)
    # l2 mode: iterate AGRI FDI files (derive geo from GEO)
    all_metrics = []

    if args.reference == "model":
        if not args.npz_dir:
            log.error("--npz_dir required for model reference mode")
            sys.exit(1)
        npz_files = sorted(Path(args.npz_dir).rglob(f"*{day}*.npz"))
        log.info("Day %s: %d npz files, %d MODIS, ref=model",
                 day, len(npz_files), len(all_modis))

        for npz_p in npz_files:
            ts = _extract_timestamp(npz_p.name)
            if ts is None:
                continue
            scene_id = f"{ts[:8]}_{ts[8:]}"

            try:
                data = np.load(npz_p)
                ref_cth = data["CTH_pred"].astype(np.float32)
                lat = data["latitude"].astype(np.float32)
                lon = data["longitude"].astype(np.float32)
                data.close()

                agri_dt = parse_agri_datetime(npz_p.name)
                if agri_dt is None:
                    continue

                m = validate_one(ref_cth, lat, lon, agri_dt, all_modis, scene_id)
                all_metrics.append(m)
            except Exception as exc:
                log.error("Failed %s: %s", scene_id, exc)
    else:
        # l2 reference mode: need AGRI FDI + GEO files
        agri_day_dir = cfg.AGRI_ROOT / day
        if not agri_day_dir.is_dir():
            log.error("AGRI L1B dir not found: %s", agri_day_dir)
            sys.exit(1)
        agri_files = sorted([
            f for f in agri_day_dir.glob("*_FDI-_*.HDF")
            if not f.name.endswith(".db")
        ])
        log.info("Day %s: %d AGRI scenes, %d MODIS, ref=l2",
                 day, len(agri_files), len(all_modis))

        for agri_f in agri_files:
            ts = _extract_timestamp(agri_f.name)
            if ts is None:
                continue
            scene_id = f"{ts[:8]}_{ts[8:]}"

            agri = read_agri_geo(agri_f)
            if agri is None:
                continue
            agri_dt = parse_agri_datetime(agri_f.name)
            if agri_dt is None:
                continue

            cth_nc = _find_l2_file(agri_f, "CTH")
            if cth_nc is None:
                continue
            ref_cth = read_agri_l2_cth(cth_nc)
            if ref_cth is None:
                continue

            try:
                m = validate_one(ref_cth, agri["lat"], agri["lon"],
                                 agri_dt, all_modis, scene_id)
                all_metrics.append(m)
            except Exception as exc:
                log.error("Failed %s: %s", scene_id, exc)
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
        tag = "model" if args.reference == "model" else "l2"
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"cth_stratified_{tag}_{day}.csv", index=False)

        ok = [m for m in all_metrics
              if m.get("status") == "ok" and m.get("n_cth_total", 0) > 10]
        if ok:
            r_vals    = [m["cth_r_all"] for m in ok]
            rmse_vals = [m["cth_rmse_all"] for m in ok]
            bias_vals = [m["cth_bias_all"] for m in ok]
            log.info("=" * 60)
            log.info("CTH stratified summary: %s vs MODIS  %s (n=%d scenes)",
                     tag, day, len(ok))
            log.info("  CTH R:     %+.4f ± %.4f", np.mean(r_vals), np.std(r_vals))
            log.info("  CTH RMSE:  %.0f ± %.0f m", np.mean(rmse_vals), np.std(rmse_vals))
            log.info("  CTH Bias:  %+.0f ± %.0f m", np.mean(bias_vals), np.std(bias_vals))
            for layer in ["Low", "Mid", "High"]:
                r_layer = [m[f"cth_r_{layer}"] for m in ok
                           if m.get(f"n_{layer}", 0) > 10
                           and np.isfinite(m.get(f"cth_r_{layer}", np.nan))]
                rmse_layer = [m[f"cth_rmse_{layer}"] for m in ok
                              if m.get(f"n_{layer}", 0) > 10
                              and np.isfinite(m.get(f"cth_rmse_{layer}", np.nan))]
                if r_layer:
                    log.info("  CTH %s: R=%+.3f±%.3f  RMSE=%.0f±%.0f m  (n=%d)",
                             layer, np.mean(r_layer), np.std(r_layer),
                             np.mean(rmse_layer), np.std(rmse_layer), len(r_layer))


if __name__ == "__main__":
    main()
