"""
data_fusion.py
==============
Fuses AGRI FDI/GEO (FY-4A/B) with MYD06 (Aqua) cloud-property labels.

For every AGRI scene it:
  1. Finds MYD06 granules within ±MAX_TIME_DIFF_MIN minutes.
  2. Reads AGRI thermal BT channels + geolocation + auxiliary fields.
  3. Reads MYD06 CLP/CER/COT/CTH via a kd-tree spatial match.
  4. Writes one HDF5 file per AGRI scene to PAIRED_{TRAIN/VAL/TEST}_DIR.
  5. [NEW] Generates a 4-panel QC diagnostic PNG for the first N scenes per day,
     so you can visually confirm that MODIS labels are correctly co-located with
     AGRI brightness temperatures.

QC figure layout (saved as <paired_h5_stem>_qc.png):
  ┌─────────────────────┬──────────────────────┐
  │ AGRI BT (ch8, K)    │ MODIS CTH (m)        │
  ├─────────────────────┼──────────────────────┤
  │ MODIS CLP (class)   │ BT vs CTH scatter    │
  │                     │ (matched pixels only) │
  └─────────────────────┴──────────────────────┘

Usage (called by main.py, or standalone):
    python data_fusion.py --split train --day 20230601
    python data_fusion.py --split train --day 20230601 --max_qc 5
"""

import argparse
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List

import h5py
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.spatial import cKDTree

import config as cfg

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_agri_datetime(filename: str) -> Optional[datetime]:
    """
    Extract observation datetime from an AGRI filename.
    Expected: any 14-digit block YYYYMMDDHHMMSS inside the name.
    """
    m = re.search(r"(\d{8})(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def _parse_modis_datetime(filename: str) -> Optional[datetime]:
    """MYD06_L2.AYYYYDDD.HHMM.CCC.yyyydddhhmmss.hdf"""
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


def _find_day_folders(root: Path, dates: list) -> list:
    """Return list of day-directories to process."""
    if dates:
        return [root / d for d in dates if (root / d).is_dir()]
    return sorted(p for p in root.iterdir() if p.is_dir())


# def _find_matching_modis(agri_dt: datetime, modis_files: list) -> list:
#     td = timedelta(minutes=cfg.MAX_TIME_DIFF_MIN)
#     return [f for f in modis_files
#             if (mdt := _parse_modis_datetime(f.name)) and abs(mdt - agri_dt) <= td]

# def _find_matching_modis(agri_dt: datetime, modis_files: list) -> list:
#     td = timedelta(minutes=cfg.MAX_TIME_DIFF_MIN)
#     candidates = []
#
#     for f in modis_files:
#         mdt = _parse_modis_datetime(f.name)
#         if mdt is None:
#             continue
#
#         dt = abs(mdt - agri_dt)
#         if dt <= td:
#             candidates.append((dt, f))
#
#     candidates.sort(key=lambda x: x[0])
#     return [candidates[0][1]] if candidates else []

def _find_matching_modis(agri_dt: datetime, modis_files: list) -> list:
    td = timedelta(minutes=cfg.MAX_TIME_DIFF_MIN)
    candidates = []

    for f in modis_files:
        mdt = _parse_modis_datetime(f.name)
        if mdt is None:
            continue

        dt = abs(mdt - agri_dt)
        if dt <= td:
            candidates.append((dt, f))

    candidates.sort(key=lambda x: x[0])
    return [f for _, f in candidates]


# ─────────────────────────────────────────────────────────────────────────────
# AGRI reader
# ─────────────────────────────────────────────────────────────────────────────

def _paired_geo_file(fdi_file: Path) -> Path:
    return Path(str(fdi_file).replace("_FDI-_", "_GEO-_"))


def _read_first_dataset(hf: h5py.File, candidates: Sequence[str]) -> np.ndarray:
    for name in candidates:
        if name in hf:
            return hf[name][()]
        try:
            return hf[name][()]
        except Exception:
            pass
    raise KeyError(f"Dataset not found. Tried: {candidates}")


def _read_first_dataset_or_default(
    hf: h5py.File,
    candidates: Sequence[str],
    default: np.ndarray
) -> np.ndarray:
    for name in candidates:
        if name in hf:
            return hf[name][()]
        try:
            return hf[name][()]
        except Exception:
            pass
    return default


def _calibrate_with_lut(raw: np.ndarray, cal: np.ndarray) -> np.ndarray:
    raw_i = raw.astype(np.int64)
    out   = np.full(raw.shape, np.nan, dtype=np.float32)
    valid = (raw_i >= 0) & (raw_i < len(cal))
    out[valid] = cal[raw_i[valid]].astype(np.float32)
    out[raw_i >= 65534] = np.nan
    out[raw_i < 0]      = np.nan
    return out


def _attr_scalar(obj, key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in obj.attrs:
        return default
    v = obj.attrs[key]
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return default
        v = v.reshape(-1)[0]
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8")
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default


def _dataset_scaled(ds: h5py.Dataset) -> np.ndarray:
    arr   = ds[()].astype(np.float64)
    fillv = _attr_scalar(ds, "FillValue", None)
    if fillv is not None:
        arr[arr == fillv] = np.nan
    slope     = _attr_scalar(ds, "Slope", 1.0)
    intercept = _attr_scalar(ds, "Intercept", 0.0)
    return arr * slope + intercept


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)


def _derive_latlon_from_linecol(gf: h5py.File):
    if "LineNumber" not in gf or "ColumnNumber" not in gf:
        raise KeyError("LineNumber / ColumnNumber datasets are required.")

    line = _dataset_scaled(gf["LineNumber"])
    col  = _dataset_scaled(gf["ColumnNumber"])

    lon0_deg = _attr_scalar(gf, "NOMCenterLon", None)
    sat_h    = _attr_scalar(gf, "NOMSatHeight", None)
    ea_attr  = _attr_scalar(gf, "dEA", None)
    flat_inv = _attr_scalar(gf, "dObRecFlat", None)
    samp_ang = _attr_scalar(gf, "dSamplingAngle", None)
    step_ang = _attr_scalar(gf, "dSteppingAngle", None)

    if None in [lon0_deg, sat_h, ea_attr, flat_inv, samp_ang, step_ang]:
        raise KeyError("Missing GEO attrs for line/col → lat/lon derivation.")

    ea_km   = ea_attr / 1000.0 if ea_attr > 1e5 else ea_attr
    sat_h_km = sat_h / 1000.0 if sat_h > 1e5 else sat_h
    H       = sat_h_km + ea_km if sat_h_km < 40000 else sat_h_km
    eb_km   = ea_km * (1.0 - 1.0 / flat_inv)

    line[(~np.isfinite(line)) | (line < 0)] = np.nan
    col[ (~np.isfinite(col))  | (col  < 0)] = np.nan

    coff = 0.5 * (np.nanmin(col) + np.nanmax(col))
    loff = 0.5 * (np.nanmin(line) + np.nanmax(line))

    col_vals  = np.unique(col[0, :][np.isfinite(col[0, :])])
    col_step  = float(np.nanmedian(np.diff(col_vals))) if len(col_vals) >= 2 else 1.0
    line_vals = np.unique(line[:, 0][np.isfinite(line[:, 0])])
    line_step = float(np.nanmedian(np.diff(line_vals))) if len(line_vals) >= 2 else 1.0

    x_per_index = samp_ang * 1.0e-6
    y_per_index = step_ang * 1.0e-6
    if col_step  > 1.5: x_per_index /= col_step
    if line_step > 1.5: y_per_index /= line_step

    x = (col - coff) * x_per_index
    y = (loff - line) * y_per_index

    lon0 = np.deg2rad(lon0_deg)
    ea2, eb2 = ea_km ** 2, eb_km ** 2
    cosx, sinx = np.cos(x), np.sin(x)
    cosy, siny = np.cos(y), np.sin(y)

    a = sinx**2 + cosx**2 * (cosy**2 + (ea2/eb2)*siny**2)
    b = -2.0 * H * cosx * cosy
    c = H**2 - ea2

    disc = b**2 - 4.0 * a * c
    lat  = np.full(line.shape, np.nan, dtype=np.float64)
    lon  = np.full(line.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(line) & np.isfinite(col) & np.isfinite(disc) & (disc >= 0.0)
    if np.any(valid):
        sd  = np.sqrt(disc[valid])
        sn  = (-b[valid] - sd) / (2.0 * a[valid])
        s1  = H - sn * cosx[valid] * cosy[valid]
        s2  = sn * sinx[valid] * cosy[valid]
        s3  = -sn * siny[valid]
        sxy = np.sqrt(s1**2 + s2**2)
        lat[valid] = np.rad2deg(np.arctan((ea2/eb2) * (s3/sxy)))
        lon[valid] = np.rad2deg(np.arctan2(s2, s1) + lon0)

    bad = (lat < -90.0) | (lat > 90.0)
    lat[bad] = np.nan
    lon[bad] = np.nan
    lon = _wrap_lon(lon)
    return lat.astype(np.float32), lon.astype(np.float32)


def _read_agri_latlon_vza_sza_ele(geo_file: Path):
    with h5py.File(geo_file, "r") as gf:
        try:
            lat = _read_first_dataset(
                gf, ["Geolocation/NOMLatitude", "NOMLatitude", "Latitude"]
            ).astype(np.float32)
            lon = _read_first_dataset(
                gf, ["Geolocation/NOMLongitude", "Geolocation/NOMlongitude",
                     "NOMLongitude", "NOMlongitude", "Longitude"]
            ).astype(np.float32)
        except Exception:
            lat, lon = _derive_latlon_from_linecol(gf)

        vza = _read_first_dataset(
            gf, ["Geolocation/NOMSatelliteZenith", "NOMSatelliteZenith", "VZA"]
        ).astype(np.float32)
        sza = _read_first_dataset(
            gf, ["Geolocation/NOMSunZenith", "NOMSunZenith", "SZA"]
        ).astype(np.float32)
        ele = _read_first_dataset_or_default(
            gf, ["Geolocation/DEM", "Geolocation/NOMDEM", "NOMDEM", "DEM", "ELE"],
            default=np.zeros_like(lat, dtype=np.float32)
        ).astype(np.float32)

    # for arr in [lat, lon, vza, sza, ele]:
    for arr in [lat, lon, vza, sza]:
        arr[(arr > 1e4) | (arr < -1e4)] = np.nan

    if np.isfinite(vza).any() and np.nanmax(np.abs(vza)) > 180:
        vza /= 100.0
    if np.isfinite(sza).any() and np.nanmax(np.abs(sza)) > 180:
        sza /= 100.0

    log.info(
        "read geo | %s | lat finite=%.2f%% lon finite=%.2f%% "
        "| vza finite=%.2f%% min=%.2f max=%.2f "
        "| sza finite=%.2f%% min=%.2f max=%.2f",
        geo_file.name,
        100.0 * np.isfinite(lat).mean(),
        100.0 * np.isfinite(lon).mean(),
        100.0 * np.isfinite(vza).mean(),
        np.nanmin(vza) if np.isfinite(vza).any() else np.nan,
        np.nanmax(vza) if np.isfinite(vza).any() else np.nan,
        100.0 * np.isfinite(sza).mean(),
        np.nanmin(sza) if np.isfinite(sza).any() else np.nan,
        np.nanmax(sza) if np.isfinite(sza).any() else np.nan,
    )

    lon = _wrap_lon(lon)
    return lat, lon, vza, sza


def read_agri_scene(agri_file: Path) -> Optional[dict]:
    try:
        if "_FDI-_" not in agri_file.name:
            return None

        geo_file = _paired_geo_file(agri_file)
        if not geo_file.exists():
            log.warning("Paired GEO file not found for %s", agri_file.name)
            return None

        # lat, lon, vza, sza, ele = _read_agri_latlon_vza_sza_ele(geo_file)
        lat, lon, vza, sza = _read_agri_latlon_vza_sza_ele(geo_file)

        bt_list = []
        with h5py.File(agri_file, "r") as ff:
            for idx in cfg.AGRI_BT_CHANNEL_INDICES:
                ch_no = idx + 1
                raw = _read_first_dataset(
                    ff,
                    [f"NOMChannel{ch_no:02d}",
                     f"Data/NOMChannelBT{ch_no:02d}",
                     f"Data/NOMChannel{ch_no:02d}"],
                ).astype(np.float32)

                cal = _read_first_dataset_or_default(
                    ff,
                    [f"CALChannel{ch_no:02d}",
                     f"Calibration/CALChannel{ch_no:02d}"],
                    default=np.array([], dtype=np.float32),
                )

                if cal.size > 0:
                    bt = _calibrate_with_lut(raw, cal)
                else:
                    bt = raw.astype(np.float32)
                    bt[bt > 60000] = np.nan
                    if np.isfinite(bt).any() and np.nanmedian(bt[np.isfinite(bt)]) > 500:
                        bt /= 100.0

                bt_list.append(bt)

        BT = np.stack(bt_list, axis=-1).astype(np.float32)

        if lat.shape != BT.shape[:2]:
            log.warning("Shape mismatch %s: GEO=%s, BT=%s",
                        agri_file.name, lat.shape, BT.shape[:2])
            return None

        if not np.isfinite(lat).any():
            log.warning("No valid lat/lon for %s", agri_file.name)
            return None

        # return dict(lat=lat, lon=lon, VZA=vza, SZA=sza, ELE=ele, BT=BT)
        return dict(lat=lat, lon=lon, VZA=vza, SZA=sza, BT=BT)

    except Exception as exc:
        log.warning("Failed to read AGRI %s: %s", agri_file, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MYD06 reader
# ─────────────────────────────────────────────────────────────────────────────

def read_myd06(modis_file: Path) -> Optional[dict]:
    try:
        sd  = SD(str(modis_file), SDC.READ)
        lat = sd.select("Latitude")[:]
        lon = sd.select("Longitude")[:]

        def _read(name, scale):
            ds    = sd.select(name)
            raw   = ds[:].astype(np.float32)
            attrs = ds.attributes()
            fv    = attrs.get("_FillValue", attrs.get("fill_value", -9999))
            raw[raw == fv] = np.nan
            return raw * scale

        clp_raw = sd.select(cfg.MODIS_VARS["CLP"])[:].astype(np.int32)
        # clp     = np.vectorize(cfg.MODIS_PHASE_MAP.get)(clp_raw, 0).astype(np.int32)
        clp = np.vectorize(lambda x: cfg.MODIS_PHASE_MAP.get(int(x), -1))(clp_raw).astype(np.int16)
        cer     = _read(cfg.MODIS_VARS["CER"], cfg.MODIS_SCALE["CER"])
        cot     = _read(cfg.MODIS_VARS["COT"], cfg.MODIS_SCALE["COT"])
        cth     = _read(cfg.MODIS_VARS["CTH"], cfg.MODIS_SCALE["CTH"])
        sd.end()

        return dict(
            lat=lat.ravel(), lon=lon.ravel(),
            CLP=clp.ravel(), CER=cer.ravel(),
            COT=cot.ravel(), CTH=cth.ravel(),
        )
    except Exception as exc:
        log.warning("Failed to read MYD06 %s: %s", modis_file, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Spatial matching
# ─────────────────────────────────────────────────────────────────────────────

def _latlon_to_xyz(lat, lon):
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


# def match_modis_to_agri(agri: dict, modis_list: list) -> Optional[dict]:
#     if not modis_list:
#         return None
#
#     m_lat = np.concatenate([m["lat"] for m in modis_list])
#     m_lon = np.concatenate([m["lon"] for m in modis_list])
#     m_clp = np.concatenate([m["CLP"] for m in modis_list])
#     m_cer = np.concatenate([m["CER"] for m in modis_list])
#     m_cot = np.concatenate([m["COT"] for m in modis_list])
#     m_cth = np.concatenate([m["CTH"] for m in modis_list])
#
#     valid = ~(np.isnan(m_lat) | np.isnan(m_lon))
#     if valid.sum() == 0:
#         return None
#
#     m_xyz = _latlon_to_xyz(m_lat[valid], m_lon[valid])
#     tree  = cKDTree(m_xyz)
#
#     H, W   = agri["lat"].shape
#     a_flat = agri["lat"].ravel()
#     b_flat = agri["lon"].ravel()
#     nan_px = np.isnan(a_flat) | np.isnan(b_flat)
#
#     a_xyz = _latlon_to_xyz(
#         np.where(nan_px, 0, a_flat),
#         np.where(nan_px, 0, b_flat),
#     )
#
#     # Chord distance on a unit sphere.
#     # arc_rad = dist_km / R_km  (already in radians — no deg2rad needed)
#     # chord   = 2 * sin(arc_rad / 2)
#     arc_rad   = cfg.MAX_MATCH_DIST_KM / 6371.0          # radians
#     max_chord = 2.0 * np.sin(arc_rad / 2.0)             # unit-sphere chord
#     dists, idx = tree.query(a_xyz, k=1, distance_upper_bound=max_chord + 1e-6)
#
#     valid_match = (dists <= max_chord) & (~nan_px)
#     full_idx    = np.where(valid)[0]   # maps compressed → full MODIS index
#
#     out_clp = np.full(H * W, np.nan)
#     out_cer = np.full(H * W, np.nan)
#     out_cot = np.full(H * W, np.nan)
#     out_cth = np.full(H * W, np.nan)
#
#     hits   = np.where(valid_match)[0]
#     mapped = full_idx[idx[hits]]
#
#     out_clp[hits] = m_clp[mapped]
#     out_cer[hits] = m_cer[mapped]
#     out_cot[hits] = m_cot[mapped]
#     out_cth[hits] = m_cth[mapped]
#
#     return dict(
#         CLP=out_clp.reshape(H, W).astype(np.float32),
#         CER=out_cer.reshape(H, W).astype(np.float32),
#         COT=out_cot.reshape(H, W).astype(np.float32),
#         CTH=out_cth.reshape(H, W).astype(np.float32),
#     )

def match_modis_to_agri(agri: dict, modis_list: list) -> Optional[dict]:
    if not modis_list:
        return None

    H, W = agri["lat"].shape
    a_lat = agri["lat"].ravel()
    a_lon = agri["lon"].ravel()
    nan_px = np.isnan(a_lat) | np.isnan(a_lon)

    a_xyz = _latlon_to_xyz(
        np.where(nan_px, 0, a_lat),
        np.where(nan_px, 0, a_lon),
    )

    arc_rad   = cfg.MAX_MATCH_DIST_KM / 6371.0
    max_chord = 2.0 * np.sin(arc_rad / 2.0)

    out = {
        "CLP": np.full(H * W, np.nan, dtype=np.float32),
        "CER": np.full(H * W, np.nan, dtype=np.float32),
        "COT": np.full(H * W, np.nan, dtype=np.float32),
        "CTH": np.full(H * W, np.nan, dtype=np.float32),
    }

    best_dt   = np.full(H * W, np.inf, dtype=np.float32)
    best_dist = np.full(H * W, np.inf, dtype=np.float32)
    best_ok   = np.zeros(H * W, dtype=bool)

    modis_list = sorted(modis_list, key=lambda m: m["_dt_min"])

    for m in modis_list:
        valid_geo = ~(np.isnan(m["lat"]) | np.isnan(m["lon"]))
        if valid_geo.sum() == 0:
            continue

        m_xyz = _latlon_to_xyz(m["lat"][valid_geo], m["lon"][valid_geo])
        tree  = cKDTree(m_xyz)

        dists, idx = tree.query(
            a_xyz, k=1, distance_upper_bound=max_chord + 1e-6
        )
        valid_match = (dists <= max_chord) & (~nan_px)
        if not np.any(valid_match):
            continue

        full_idx = np.where(valid_geo)[0]
        hits     = np.where(valid_match)[0]
        mapped   = full_idx[idx[hits]]

        cand_clp = m["CLP"][mapped]
        cand_cer = m["CER"][mapped]
        cand_cot = m["COT"][mapped]
        cand_cth = m["CTH"][mapped]

        # 这里定义“候选标签有效”
        cand_ok = np.isfinite(cand_clp) & (cand_clp >= 0)

        cur_ok = best_ok[hits]
        cur_dt = best_dt[hits]
        cur_ds = best_dist[hits]

        better = (
            (~cur_ok & cand_ok) |
            (cur_ok == cand_ok) & (
                (m["_dt_min"] < cur_dt) |
                ((m["_dt_min"] == cur_dt) & (dists[hits] < cur_ds))
            )
        )

        if not np.any(better):
            continue

        keep   = hits[better]
        kmapped = mapped[better]

        out["CLP"][keep] = m["CLP"][kmapped]
        out["CER"][keep] = m["CER"][kmapped]
        out["COT"][keep] = m["COT"][kmapped]
        out["CTH"][keep] = m["CTH"][kmapped]

        best_ok[keep]   = cand_ok[better]
        best_dt[keep]   = m["_dt_min"]
        best_dist[keep] = dists[keep]

    return {
        "CLP": out["CLP"].reshape(H, W),
        "CER": out["CER"].reshape(H, W),
        "COT": out["COT"].reshape(H, W),
        "CTH": out["CTH"].reshape(H, W),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quality filter
# ─────────────────────────────────────────────────────────────────────────────

# def apply_quality_filter(agri: dict, labels: dict) -> dict:
#     vza = agri["VZA"]
#     sza = agri["SZA"]
#
#     # Clear pixels have no cloud property — set regression targets to NaN so
#     # that train.py's masked loss ignores them entirely.  Setting to 0.0 forces
#     # the regression head toward CER=0 µm / CTH=0 m for the majority of
#     # clear-sky pixels (often 50-80% of a scene), biasing the model low on all
#     # cloud property retrievals.
#     clear_mask = labels["CLP"] == 0
#     for k in ["CER", "COT", "CTH"]:
#         labels[k][clear_mask] = np.nan
#
#     bad_clp = (labels["CLP"] < 0) | (labels["CLP"] >= cfg.CLP_CLASSES)
#     for k in labels:
#         labels[k][bad_clp] = np.nan
#
#     for k in list(labels.keys()):
#         labels[k][vza > cfg.MAX_VZA_DEG] = np.nan
#         labels[k][sza > cfg.MAX_SZA_DEG] = np.nan
#
#     bad = (labels["CER"] < 0) | (labels["CER"] > 100)
#     for k in list(labels.keys()): labels[k][bad] = np.nan
#     bad = (labels["COT"] < 0) | (labels["COT"] > 200)
#     for k in list(labels.keys()): labels[k][bad] = np.nan
#     bad = (labels["CTH"] < 0) | (labels["CTH"] > 25000)
#     for k in list(labels.keys()): labels[k][bad] = np.nan
#
#     return labels

def apply_quality_filter(agri: dict, labels: dict) -> dict:
    vza = agri["VZA"]
    sza = agri["SZA"]

    # 先看几何量本身是不是有问题
    log.info(
        "geo stats | "
        "VZA finite=%.2f%% min=%.2f max=%.2f | "
        "SZA finite=%.2f%% min=%.2f max=%.2f",
        100.0 * np.isfinite(vza).mean(),
        np.nanmin(vza) if np.isfinite(vza).any() else np.nan,
        np.nanmax(vza) if np.isfinite(vza).any() else np.nan,
        100.0 * np.isfinite(sza).mean(),
        np.nanmin(sza) if np.isfinite(sza).any() else np.nan,
        np.nanmax(sza) if np.isfinite(sza).any() else np.nan,
    )

    geo_ok = (
        np.isfinite(vza) & np.isfinite(sza) &
        (vza <= cfg.MAX_VZA_DEG) &
        (sza <= cfg.MAX_SZA_DEG)
    )

    log.info(
        "filter ratio | geo_ok=%d (%.2f%%) | vza_thr=%.2f | sza_thr=%.2f",
        int(geo_ok.sum()),
        100.0 * geo_ok.mean(),
        cfg.MAX_VZA_DEG,
        cfg.MAX_SZA_DEG,
    )

    # 1) 分类单独过滤
    clp_raw = labels["CLP"].copy()

    clp_ok = (
        np.isfinite(clp_raw) &
        (clp_raw >= 0) &
        (clp_raw < cfg.CLP_CLASSES)
    )

    overlap_ok = geo_ok & clp_ok

    # 看 clp 有效区和 geo 有效区分别落在图上的哪里
    hits_clp = np.where(clp_ok)
    hits_geo = np.where(geo_ok)
    hits_ovl = np.where(overlap_ok)

    if hits_clp[0].size > 0:
        log.info(
            "clp bbox | row=[%d,%d] col=[%d,%d]",
            hits_clp[0].min(), hits_clp[0].max(),
            hits_clp[1].min(), hits_clp[1].max(),
        )

    if hits_geo[0].size > 0:
        log.info(
            "geo bbox | row=[%d,%d] col=[%d,%d]",
            hits_geo[0].min(), hits_geo[0].max(),
            hits_geo[1].min(), hits_geo[1].max(),
        )

    if hits_ovl[0].size > 0:
        log.info(
            "overlap bbox | row=[%d,%d] col=[%d,%d]",
            hits_ovl[0].min(), hits_ovl[0].max(),
            hits_ovl[1].min(), hits_ovl[1].max(),
        )
    else:
        log.info("overlap bbox | empty")

    clear_raw = np.isfinite(clp_raw) & (clp_raw == 0)
    cloudy_raw = np.isfinite(clp_raw) & (clp_raw > 0)

    log.info(
        "clp before qc | finite=%d | valid_range=%d | clear=%d | cloudy=%d | overlap_geo=%d",
        int(np.isfinite(clp_raw).sum()),
        int(clp_ok.sum()),
        int(clear_raw.sum()),
        int(cloudy_raw.sum()),
        int(overlap_ok.sum()),
    )

    # 如果 overlap 很低，基本说明“匹配到了，但几乎都落在 geo_ok 外”
    if clp_ok.sum() > 0:
        log.info(
            "clp overlap ratio | overlap/valid_range=%.2f%% | overlap/all_pixels=%.2f%%",
            100.0 * overlap_ok.sum() / clp_ok.sum(),
            100.0 * overlap_ok.mean(),
        )

    labels["CLP"] = np.where(overlap_ok, clp_raw, np.nan)

    log.info(
        "clp after qc | finite=%d | clear=%d | cloudy=%d",
        int(np.isfinite(labels["CLP"]).sum()),
        int((np.isfinite(labels["CLP"]) & (labels["CLP"] == 0)).sum()),
        int((np.isfinite(labels["CLP"]) & (labels["CLP"] > 0)).sum()),
    )

    # 2) 回归各自单独过滤
    cloudy = np.isfinite(labels["CLP"]) & (labels["CLP"] > 0)

    cer_raw = labels["CER"].copy()
    cot_raw = labels["COT"].copy()
    cth_raw = labels["CTH"].copy()

    cer_ok = cloudy & np.isfinite(cer_raw) & (cer_raw >= 0) & (cer_raw <= 100)
    cot_ok = cloudy & np.isfinite(cot_raw) & (cot_raw >= 0) & (cot_raw <= 200)
    cth_ok = cloudy & np.isfinite(cth_raw) & (cth_raw >= 0) & (cth_raw <= 25000)

    log.info(
        "reg before qc | CER=%d | COT=%d | CTH=%d | cloudy_mask=%d",
        int(np.isfinite(cer_raw).sum()),
        int(np.isfinite(cot_raw).sum()),
        int(np.isfinite(cth_raw).sum()),
        int(cloudy.sum()),
    )

    labels["CER"] = np.where(cer_ok, cer_raw, np.nan)
    labels["COT"] = np.where(cot_ok, cot_raw, np.nan)
    labels["CTH"] = np.where(cth_ok, cth_raw, np.nan)

    log.info(
        "reg after qc | CER=%d | COT=%d | CTH=%d",
        int(np.isfinite(labels["CER"]).sum()),
        int(np.isfinite(labels["COT"]).sum()),
        int(np.isfinite(labels["CTH"]).sum()),
    )

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

def write_paired_hdf5(out_path: Path, agri: dict, labels: dict, agri_dt: datetime):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["agri_datetime"] = agri_dt.strftime("%Y%m%d%H%M%S")
        f.attrs["agri_channels"] = cfg.AGRI_BT_CHANNEL_INDICES
        f.attrs["patch_size"]    = list(cfg.PATCH_SIZE)

        geo = f.create_group("AGRI/Geolocation")
        geo.create_dataset("lat", data=agri["lat"], compression="gzip", compression_opts=4)
        geo.create_dataset("lon", data=agri["lon"], compression="gzip", compression_opts=4)
        geo.create_dataset("VZA", data=agri["VZA"], compression="gzip", compression_opts=4)
        geo.create_dataset("SZA", data=agri["SZA"], compression="gzip", compression_opts=4)

        # aux = f.create_group("AGRI/Aux")
        # aux.create_dataset("ELE", data=agri["ELE"], compression="gzip", compression_opts=4)

        bt_grp = f.create_group("AGRI/BT")
        for ci, ch_idx in enumerate(cfg.AGRI_BT_CHANNEL_INDICES):
            bt_grp.create_dataset(
                f"ch{ch_idx+1:02d}", data=agri["BT"][..., ci],
                compression="gzip", compression_opts=4
            )

        lbl = f.create_group("Labels")
        for k, v in labels.items():
            lbl.create_dataset(k, data=v, compression="gzip", compression_opts=4)

    log.debug("Wrote %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# QC diagnostic figure  [NEW]
# ─────────────────────────────────────────────────────────────────────────────

def _make_qc_figure(
    agri: dict,
    labels: dict,
    agri_dt: datetime,
    out_path: Path,
) -> None:
    """
    Generate a 4-panel diagnostic PNG to verify spatial co-location between
    the AGRI scene and the matched MYD06 labels.

    Panels
    ------
    Top-left  : AGRI BT for the first thermal channel (K), full scene.
                If the geolocation derived lat/lon are correct, the Earth disk
                edge should be clearly visible.

    Top-right : MYD06 CTH (m) re-gridded onto the AGRI grid.
                Should spatially overlap with cloud structures visible in BT.
                NaN pixels (no MODIS match, or clear sky) are shown as grey.

    Bottom-left  : MYD06 CLP class on the AGRI grid.
                   Color coding: 0=clear (white), 1=water (blue),
                   2=supercool (cyan), 3=mixed (orange), 4=ice (red).
                   NaN pixels shown as grey.

    Bottom-right : Scatter plot of AGRI BT (ch1) vs MODIS CTH for all matched
                   cloudy pixels.  If matching is correct, cold BT → high CTH
                   (negative correlation expected).  A random subsample of up
                   to 5 000 pixels is used to keep the plot readable.

    Interpretation guide (printed in figure title)
    -----------------------------------------------
    ✓  BT cold patches (dark) align with high CTH and cloudy CLP regions
       → spatial match is good.
    ✗  CTH / CLP "blobs" appear shifted or scattered relative to BT structure
       → check time offset (MAX_TIME_DIFF_MIN) or spatial distance threshold.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend – safe in scripts
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError:
        log.warning("matplotlib not available – QC figure skipped")
        return

    BT  = agri["BT"]            # (H, W, C)
    lat = agri["lat"]
    lon = agri["lon"]

    CLP = labels["CLP"]         # (H, W)
    CTH = labels["CTH"]         # (H, W)

    bt_ch0 = BT[..., 0]        # first thermal channel

    # ---------- subsample for scatter ----------
    clp_flat = CLP.ravel()
    cth_flat = CTH.ravel()
    bt_flat  = bt_ch0.ravel()

    cloudy_mask = (
        np.isfinite(clp_flat) & (clp_flat > 0) &
        np.isfinite(cth_flat) & np.isfinite(bt_flat)
    )
    n_cloudy = cloudy_mask.sum()
    n_sample = min(5000, n_cloudy)

    if n_sample > 0:
        sample_idx = np.random.choice(np.where(cloudy_mask)[0], n_sample, replace=False)
        sc_bt  = bt_flat[sample_idx]
        sc_cth = cth_flat[sample_idx]
    else:
        sc_bt = sc_cth = np.array([])

    # ---------- CLP colormap ----------
    clp_colors = ["white", "deepskyblue", "cyan", "orange", "red"]
    clp_cmap   = ListedColormap(clp_colors)
    clp_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    clp_norm   = BoundaryNorm(clp_bounds, clp_cmap.N)

    # ---------- figure layout ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Fusion QC  –  AGRI {agri_dt:%Y-%m-%d %H:%M UTC}\n"
        f"Matched pixels: {int(np.isfinite(CLP).sum()):,}  |  "
        f"Cloudy: {int(n_cloudy):,}  |  "
        f"Scene size: {BT.shape[0]}×{BT.shape[1]}",
        fontsize=12, fontweight="bold", y=1.01
    )

    # ── Panel 1: AGRI BT ─────────────────────────────────────────────────
    ax = axes[0, 0]
    finite_bt = bt_ch0[np.isfinite(bt_ch0)]
    vmin_bt = float(np.percentile(finite_bt, 2))  if finite_bt.size else 200
    vmax_bt = float(np.percentile(finite_bt, 98)) if finite_bt.size else 310
    im1 = ax.imshow(bt_ch0, cmap="RdYlBu_r", vmin=vmin_bt, vmax=vmax_bt,
                    aspect="auto", interpolation="none")
    plt.colorbar(im1, ax=ax, label="BT (K)", fraction=0.046, pad=0.04)
    ax.set_title(f"AGRI BT ch{cfg.AGRI_BT_CHANNEL_INDICES[0]+1} (K)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # ── Panel 2: MODIS CTH ────────────────────────────────────────────────
    ax = axes[0, 1]
    cth_disp = np.where(np.isfinite(CTH), CTH, np.nan)
    finite_cth = cth_disp[np.isfinite(cth_disp)]
    vmax_cth = float(np.percentile(finite_cth, 98)) if finite_cth.size else 15000

    # grey background for NaN (no match / clear)
    cmap_cth = plt.cm.viridis_r.copy()
    cmap_cth.set_bad(color="lightgrey")
    im2 = ax.imshow(cth_disp, cmap=cmap_cth, vmin=0, vmax=vmax_cth,
                    aspect="auto", interpolation="none")
    plt.colorbar(im2, ax=ax, label="CTH (m)", fraction=0.046, pad=0.04)
    match_pct = 100.0 * np.isfinite(CTH).sum() / CTH.size
    ax.set_title(f"MODIS CTH (m)  |  match coverage: {match_pct:.1f}%")
    ax.set_xlabel("Column")

    # ── Panel 3: MODIS CLP ────────────────────────────────────────────────
    ax = axes[1, 0]
    clp_disp = np.where(np.isfinite(CLP), CLP, np.nan)
    cmap_clp = clp_cmap.copy()
    cmap_clp.set_bad(color="lightgrey")
    im3 = ax.imshow(clp_disp, cmap=cmap_clp, norm=clp_norm,
                    aspect="auto", interpolation="none")
    cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04,
                          ticks=[0, 1, 2, 3, 4])
    cbar3.ax.set_yticklabels(["Clear", "Water", "Supercool", "Mixed", "Ice"],
                              fontsize=8)
    ax.set_title("MODIS CLP class")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # ── Panel 4: BT vs CTH scatter ────────────────────────────────────────
    ax = axes[1, 1]
    if sc_bt.size > 0:
        ax.scatter(sc_bt, sc_cth / 1000.0, s=3, alpha=0.3,
                   c=sc_cth, cmap="viridis_r", rasterized=True)

        # Pearson r
        if sc_bt.std() > 0 and sc_cth.std() > 0:
            r = float(np.corrcoef(sc_bt, sc_cth)[0, 1])
        else:
            r = 0.0

        ax.set_xlabel(f"AGRI BT ch{cfg.AGRI_BT_CHANNEL_INDICES[0]+1} (K)")
        ax.set_ylabel("MODIS CTH (km)")
        ax.set_title(
            f"BT vs CTH scatter  (n={n_sample:,}, r={r:.3f})\n"
            f"[Expect cold BT ↔ high CTH, r < 0]",
            fontsize=9
        )

        # Annotation: good match hint
        hint_color = "green" if r < -0.2 else ("orange" if r < 0 else "red")
        quality = "Good" if r < -0.2 else ("Weak" if r < 0 else "Check!")
        ax.text(0.05, 0.93, f"Match quality: {quality}",
                transform=ax.transAxes, color=hint_color,
                fontsize=10, fontweight="bold", va="top")
    else:
        ax.text(0.5, 0.5, "No cloudy matched pixels",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="grey")
        ax.set_title("BT vs CTH scatter (no data)")

    # ── Additional info: lat/lon coverage ────────────────────────────────
    fin_lat = lat[np.isfinite(lat)]
    fin_lon = lon[np.isfinite(lon)]
    if fin_lat.size > 0:
        info = (f"Lat: [{fin_lat.min():.1f}, {fin_lat.max():.1f}]°  "
                f"Lon: [{fin_lon.min():.1f}, {fin_lon.max():.1f}]°")
        fig.text(0.5, -0.01, info, ha="center", fontsize=9, color="grey")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("QC figure saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-day fusion driver
# ─────────────────────────────────────────────────────────────────────────────

def fuse_day(
    agri_day_dir: Path,
    modis_day_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    max_qc: int = 3,
) -> int:
    """
    Process all AGRI scenes in agri_day_dir, match to MYD06 in modis_day_dir,
    write paired HDF5 files + QC PNG figures to out_dir.

    Parameters
    ----------
    max_qc : int
        Maximum number of QC figures to generate per day.
        Set to 0 to disable, or a large number to generate for all scenes.
        Default is 3 (enough to spot problems without flooding disk).
    """
    agri_files = sorted([
        p for p in list(agri_day_dir.glob("*.HDF")) + list(agri_day_dir.glob("*.hdf"))
        if "_FDI-_" in p.name
    ])
    modis_files = (sorted(modis_day_dir.glob("MYD06*.hdf")) +
                   sorted(modis_day_dir.glob("MYD06*.HDF")))

    log.info("Day %s – %d AGRI files, %d MYD06 files",
             agri_day_dir.name, len(agri_files), len(modis_files))

    paired_count = 0
    qc_count     = 0

    for agri_file in agri_files:
        agri_dt = _parse_agri_datetime(agri_file.name)
        if agri_dt is None:
            log.warning("Cannot parse datetime from %s – skipping", agri_file.name)
            continue

        out_name = f"AGRI_MYD06_pair_{agri_dt.strftime('%Y%m%d_%H%M%S')}.h5"
        out_path = out_dir / out_name
        qc_path  = out_dir / (out_path.stem + "_qc.png")

        if out_path.exists() and not overwrite:
            log.debug("Skip existing %s", out_path.name)
            paired_count += 1
            # Still generate QC figure if missing and budget remains
            if qc_count < max_qc and not qc_path.exists():
                _generate_qc_from_h5(out_path, qc_path, agri_dt)
                qc_count += 1
            continue

        # Find matching MODIS granules
        matched_modis_files = _find_matching_modis(agri_dt, modis_files)
        if not matched_modis_files:
            log.debug("No MODIS match for %s", agri_file.name)
            continue

        # Read AGRI
        agri = read_agri_scene(agri_file)
        if agri is None:
            continue

        # Read & merge MYD06 granules
        # modis_data_list = [m for f in matched_modis_files if (m := read_myd06(f)) is not None]
        # if not modis_data_list:
        #     continue
        #
        # labels = match_modis_to_agri(agri, modis_data_list)

        # Read & merge MYD06 granules
        modis_data_list = []
        for mf in matched_modis_files:
            m = read_myd06(mf)
            if m is None:
                continue

            mdt = _parse_modis_datetime(mf.name)
            if mdt is None:
                continue

            m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
            m["_file"] = mf.name
            modis_data_list.append(m)

        if not modis_data_list:
            continue

        labels = match_modis_to_agri(agri, modis_data_list)
        log.info(
            "before qc | raw_clp=%d raw_cer=%d raw_cot=%d raw_cth=%d | file=%s",
            int(np.isfinite(labels["CLP"]).sum()),
            int(np.isfinite(labels["CER"]).sum()),
            int(np.isfinite(labels["COT"]).sum()),
            int(np.isfinite(labels["CTH"]).sum()),
            agri_file.name,
        )

        if labels is None:
            log.debug("No valid MODIS-AGRI matches for %s", agri_file.name)
            continue

        labels_before_qc = {k: v.copy() for k, v in labels.items()}
        raw_match = np.isfinite(labels_before_qc["CLP"]).sum()
        labels = apply_quality_filter(agri, labels)
        log.info(
            "after qc | clp=%d cer=%d cot=%d cth=%d | file=%s",
            int(np.isfinite(labels["CLP"]).sum()),
            int(np.isfinite(labels["CER"]).sum()),
            int(np.isfinite(labels["COT"]).sum()),
            int(np.isfinite(labels["CTH"]).sum()),
            agri_file.name,
        )

        final_clp = np.isfinite(labels["CLP"]).sum()
        final_cer = np.isfinite(labels["CER"]).sum()
        final_cot = np.isfinite(labels["COT"]).sum()
        final_cth = np.isfinite(labels["CTH"]).sum()

        log.info(
            "match stats | raw=%d | clp=%d | cer=%d | cot=%d | cth=%d | file=%s",
            raw_match, final_clp, final_cer, final_cot, final_cth, agri_file.name
        )

        total_px = labels["CLP"].size
        log.info(
            "match ratio | raw=%.2f%% | clp=%.2f%% | cer=%.2f%% | cot=%.2f%% | cth=%.2f%% | file=%s",
            100.0 * raw_match / total_px,
            100.0 * final_clp / total_px,
            100.0 * final_cer / total_px,
            100.0 * final_cot / total_px,
            100.0 * final_cth / total_px,
            agri_file.name
        )

        # Check there are enough valid pixels
        valid_px = (~np.isnan(labels["CLP"])).sum()
        if valid_px < cfg.PATCH_SIZE[0] * cfg.PATCH_SIZE[1]:
            log.debug("Too few valid pixels (%d) for %s – skipping", valid_px, agri_file.name)
            continue

        write_paired_hdf5(out_path, agri, labels, agri_dt)
        paired_count += 1

        # ── QC figure ─────────────────────────────────────────────────────
        if qc_count < max_qc:
            try:
                _make_qc_figure(agri, labels, agri_dt, qc_path)
            except Exception as exc:
                log.warning("QC figure failed for %s: %s", agri_file.name, exc)
            qc_count += 1

    log.info("Day %s – produced %d paired files, %d QC figures in %s",
             agri_day_dir.name, paired_count, qc_count, out_dir)
    return paired_count


def _generate_qc_from_h5(h5_path: Path, qc_path: Path, agri_dt: datetime) -> None:
    """
    Re-read an existing paired HDF5 and regenerate the QC figure.
    Called when the .h5 already exists but the _qc.png is missing.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            bt_keys = sorted(f["AGRI/BT"].keys())
            BT  = np.stack([f[f"AGRI/BT/{k}"][()] for k in bt_keys], axis=-1)
            lat = f["AGRI/Geolocation/lat"][()]
            lon = f["AGRI/Geolocation/lon"][()]
            vza = f["AGRI/Geolocation/VZA"][()]
            sza = f["AGRI/Geolocation/SZA"][()]
            # ele = f["AGRI/Aux/ELE"][()]
            CLP = f["Labels/CLP"][()]
            CER = f["Labels/CER"][()]
            COT = f["Labels/COT"][()]
            CTH = f["Labels/CTH"][()]

        # agri   = dict(lat=lat, lon=lon, VZA=vza, SZA=sza, ELE=ele, BT=BT)
        agri   = dict(lat=lat, lon=lon, VZA=vza, SZA=sza, BT=BT)
        labels = dict(CLP=CLP, CER=CER, COT=COT, CTH=CTH)
        _make_qc_figure(agri, labels, agri_dt, qc_path)

    except Exception as exc:
        log.warning("Could not regenerate QC from %s: %s", h5_path.name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _setup_logging()
    parser = argparse.ArgumentParser(description="AGRI + MYD06 data fusion")
    parser.add_argument("--split",   choices=["train", "val", "test"], default="train")
    parser.add_argument("--day",     default=None,
                        help="Single day YYYYMMDD to process (default: all days)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_qc",  type=int, default=3,
                        help="Max QC figures to generate per day (0 to disable)")
    args = parser.parse_args()

    split_out = {
        "train": cfg.PAIRED_TRAIN_DIR,
        "val":   cfg.PAIRED_VAL_DIR,
        "test":  cfg.PAIRED_TEST_DIR,
    }[args.split]

    dates = {
        "train": cfg.TRAIN_DATES,
        "val":   cfg.VAL_DATES,
        "test":  cfg.TEST_DATES,
    }[args.split]
    if args.day:
        dates = [args.day]

    agri_days  = _find_day_folders(cfg.AGRI_ROOT, dates)
    modis_days = {d.name: d for d in _find_day_folders(cfg.MODIS_ROOT, dates)}

    total = 0
    for agri_day in agri_days:
        modis_day = modis_days.get(agri_day.name)
        if modis_day is None:
            log.warning("No MODIS folder for day %s – skipping", agri_day.name)
            continue
        total += fuse_day(
            agri_day, modis_day,
            split_out / agri_day.name,
            overwrite=args.overwrite,
            max_qc=args.max_qc,
        )

    log.info("Fusion complete – %d paired files total", total)


if __name__ == "__main__":
    main()
