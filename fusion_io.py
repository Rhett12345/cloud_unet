"""
fusion_io.py
============
AGRI 和 GPM 文件读取 + HDF5 写出，与聚合逻辑完全解耦。

主要函数
--------
  read_agri_scene(path)   → dict(lat, lon, VZA, SZA, BT)
  read_gpm_file(path)     → dict(precip, lat, lon, time, quality)
  write_gpm_fused_samples(...) → 写出 samples_v3 格式 HDF5
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

import config as cfg
import fusion_config as fc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文件名时间解析
# ---------------------------------------------------------------------------

def parse_agri_datetime(filename: str) -> Optional[datetime]:
    m = re.search(r"(\d{8})(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def parse_gpm_datetime(filename: str) -> Optional[datetime]:
    """GPM IMERG filename → centre time as datetime.
    Format: 3B-HHR.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.MMMM.V07B.HDF5
    """
    m = re.search(r"(\d{8})-S(\d{6})-E(\d{6})", filename)
    if m:
        try:
            date_str = m.group(1)
            start_time = m.group(2)
            dt_start = datetime.strptime(date_str + start_time, "%Y%m%d%H%M%S")
            end_time = m.group(3)
            dt_end = datetime.strptime(date_str + end_time, "%Y%m%d%H%M%S")
            centre = dt_start + (dt_end - dt_start) / 2
            return centre
        except ValueError:
            pass
    return None


def find_day_folders(root: Path, dates: list) -> list:
    if dates:
        return [root / d for d in dates if (root / d).is_dir()]
    return sorted(p for p in root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# AGRI 读取（保留自原 fusion_io.py）
# ---------------------------------------------------------------------------

def _paired_geo_file(fdi: Path) -> Path:
    """FDI 路径 → 对应 GEO 路径。兼容 FY-4A (同目录) 与 FY-4B (FDI/GEO 分目录)。"""
    geo_name = fdi.name.replace("_FDI-_", "_GEO-_")
    day_dir = fdi.parent
    candidates = [
        day_dir / geo_name,
        day_dir.parent.parent / "GEO" / day_dir.name / geo_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _h5_read_first(hf: h5py.File, candidates: Sequence[str]) -> np.ndarray:
    for name in candidates:
        if name in hf:
            return hf[name][()]
    raise KeyError(f"None of {candidates} found in HDF5")


def _h5_read_first_or(hf: h5py.File, candidates: Sequence[str],
                       default: np.ndarray) -> np.ndarray:
    try:
        return _h5_read_first(hf, candidates)
    except KeyError:
        return default


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


def _lut_calibrate(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    raw_i = raw.astype(np.int64)
    out = np.full(raw.shape, np.nan, np.float32)
    ok = (raw_i >= 0) & (raw_i < len(lut))
    out[ok] = lut[raw_i[ok]].astype(np.float32)
    out[(raw_i >= 65534) | (raw_i < 0)] = np.nan
    return out


def _dataset_scaled(ds: h5py.Dataset) -> np.ndarray:
    arr = ds[()].astype(np.float64)
    fv = _attr_scalar(ds, "FillValue")
    if fv is not None:
        arr[arr == fv] = np.nan
    return arr * _attr_scalar(ds, "Slope", 1.0) + _attr_scalar(ds, "Intercept", 0.0)


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)


def _derive_latlon(gf: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    """从 LineNumber / ColumnNumber + 卫星轨道参数反算经纬度（AGRI GEO 文件）。"""
    line_ds = gf.get("Navigation/LineNumber") or gf.get("LineNumber")
    col_ds  = gf.get("Navigation/ColumnNumber") or gf.get("ColumnNumber")
    if line_ds is None or col_ds is None:
        raise KeyError("Missing LineNumber/ColumnNumber in GEO file")
    line = _dataset_scaled(line_ds)
    col  = _dataset_scaled(col_ds)
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


def _read_geo(geo_file: Path):
    with h5py.File(geo_file, "r") as gf:
        try:
            lat = _h5_read_first(gf, ["Geolocation/NOMLatitude", "NOMLatitude",
                                       "Navigation/NOMLatitude", "Latitude"]).astype(np.float32)
            lon = _h5_read_first(gf, ["Geolocation/NOMLongitude", "Geolocation/NOMlongitude",
                                       "Navigation/NOMLongitude", "NOMLongitude", "Longitude"]).astype(np.float32)
        except KeyError:
            lat, lon = _derive_latlon(gf)

        vza = _h5_read_first(gf, ["Geolocation/NOMSatelliteZenith", "NOMSatelliteZenith",
                                   "Navigation/NOMSatelliteZenith", "VZA"]).astype(np.float32)
        sza = _h5_read_first(gf, ["Geolocation/NOMSunZenith", "NOMSunZenith",
                                   "Navigation/NOMSunZenith", "SZA"]).astype(np.float32)

    for arr in [lat, lon, vza, sza]:
        arr[(arr > 1e4) | (arr < -1e4)] = np.nan
    if np.isfinite(vza).any() and np.nanmax(np.abs(vza)) > 180:
        vza /= 100.0
    if np.isfinite(sza).any() and np.nanmax(np.abs(sza)) > 180:
        sza /= 100.0

    lon = _wrap_lon(lon)
    return lat, lon, vza, sza


# ═══════════════════════════════════════════════════════════════════
# FY-4B → FY-4A 通道转换（基于交叉定标系数）
# ═══════════════════════════════════════════════════════════════════

def _load_b2a_coeffs() -> dict:
    import csv
    csv_path = Path(__file__).resolve().parent / "transfer_coeff_fy4a_fy4b_v1.csv"
    coeffs = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["Direction"] != "B2A":
                continue
            src = row["Source_Ch"]
            src_ch = int(src.split("ch")[1])
            model_type = row.get("Model", "linear").strip()
            c1 = float(row["Coeff_1"]) if row.get("Coeff_1", "").strip() else 0.0
            c2 = float(row["Coeff_2"]) if row.get("Coeff_2", "").strip() else 0.0
            intercept = float(row["Intercept"]) if row.get("Intercept", "").strip() else 0.0
            coeffs[src_ch] = (model_type, c1, c2, intercept)
    return coeffs


_B2A_COEFFS = None


def convert_bt_fy4b_to_fy4a(bt: np.ndarray, channel_indices_b: list) -> np.ndarray:
    """将 FY-4B BT (H, W, C) 转换为 FY-4A 等效 BT。"""
    global _B2A_COEFFS
    if _B2A_COEFFS is None:
        _B2A_COEFFS = _load_b2a_coeffs()

    H, W, C = bt.shape
    bt_a = np.full_like(bt, np.nan)
    for ci, idx_b in enumerate(channel_indices_b):
        ch_b = idx_b + 1
        if ch_b not in _B2A_COEFFS:
            log.warning("No B2A coeff for B%02d, pass-through", ch_b)
            bt_a[:, :, ci] = bt[:, :, ci]
            continue
        model_type, c1, c2, intercept = _B2A_COEFFS[ch_b]
        val = bt[:, :, ci]
        if model_type == "linear":
            bt_a[:, :, ci] = c1 * val + intercept
        else:
            bt_a[:, :, ci] = c2 * val**2 + c1 * val + intercept
    return bt_a


def read_agri_scene(agri_file: Path) -> Optional[dict]:
    """读取 AGRI FDI + GEO 文件，返回 dict(lat, lon, VZA, SZA, BT)。"""
    try:
        if "_FDI-_" not in agri_file.name:
            return None
        geo_file = _paired_geo_file(agri_file)
        if not geo_file.exists():
            log.warning("GEO file missing for %s", agri_file.name)
            return None

        fname = agri_file.name
        if "FY4B" in fname:
            channel_indices = [c - 1 for c in cfg.AGRI_PHYSICAL_CHANNELS_B]
        else:
            channel_indices = [c - 1 for c in cfg.AGRI_PHYSICAL_CHANNELS_A]

        lat, lon, vza, sza = _read_geo(geo_file)

        bt_list = []
        with h5py.File(agri_file, "r") as ff:
            for idx in channel_indices:
                ch = idx + 1
                raw = _h5_read_first(ff, [f"NOMChannel{ch:02d}",
                                          f"Data/NOMChannelBT{ch:02d}",
                                          f"Data/NOMChannel{ch:02d}"]).astype(np.float32)
                lut = _h5_read_first_or(ff, [f"CALChannel{ch:02d}",
                                             f"Calibration/CALChannel{ch:02d}"],
                                        np.array([], np.float32))
                if lut.size > 0:
                    bt = _lut_calibrate(raw, lut)
                else:
                    bt = raw.astype(np.float32)
                    bt[bt > 60000] = np.nan
                    if np.isfinite(bt).any() and np.nanmedian(bt[np.isfinite(bt)]) > 500:
                        bt /= 100.0
                bt_list.append(bt)

        BT = np.stack(bt_list, axis=-1)
        if lat.shape != BT.shape[:2]:
            log.warning("Shape mismatch %s GEO=%s BT=%s", agri_file.name, lat.shape, BT.shape[:2])
            return None
        if not np.isfinite(lat).any():
            return None

        return dict(lat=lat, lon=lon, VZA=vza, SZA=sza, BT=BT)
    except Exception as exc:
        log.warning("read_agri_scene failed %s: %s", agri_file, exc)
        return None


# ---------------------------------------------------------------------------
# GPM 读取
# ---------------------------------------------------------------------------

def read_gpm_file(gpm_path: Path) -> Optional[dict]:
    """
    读取 GPM IMERG V07B HDF5 文件。

    Returns
    -------
    dict with keys:
        precip      : (1800, 3600) float32 降水率 mm/h
        lat          : (1800,)     float32 纬度
        lon          : (3600,)     float32 经度
        time_centre  : datetime    文件中心时间
        quality      : (1800, 3600) float32 降水质量指数 (可选)
    """
    try:
        with h5py.File(gpm_path, "r") as f:
            grid = f["Grid"]
            precip = grid["precipitation"][()].astype(np.float32)
            lat    = grid["lat"][()].astype(np.float32)
            lon    = grid["lon"][()].astype(np.float32)

            # 时间：Unix epoch seconds → datetime
            t = grid["time"][()]
            if isinstance(t, np.ndarray):
                t = float(t.ravel()[0])
            time_centre = datetime.fromtimestamp(t, tz=timezone.utc)

            # 质量指数（可选）
            quality = None
            if "precipitationQualityIndex" in grid:
                quality = grid["precipitationQualityIndex"][()].astype(np.float32)

            # 预处理：transpose 为 (lat, lon) 并处理 fill value
            if precip.ndim == 3:
                precip = precip[0]  # (1, 3600, 1800) → (3600, 1800)
            if quality is not None and quality.ndim == 3:
                quality = quality[0]

            # GPM 数据是 (lon, lat) → 转为 (lat, lon)
            if precip.shape == (cfg.GPM_LON_SIZE, cfg.GPM_LAT_SIZE):
                precip = precip.T   # → (1800, 3600)
                if quality is not None and quality.shape == (cfg.GPM_LON_SIZE, cfg.GPM_LAT_SIZE):
                    quality = quality.T

            # 填充值 → NaN
            precip = precip.copy()
            precip[precip < -9000] = np.nan

            if quality is not None:
                quality = quality.copy()
                quality[quality < -9000] = np.nan

        return dict(
            precip=precip,
            lat=lat,
            lon=lon,
            time_centre=time_centre,
            quality=quality,
        )
    except Exception as exc:
        log.warning("read_gpm_file failed %s: %s", gpm_path, exc)
        return None


# ---------------------------------------------------------------------------
# 夜间检测
# ---------------------------------------------------------------------------

def is_nighttime(sza: np.ndarray) -> bool:
    """
    判断 AGRI 景是否为夜间。
    策略：若有效 SZA 的中位数 > 85°，视为夜间。
    """
    valid = np.isfinite(sza)
    if not valid.any():
        return False
    return bool(np.nanmedian(sza[valid]) > 85.0)


# ---------------------------------------------------------------------------
# HDF5 写出 (tile 格式)
# ---------------------------------------------------------------------------

def _create_ds(grp: h5py.Group, name: str, tail: tuple, dtype=np.float32):
    return grp.create_dataset(
        name, shape=(0,) + tail, maxshape=(None,) + tail,
        dtype=dtype, compression="gzip", compression_opts=4, chunks=(1,) + tail,
    )


def _append(ds: h5py.Dataset, val: np.ndarray):
    n = ds.shape[0]
    ds.resize((n + 1,) + ds.shape[1:])
    ds[n] = val


def write_gpm_fused_tiles(
    out_path: Path,
    samples: List[dict],
    agri_dt: datetime,
    gpm_dt: datetime,
    mode: str,
) -> int:
    """
    将 GPM+AGRI 融合 tile 写为 HDF5。

    Parameters
    ----------
    out_path : 输出 HDF5 路径
    samples : list of dict, each with:
        agri_tile  : (9, 128, 128) float32  7 BT + 2 geo
        gpm_tile   : (1, 128, 128) float32  interpolated precipitation
        lat_center : float
        lon_center : float
        has_rain   : bool
    agri_dt : AGRI 扫描时间
    gpm_dt : GPM 中心时间
    mode : "train" / "val" / "test"
    """
    if not samples:
        raise RuntimeError(f"No samples for {agri_dt:%Y%m%d_%H%M%S}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.h5")
    if tmp.exists():
        tmp.unlink()

    th, tw = cfg.TILE_SIZE
    C = cfg.IN_CHANNELS  # 9

    with h5py.File(tmp, "w") as f:
        f.attrs.update({
            "format": "tiles_v1",
            "task": "precip_regression",
            "agri_datetime": agri_dt.strftime("%Y%m%d%H%M%S"),
            "gpm_datetime": gpm_dt.strftime("%Y%m%d%H%M%S"),
            "tile_size": list(cfg.TILE_SIZE),
            "tile_stride": list(cfg.TILE_STRIDE),
            "in_channels": C,
            "mode": mode,
        })

        s = f.create_group("Tiles")
        ds_agri  = _create_ds(s, "agri",  (C, th, tw))
        ds_gpm   = _create_ds(s, "gpm",   (1, th, tw))
        ds_lat   = _create_ds(s, "lat_center", ())
        ds_lon   = _create_ds(s, "lon_center", ())
        ds_rain  = _create_ds(s, "has_rain", (), np.bool_)

        for sample in samples:
            _append(ds_agri,  sample["agri_tile"].astype(np.float32))
            _append(ds_gpm,   sample["gpm_tile"].astype(np.float32))
            _append(ds_lat,   np.float32(sample["lat_center"]))
            _append(ds_lon,   np.float32(sample["lon_center"]))
            _append(ds_rain,  bool(sample["has_rain"]))

        f.attrs["num_samples"] = int(ds_agri.shape[0])

    _validate_gpm_fused_tiles(tmp, out_path, agri_dt)
    return len(samples)


def _validate_gpm_fused_tiles(tmp: Path, final: Path, agri_dt: datetime):
    with h5py.File(tmp, "r") as f:
        assert f.attrs.get("format") == "tiles_v1"
        assert f.attrs.get("agri_datetime") == agri_dt.strftime("%Y%m%d%H%M%S")
        s = f["Tiles"]
        n = int(s["agri"].shape[0])
        assert n > 0
        assert s["gpm"].shape[0] == n
        assert s["lat_center"].shape[0] == n

    final.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(final)
    log.info("Finalised %s (%d tiles)", final.name, n)
