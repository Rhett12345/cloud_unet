"""
fusion_io.py
============
AGRI 和 MYD06 文件读取 + HDF5 写出，与聚合逻辑完全解耦。

主要函数
--------
  read_agri_scene(path)   → dict(lat, lon, VZA, SZA, BT)
  read_myd06(path)        → dict(lat_5km, lon_5km, lat_1km, lon_1km, CLP_1km,
                                  CTH_1km, scan_time_1km, ...)
  read_agri_l2_clp(path)  → CLP 2D array (phase-remapped, float32)
  read_agri_l2_cth(path)  → CTH 2D array (fill-filtered, float32)
  write_fused_hdf5(...)   → 写出 samples_v2 格式 HDF5

注：本模块对 pyhdf / h5py / netCDF4 的依赖都在这里，fusion_core.py 只做纯数值计算。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
from pyhdf.SD import SD, SDC

import config as cfg
import fusion_config as fc
from sample_filters import get_patch_supervision_thresholds, patch_passes_supervision

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


def parse_modis_datetime(filename: str) -> Optional[datetime]:
    """MYD06_L2.AYYYYDDD.HHMM.*"""
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


def find_day_folders(root: Path, dates: list) -> list:
    if dates:
        return [root / d for d in dates if (root / d).is_dir()]
    return sorted(p for p in root.iterdir() if p.is_dir())


def find_matching_modis(agri_dt: datetime, modis_files: list) -> list:
    """
    返回文件名时间差 <= TIME_LOW_Q_MIN 的 MYD06 文件（粗筛）。
    像元级精细时间过滤在聚合阶段进行。
    """
    td = timedelta(minutes=fc.TIME_LOW_Q_MIN)
    candidates = []
    for f in modis_files:
        mdt = parse_modis_datetime(f.name)
        if mdt and abs(mdt - agri_dt) <= td:
            candidates.append((abs(mdt - agri_dt), f))
    return [f for _, f in sorted(candidates)]


def find_matching_myd03(myd06_file: Path, myd03_files: list) -> Optional[Path]:
    """返回与 MYD06 文件名时间相同的 MYD03 1km 经纬度文件。"""
    myd06_dt = parse_modis_datetime(myd06_file.name)
    if myd06_dt is None:
        return None
    candidates = []
    for f in myd03_files:
        mdt = parse_modis_datetime(f.name)
        if mdt == myd06_dt:
            candidates.append(f)
    return sorted(candidates)[0] if candidates else None


# ---------------------------------------------------------------------------
# AGRI 读取
# ---------------------------------------------------------------------------

def _paired_geo_file(fdi: Path) -> Path:
    return Path(str(fdi).replace("_FDI-_", "_GEO-_"))


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

    # ── 图像中心：从元数据读取，而非从有效数据范围反算 ──
    # ColumnNumber/LineNumber 在地球圆盘边缘有大量 NaN，nanmin/nanmax
    # 会给出错误的不对称中心（如 (0+2732)/2=1366 而非 1373.5），
    # 导致经纬度整体偏移 ~30-40 km。
    begin_pixel = _attr_scalar(gf, "Begin Pixel Number", 0.0)
    end_pixel   = _attr_scalar(gf, "End Pixel Number", 2747.0)
    begin_line  = _attr_scalar(gf, "Begin Line Number", 0.0)
    end_line    = _attr_scalar(gf, "End Line Number", 2747.0)
    coff = (begin_pixel + end_pixel) / 2.0
    loff = (begin_line + end_line) / 2.0

    # ── 扫描步长：优先用中间行/列，row0/col0 可能在太空 ──
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
            lat = _h5_read_first(gf, ["Geolocation/NOMLatitude", "NOMLatitude", "Latitude"]).astype(np.float32)
            lon = _h5_read_first(gf, ["Geolocation/NOMLongitude", "Geolocation/NOMlongitude",
                                       "NOMLongitude", "Longitude"]).astype(np.float32)
        except KeyError:
            lat, lon = _derive_latlon(gf)

        vza = _h5_read_first(gf, ["Geolocation/NOMSatelliteZenith", "NOMSatelliteZenith", "VZA"]).astype(np.float32)
        sza = _h5_read_first(gf, ["Geolocation/NOMSunZenith", "NOMSunZenith", "SZA"]).astype(np.float32)

    for arr in [lat, lon, vza, sza]:
        arr[(arr > 1e4) | (arr < -1e4)] = np.nan
    if np.isfinite(vza).any() and np.nanmax(np.abs(vza)) > 180:
        vza /= 100.0
    if np.isfinite(sza).any() and np.nanmax(np.abs(sza)) > 180:
        sza /= 100.0

    lon = _wrap_lon(lon)
    return lat, lon, vza, sza


def read_agri_scene(agri_file: Path) -> Optional[dict]:
    """读取 AGRI FDI + GEO 文件，返回 dict(lat, lon, VZA, SZA, BT)。"""
    try:
        if "_FDI-_" not in agri_file.name:
            return None
        geo_file = _paired_geo_file(agri_file)
        if not geo_file.exists():
            log.warning("GEO file missing for %s", agri_file.name)
            return None

        lat, lon, vza, sza = _read_geo(geo_file)

        bt_list = []
        with h5py.File(agri_file, "r") as ff:
            for idx in cfg.AGRI_BT_CHANNEL_INDICES:
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
# AGRI L2 读取（NetCDF）
# ---------------------------------------------------------------------------

def _extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """从 FY4A 文件名中提取 YYYYMMDDHHMMSS 时间戳。"""
    m = re.search(r"_NOM_(\d{8})(\d{6})_", filename)
    if m:
        return m.group(1) + m.group(2)
    # fallback: any 14-digit sequence
    m = re.search(r"(\d{14})", filename)
    return m.group(1) if m else None


def _find_matching_l2_file(l1_fdi_path: Path, product: str) -> Optional[Path]:
    """
    根据 L1B FDI 文件名的时间戳，匹配对应 L2 CLP/CTH .NC 文件。
    product: "CLP" 或 "CTH"
    """
    ts = _extract_timestamp_from_filename(l1_fdi_path.name)
    if ts is None:
        log.warning("Cannot extract timestamp from %s", l1_fdi_path.name)
        return None

    date_str = ts[:8]  # YYYYMMDD
    l2_dir = cfg.FY4A_L2_ROOT / product / date_str
    if not l2_dir.is_dir():
        log.info("L2 %s dir not found for date %s: %s", product, date_str, l2_dir)
        return None

    # Match by start-time (first 14-digit timestamp in filename)
    # L2 filename format: FY4A-_AGRI--_N_DISK_1047E_L2-_CLP-_MULT_NOM_YYYYMMDDHHMMSS_...
    for f in sorted(l2_dir.iterdir()):
        if not f.name.endswith(".NC"):
            continue
        f_ts = _extract_timestamp_from_filename(f.name)
        if f_ts == ts:
            return f

    log.info("No matching L2 %s file found for timestamp %s in %s", product, ts, l2_dir)
    return None


def read_agri_l2_clp(nc_path: Path) -> Optional[np.ndarray]:
    """
    读取 AGRI L2 CLP NetCDF 文件，返回 phase-remapped float32 数组 (2748×2748)。
    未映射的类别（126=Space, 127=FillValue 等）设为 NaN。
    """
    try:
        import netCDF4 as nc
        ds = nc.Dataset(str(nc_path), "r")
        var = ds.variables["CLP"]
        var.set_auto_mask(False)
        raw = np.asarray(var[:], dtype=np.int16)
        ds.close()

        clp = np.full(raw.shape, np.nan, dtype=np.float32)
        phase_map = cfg.AGRI_L2_CLP_PHASE_MAP
        for src_val, dst_val in phase_map.items():
            mask = raw == src_val
            clp[mask] = float(dst_val)

        return clp
    except Exception as exc:
        log.warning("read_agri_l2_clp failed %s: %s", nc_path, exc)
        return None


def read_agri_l2_cth(nc_path: Path) -> Optional[np.ndarray]:
    """
    读取 AGRI L2 CTH NetCDF 文件，返回 float32 数组 (2748×2748)。
    过滤 FillValue、Space 和 valid_range 外的值，设为 NaN。
    """
    try:
        import netCDF4 as nc
        ds = nc.Dataset(str(nc_path), "r")
        var = ds.variables["CTH"]
        var.set_auto_mask(False)
        raw = np.asarray(var[:], dtype=np.float32)
        ds.close()

        vmin, vmax = cfg.AGRI_L2_CTH_VALID_RANGE
        fill_val = cfg.AGRI_L2_CTH_FILL_VALUE

        cth = raw.copy()
        cth[(cth <= 0) | (cth >= 65500) | ~np.isfinite(cth)] = np.nan
        cth[cth < vmin] = np.nan
        cth[cth > vmax] = np.nan

        return cth
    except Exception as exc:
        log.warning("read_agri_l2_cth failed %s: %s", nc_path, exc)
        return None


# ---------------------------------------------------------------------------
# MYD06 读取
# ---------------------------------------------------------------------------

def _sds_scaled(sd: SD, name: str, scale: float) -> np.ndarray:
    ds = sd.select(name)
    raw = ds[:].astype(np.float32)
    fv  = ds.attributes().get("_FillValue", -9999)
    raw[raw == fv] = np.nan
    return raw * scale


def _sds_optional(sd: SD, name: str, use_sds_sf: bool = False,
                  scale: float = 1.0) -> Optional[np.ndarray]:
    try:
        ds   = sd.select(name)
        raw  = ds[:].astype(np.float32)
        attr = ds.attributes()
        fv   = attr.get("_FillValue", -9999)
        raw[raw == fv] = np.nan
        if use_sds_sf:
            raw = raw * float(attr.get("scale_factor", 1.0)) + float(attr.get("add_offset", 0.0))
        else:
            raw = raw * scale
        return raw
    except Exception:
        return None




def _isin_finite_int(values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    if finite.any():
        out[finite] = np.isin(values[finite].astype(np.int32), allowed)
    return out

def _decode_cloud_mask(cm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if cm is None:
        return None
    arr = np.asarray(cm)
    if arr.ndim >= 3:
        arr = arr[..., 0]
    arr = arr.astype(np.uint8)
    status = arr & np.uint8(1)
    cloudiness = (arr >> np.uint8(1)) & np.uint8(3)
    out = cloudiness.astype(np.float32)
    out[status == 0] = np.nan
    return out


def _scan_seconds_to_offset_grid(
    seconds: np.ndarray,
    target_shape: Tuple[int, int],
    ref_dt: datetime,
) -> Optional[np.ndarray]:
    """Expand MODIS TAI93 scan seconds to a 1km grid of minute offsets from ref_dt."""
    arr = np.asarray(seconds, dtype=np.float64)
    arr[(arr <= -1.0e9) | ~np.isfinite(arr)] = np.nan
    if arr.size == 0 or not np.isfinite(arr).any():
        return None

    tai_epoch = datetime(1993, 1, 1)
    ref_seconds = (ref_dt - tai_epoch).total_seconds()
    offset_min = ((arr - ref_seconds) / 60.0).astype(np.float32)

    rows, cols = target_shape
    if offset_min.ndim == 1:
        row_repeat = max(1, int(np.ceil(rows / max(offset_min.shape[0], 1))))
        row_vals = np.repeat(offset_min, row_repeat)[:rows]
        if row_vals.shape[0] < rows:
            row_vals = np.pad(row_vals, (0, rows - row_vals.shape[0]), mode="edge")
        return np.tile(row_vals[:, np.newaxis], (1, cols)).astype(np.float32)

    if offset_min.ndim == 2:
        rh = max(1, int(np.ceil(rows / max(offset_min.shape[0], 1))))
        rw = max(1, int(np.ceil(cols / max(offset_min.shape[1], 1))))
        grid = np.repeat(np.repeat(offset_min, rh, axis=0), rw, axis=1)
        if grid.shape[0] < rows:
            grid = np.pad(grid, ((0, rows - grid.shape[0]), (0, 0)), mode="edge")
        if grid.shape[1] < cols:
            grid = np.pad(grid, ((0, 0), (0, cols - grid.shape[1])), mode="edge")
        return grid[:rows, :cols].astype(np.float32)

    return None


def _read_scan_time_as_offset_min(
    sd: SD,
    ref_dt: datetime,
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    读取 MYD06 Scan_Start_Time → 相对 AGRI 时间的分钟偏移（1km 分辨率）。
    MYD06 常见为 5km 二维时间场，需重复展开到 1km 标签形状。
    """
    try:
        sst = sd.select("Scan_Start_Time")[:]
        return _scan_seconds_to_offset_grid(sst, target_shape, ref_dt)
    except Exception as exc:
        log.debug("Scan_Start_Time unavailable: %s", exc)
        return None


def _apply_qa_filter(
    clp: np.ndarray,
    cer: np.ndarray,
    cot: np.ndarray,
    cth: np.ndarray,
    cm_1km: Optional[np.ndarray],
    cer_unc: Optional[np.ndarray],
    cot_unc: Optional[np.ndarray],
    clp_opt_raw: Optional[np.ndarray] = None,
    ctp_1km: Optional[np.ndarray] = None,
    ctt_1km: Optional[np.ndarray] = None,
    ctm_1km: Optional[np.ndarray] = None,
) -> dict:
    """
    前期弱质量过滤（保留 2D 形状）。
    CLP + CTH only（CER/COT 已移除）：
    - Cloud_Mask_1km 统一过滤 CLP/CTH
    - cloud-top 辅助变量仅用于 CTH 质量约束（可选）
    """
    clp = clp.astype(np.float32, copy=True)
    cth = cth.astype(np.float32, copy=True)
    clp[clp < 0] = np.nan

    if not cfg.MODIS_FILTER_WEAK_QUALITY:
        return dict(CLP=clp, CTH=cth)

    if cm_1km is not None and cm_1km.shape == clp.shape:
        clp_allowed = np.asarray(
            getattr(cfg, "MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP", cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_1KM)
        )
        reg_allowed = np.asarray(
            getattr(cfg, "MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG", cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_1KM)
        )
        clp_ok = np.isin(cm_1km, clp_allowed)
        reg_ok = np.isin(cm_1km, reg_allowed)
        clp[~clp_ok] = np.nan
        cth[~reg_ok] = np.nan

    if bool(getattr(cfg, "MODIS_REQUIRE_CTH_AUX", False)):
        cth_ok = np.isfinite(cth)
        if ctp_1km is not None and ctp_1km.shape == cth.shape:
            cth_ok &= np.isfinite(ctp_1km) & (ctp_1km > 0)
        if ctt_1km is not None and ctt_1km.shape == cth.shape:
            cth_ok &= np.isfinite(ctt_1km)
        if ctm_1km is not None and ctm_1km.shape == cth.shape:
            allowed_methods = np.asarray(getattr(cfg, "MODIS_ALLOWED_CLOUD_TOP_METHODS", ()), dtype=np.int32)
            cth_ok &= _isin_finite_int(ctm_1km, allowed_methods)
        cth[~cth_ok] = np.nan

    return dict(CLP=clp, CTH=cth)


def _read_myd03_scan_time(sd: SD, target_shape: Tuple[int, int], ref_dt: datetime) -> Tuple[Optional[np.ndarray], str]:
    for name in ("EV start time", "EV_start_time", "EV_Start_Time", "EV center time", "EV_center_time"):
        try:
            grid = _scan_seconds_to_offset_grid(sd.select(name)[:], target_shape, ref_dt)
            if grid is not None:
                return grid, name
        except Exception:
            continue
    return None, "none"


def read_myd03(myd03_file: Path, ref_dt: Optional[datetime] = None) -> Optional[dict]:
    """读取 MYD03 1km Latitude / Longitude，并尽量读取扫描行时间。"""
    try:
        sd = SD(str(myd03_file), SDC.READ)
        lat_1km = sd.select("Latitude")[:].astype(np.float32)
        lon_1km = sd.select("Longitude")[:].astype(np.float32)
        scan_time = None
        scan_time_source = "none"
        if ref_dt is not None:
            scan_time, scan_time_source = _read_myd03_scan_time(sd, lat_1km.shape, ref_dt)
        sd.end()
        return {
            "lat_1km": lat_1km,
            "lon_1km": lon_1km,
            "scan_time_1km": scan_time,
            "scan_time_source": scan_time_source,
        }
    except Exception as exc:
        log.warning("read_myd03 failed %s: %s", myd03_file, exc)
        return None


def read_modis_geo_quick(modis_file: Path, myd03_file: Optional[Path] = None) -> Optional[dict]:
    """
    轻量地理预检：只读取 MODIS 经纬度，不做任何科学数据 IO。

    优先使用 MYD03 1km 经纬度（若可用），否则回退到 MYD06 5km Latitude/Longitude。
    返回的 dict 可直接传给 read_myd06 的 geo_cache 参数以避免重复读取。
    """
    # 优先 MYD03 1km（同时缓存 scan_time 避免后续重复读取）
    if myd03_file is not None:
        geo = read_myd03(myd03_file)
        if geo is not None:
            return {
                "lat_1km": geo["lat_1km"],
                "lon_1km": geo["lon_1km"],
                "scan_time_1km": geo.get("scan_time_1km"),
                "scan_time_source": geo.get("scan_time_source", "none"),
                "_geo_source": "MYD03_1KM",
            }

    # 回退到 MYD06 5km
    try:
        sd = SD(str(modis_file), SDC.READ)
        lat_5km = sd.select("Latitude")[:].astype(np.float32)
        lon_5km = sd.select("Longitude")[:].astype(np.float32)
        sd.end()
        return {
            "lat_5km": lat_5km,
            "lon_5km": lon_5km,
            "_geo_source": "MYD06_5KM",
        }
    except Exception:
        return None


def read_myd06(modis_file: Path, agri_dt: Optional[datetime] = None,
               myd03_file: Optional[Path] = None,
               geo_cache: Optional[dict] = None) -> Optional[dict]:
    """
    读取 MYD06 文件，返回保留 2D 空间形状的变量字典。

    若提供 geo_cache（来自 read_modis_geo_quick），则跳过经纬度读取。
    """
    try:
        sd = SD(str(modis_file), SDC.READ)

        # ── 经纬度：优先用 geo_cache，否则从文件读取 ──
        lat_1km = None
        lon_1km = None
        scan_t = None
        scan_time_source = "none"
        fallback = True
        geo_source = "MYD06_5KM_REPEAT"

        if geo_cache is not None and geo_cache.get("lat_5km") is not None:
            lat_5km = geo_cache["lat_5km"]
            lon_5km = geo_cache["lon_5km"]
            geo_source = geo_cache.get("_geo_source", "MYD06_5KM")
        else:
            lat_5km = sd.select("Latitude")[:]
            lon_5km = sd.select("Longitude")[:]

        clp_raw = sd.select(cfg.MODIS_VARS["CLP"])[:].astype(np.int32)
        clp_1km = np.vectorize(lambda x: cfg.MODIS_PHASE_MAP.get(int(x), -1))(clp_raw).astype(np.float32)
        cth_1km = _sds_scaled(sd, cfg.MODIS_VARS["CTH"], cfg.MODIS_SCALE["CTH"])

        cm_1km = _decode_cloud_mask(_sds_optional(sd, "Cloud_Mask_1km"))

        # CER/COT removed — no uncertainty SDS needed
        # QC aux vars (CTP/CTT/CTM) kept for optional CTH filtering
        qc_vars = getattr(cfg, "MODIS_QC_VARS", {}) or {}
        ctp_1km = _sds_optional(sd, qc_vars.get("CTP", ""), use_sds_sf=True) if qc_vars.get("CTP") else None
        ctt_1km = _sds_optional(sd, qc_vars.get("CTT", ""), use_sds_sf=True) if qc_vars.get("CTT") else None
        ctm_1km = _sds_optional(sd, qc_vars.get("CTM", "")) if qc_vars.get("CTM") else None

        file_dt = parse_modis_datetime(modis_file.name)
        ref_dt = agri_dt or file_dt
        myd06_scan_t = None
        if ref_dt is not None:
            myd06_scan_t = _read_scan_time_as_offset_min(sd, ref_dt, clp_1km.shape)

        sd.end()

        # ── MYD03 1km 经纬度：优先用 geo_cache ──
        if geo_cache is not None and geo_cache.get("lat_1km") is not None:
            lat_1km = geo_cache["lat_1km"]
            lon_1km = geo_cache["lon_1km"]
            if lat_1km.shape == clp_1km.shape and lon_1km.shape == clp_1km.shape:
                geo_source = geo_cache.get("_geo_source", "MYD03_1KM")
                scan_t = geo_cache.get("scan_time_1km")
                scan_time_source = geo_cache.get("scan_time_source", "none")
                fallback = (scan_t is None)
            else:
                lat_1km = None
                lon_1km = None
        elif myd03_file is not None:
            geo_1km = read_myd03(myd03_file, ref_dt=ref_dt)
            if geo_1km is not None:
                lat_1km = geo_1km.get("lat_1km")
                lon_1km = geo_1km.get("lon_1km")
                if lat_1km.shape != clp_1km.shape or lon_1km.shape != clp_1km.shape:
                    log.warning(
                        "MYD03 shape mismatch %s: lat=%s lon=%s label=%s; fallback to MYD06 5km geo",
                        myd03_file.name, lat_1km.shape, lon_1km.shape, clp_1km.shape,
                    )
                    lat_1km = None
                    lon_1km = None
                else:
                    geo_source = "MYD03_1KM"
                    scan_t = geo_1km.get("scan_time_1km")
                    scan_time_source = geo_1km.get("scan_time_source", "none")
                    fallback = (scan_t is None)

        if lat_1km is None and fc.REQUIRE_MYD03_1KM:
            log.warning("MYD03 1km geolocation required; skip %s", modis_file.name)
            return None

        if scan_t is None and ref_dt is not None:
            scan_t = myd06_scan_t
            scan_time_source = "MYD06_Scan_Start_Time" if scan_t is not None else scan_time_source
            fallback = (scan_t is None)

        if scan_t is None and fc.REQUIRE_SCAN_TIME:
            log.warning("scan time required; skip %s", modis_file.name)
            return None

        # CER/COT removed — pass NaN arrays to keep function signature compatible
        _zeros = np.full_like(clp_1km, np.nan, dtype=np.float32)
        filt = _apply_qa_filter(
            clp_1km,
            _zeros,   # cer (placeholder)
            _zeros,   # cot (placeholder)
            cth_1km,
            cm_1km,
            None,     # cer_unc
            None,     # cot_unc
            clp_opt_raw=None,
            ctp_1km=ctp_1km,
            ctt_1km=ctt_1km,
            ctm_1km=ctm_1km,
        )

        if fc.FUSION_LOG_PIXEL_STATS:
            log.info(
                "MYD06 qa | %s | clp=%d cth=%d | scantime=%s",
                modis_file.name,
                int(np.isfinite(filt["CLP"]).sum()),
                int(np.isfinite(filt["CTH"]).sum()),
                scan_time_source if not fallback else "file-level-fallback",
            )

        return dict(
            lat_5km=lat_5km,
            lon_5km=lon_5km,
            lat_1km=lat_1km,
            lon_1km=lon_1km,
            CLP_1km=filt["CLP"],
            CTH_1km=filt["CTH"],
            scan_time_1km=scan_t,
            _scan_time_is_fallback=fallback,
            _scan_time_source=scan_time_source if not fallback else "file-level-fallback",
            _geo_source=geo_source,
        )

    except Exception as exc:
        log.warning("read_myd06 failed %s: %s", modis_file, exc)
        return None


# ---------------------------------------------------------------------------
# 质量后处理（apply_quality_filter）
# ---------------------------------------------------------------------------

def _finite_count(arr) -> int:
    return int(np.isfinite(arr).sum()) if arr is not None else 0


def _finite_stat(arr, fn):
    if arr is None:
        return np.nan
    vals = np.asarray(arr)[np.isfinite(arr)]
    if vals.size == 0:
        return np.nan
    return float(fn(vals))


def _mean_finite(arr):
    return _finite_stat(arr, np.mean)


def _build_qc_diagnostics_row(
    diagnostics: dict,
    labels: dict,
    raw_labels: dict,
    clp_raw: np.ndarray,
    time_ok: np.ndarray,
    geo_ok_clp: np.ndarray,
    reg_time_ok: np.ndarray,
    geo_ok_reg: np.ndarray,
):
    clp_base = np.isfinite(clp_raw) & (clp_raw >= 0) & (clp_raw < cfg.CLP_CLASSES)

    cumulative_after_time = clp_base & time_ok
    cumulative_after_geo = cumulative_after_time & geo_ok_clp

    reg_base = cumulative_after_geo & (clp_raw > 0)
    cumulative_after_reg_time = reg_base & reg_time_ok
    cumulative_after_reg_geo = cumulative_after_reg_time & geo_ok_reg

    row = {
        "scene_id": diagnostics.get("scene_id"),
        "agri_file": diagnostics.get("agri_file"),
        "myd06_file": diagnostics.get("myd06_file"),
        "myd03_file": diagnostics.get("myd03_file"),
        "raw_clp_valid_px": int(clp_base.sum()),
        "raw_cer_valid_px": _finite_count(raw_labels.get("CER")),
        "raw_cot_valid_px": _finite_count(raw_labels.get("COT")),
        "raw_cth_valid_px": _finite_count(raw_labels.get("CTH")),
        "time_ok_px": int(time_ok.sum()),
        "geo_ok_px": int(geo_ok_clp.sum()),
        "reg_time_ok_px": int(reg_time_ok.sum()),
        "reg_geo_ok_px": int(geo_ok_reg.sum()),
        "cumulative_base_px": int(clp_base.sum()),
        "cumulative_after_time_px": int(cumulative_after_time.sum()),
        "cumulative_after_geo_px": int(cumulative_after_geo.sum()),
        "cumulative_after_reg_time_px": int(cumulative_after_reg_time.sum()),
        "cumulative_after_reg_geo_px": int(cumulative_after_reg_geo.sum()),
        "final_clp_px": _finite_count(labels.get("CLP")),
        "final_cer_px": _finite_count(labels.get("CER")),
        "final_cot_px": _finite_count(labels.get("COT")),
        "final_cth_px": _finite_count(labels.get("CTH")),
        "time_delta_min_p50": _finite_stat(raw_labels.get("MATCH_DT_MIN"), lambda v: np.percentile(v, 50)),
        "time_delta_min_p90": _finite_stat(raw_labels.get("MATCH_DT_MIN"), lambda v: np.percentile(v, 90)),
        "time_delta_min_max": _finite_stat(raw_labels.get("MATCH_DT_MIN"), np.max),
    }
    diagnostics["row"] = row


def apply_quality_filter(agri: dict, labels: dict, diagnostics: Optional[dict] = None) -> dict:
    """
    融合后的质量控制。单像元最近邻模式下只保留有意义的门控：
      - 时间：CLP 要求 dt <= TIME_LOW_Q_MIN，回归要求 dt <= REG_TIME_MAX_MIN
      - 几何：VZA/SZA 角度限制（可选，由 cfg 控制）
      - 值域：CER [0,100], COT [0,200], CTH [0, MAX_CTH_M]
    """
    vza, sza = agri["VZA"], agri["SZA"]
    geo_ok_reg = (
        np.isfinite(vza) & np.isfinite(sza) &
        (vza <= cfg.MAX_VZA_DEG) & (sza <= cfg.MAX_SZA_DEG)
    )
    max_vza_clp = getattr(cfg, "MAX_VZA_DEG_CLP", cfg.MAX_VZA_DEG)
    max_sza_clp = getattr(cfg, "MAX_SZA_DEG_CLP", cfg.MAX_SZA_DEG)
    geo_ok_clp = (
        np.isfinite(vza) & np.isfinite(sza) &
        (vza <= max_vza_clp) & (sza <= max_sza_clp)
    )

    dt = labels.get("MATCH_DT_MIN")
    shape = labels["CLP"].shape

    time_ok = (np.isfinite(dt) & (dt <= fc.TIME_LOW_Q_MIN)) if dt is not None else np.ones(shape, bool)
    reg_time_ok = (np.isfinite(dt) & (dt <= fc.REG_TIME_MAX_MIN)) if dt is not None else np.zeros(shape, bool)

    raw_diag_labels = None
    if diagnostics is not None:
        raw_diag_labels = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in labels.items()
        }

    # ── CLP 过滤 ──
    clp_raw = labels["CLP"].copy()
    clp_ok = (
        np.isfinite(clp_raw) & (clp_raw >= 0) & (clp_raw < cfg.CLP_CLASSES) &
        time_ok
    )
    if cfg.CLP_USE_GEO_FILTER:
        clp_ok &= geo_ok_clp
    labels["CLP"] = np.where(clp_ok, clp_raw, np.nan)

    cloudy = np.isfinite(labels["CLP"]) & (labels["CLP"] > 0)

    # ── CTH 回归变量过滤 ──
    max_cth = getattr(cfg, "MAX_CTH_M", 18000)
    raw = labels["CTH"].copy()
    ok = (
        cloudy & np.isfinite(raw) & (raw >= 0) & (raw <= max_cth) &
        reg_time_ok
    )
    if cfg.REG_USE_GEO_FILTER:
        ok &= geo_ok_reg
    labels["CTH"] = np.where(ok, raw, np.nan)

    # ── 质量字段：仅在 CLP 有效处保留 ──
    valid = np.isfinite(labels["CLP"])
    for k in ["MATCH_DT_MIN", "MATCH_DT_MEAN", "MATCH_DT_MAX",
              "MATCH_DIST_MEAN_KM", "MATCH_DIST_P95_KM",
              "OVERLAP_FRACTION", "CLOUD_FRACTION", "PHASE_CONSISTENCY"]:
        if k in labels:
            labels[k] = np.where(valid, labels[k], np.nan)
    for k in ["VALID_PIX_1KM", "VALID_PIX_5KM"]:
        if k in labels:
            labels[k] = np.where(valid, labels[k], 0)
    if "SAMPLE_WEIGHT" in labels:
        labels["SAMPLE_WEIGHT"] = np.where(valid, labels["SAMPLE_WEIGHT"], 0.0)

    # 晴空像元 CTH 保持 NaN：物理上晴空没有云顶高度，
    # 0 是人为填充值，通过共享编码器反向传播会污染 CLP 分支特征。
    # 训练 loss 用 torch.isfinite(target) 做 mask，NaN 自然不入梯度。

    if diagnostics is not None:
        _build_qc_diagnostics_row(
            diagnostics, labels, raw_diag_labels, clp_raw,
            time_ok, geo_ok_clp, reg_time_ok, geo_ok_reg,
        )

    if fc.FUSION_LOG_PIXEL_STATS:
        wt_values = labels.get("SAMPLE_WEIGHT", np.array([np.nan]))
        wt_valid = wt_values[np.isfinite(wt_values)]
        dt_values = labels.get("MATCH_DT_MIN", np.array([np.nan]))
        dt_valid = dt_values[np.isfinite(dt_values)]
        log.info(
            "post-qc | clp=%d cth=%d | "
            "clp_pct=%.1f%% wt_mean=%.3f dt_mean=%.2fmin",
            int(valid.sum()),
            int(np.isfinite(labels["CTH"]).sum()),
            100.0 * valid.mean(),
            float(np.nanmean(wt_valid)) if wt_valid.size else float("nan"),
            float(np.nanmean(dt_valid)) if dt_valid.size else float("nan"),
        )

    return labels


# ---------------------------------------------------------------------------
# HDF5 写出
# ---------------------------------------------------------------------------

def _infer_split(out_dir: Path) -> str:
    parts = {p.lower() for p in out_dir.parts}
    if "train" in parts: return "train"
    if "val" in parts or "valid" in parts: return "val"
    if "test" in parts: return "test"
    return "train"


def _create_ds(grp: h5py.Group, name: str, tail: tuple, dtype=np.float32):
    return grp.create_dataset(
        name, shape=(0,) + tail, maxshape=(None,) + tail,
        dtype=dtype, compression="gzip", compression_opts=4, chunks=(1,) + tail,
    )


def _append(ds: h5py.Dataset, val: np.ndarray):
    n = ds.shape[0]
    ds.resize((n + 1,) + ds.shape[1:])
    ds[n] = val


def _iter_patch_positions(labels: dict, patch_size: tuple, mode: str):
    ph, pw = patch_size
    clp, cth = labels["CLP"], labels["CTH"]
    H, W = clp.shape
    sh, sw = ph, pw  # non-overlapping for all modes

    h_pos = list(range(0, H - ph + 1, sh))
    if h_pos and h_pos[-1] != H - ph:
        h_pos.append(H - ph)
    w_pos = list(range(0, W - pw + 1, sw))
    if w_pos and w_pos[-1] != W - pw:
        w_pos.append(W - pw)

    for i in h_pos:
        for j in w_pos:
            _nan = np.full_like(clp[i:i+ph, j:j+pw], np.nan, dtype=np.float32)
            keep, counts, _ = patch_passes_supervision(
                clp[i:i+ph, j:j+pw], _nan,   # CER (placeholder)
                _nan,                         # COT (placeholder)
                cth[i:i+ph, j:j+pw],
                mode, patch_size
            )
            if keep:
                yield i, j, counts["valid_label_pixels"], counts["valid_cloudy_pixels"]


def write_fused_samples(
    out_path: Path,
    agri: dict,
    labels: dict,
    agri_dt: datetime,
    mode: str,
) -> int:
    """
    将融合结果写为 samples_v2 格式 HDF5（每个 patch 一个样本）。
    采用先写临时文件再校验转正的安全写入策略。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.h5")
    if tmp.exists():
        tmp.unlink()

    patch_size = tuple(cfg.PATCH_SIZE)
    positions  = list(_iter_patch_positions(labels, patch_size, mode))
    if not positions:
        raise RuntimeError(f"No qualified patches for {agri_dt:%Y%m%d_%H%M%S}")

    ph, pw = patch_size
    C = agri["BT"].shape[-1]
    thresh = get_patch_supervision_thresholds(mode, patch_size)

    with h5py.File(tmp, "w") as f:
        f.attrs.update({
            "format": "samples_v2",
            "agri_datetime": agri_dt.strftime("%Y%m%d%H%M%S"),
            "agri_channels": cfg.AGRI_BT_CHANNEL_INDICES,
            "patch_size": list(patch_size),
            "scene_shape": list(agri["lat"].shape),
            "mode": mode,
            "time_low_q_min":  fc.TIME_LOW_Q_MIN,
            "reg_time_max_min": fc.REG_TIME_MAX_MIN,
            "min_valid_label_px": thresh["min_valid_label_pixels"],
            "min_valid_cloudy_px":thresh["min_valid_cloudy_pixels"],
            "clp_class_names": ",".join(getattr(cfg, "CLP_CLASS_NAMES", [])),
            "scan_time_sources": ",".join(labels.get("_scan_time_sources", ["agri_l2"])),
            "geo_sources": ",".join(labels.get("_geo_sources", ["agri_geo"])),
            "fallback_granules": int(labels.get("_fallback_granules", 0)),
            "label_source": labels.get("_label_source", "agri_l2"),
        })

        s = f.create_group("Samples")
        ds_agri   = _create_ds(s, "agri",   (C, ph, pw))
        ds_geo    = _create_ds(s, "geo",    (4, ph, pw))
        ds_lbl    = _create_ds(s, "labels", (2, ph, pw))  # CLP + CTH
        ds_row    = _create_ds(s, "row",    (), np.int32)
        ds_col    = _create_ds(s, "col",    (), np.int32)
        ds_clppx  = _create_ds(s, "valid_clp_px",   (), np.int32)
        ds_cldpx  = _create_ds(s, "valid_cloudy_px",(), np.int32)
        ds_clearpx = _create_ds(s, "valid_clear_px", (), np.int32)
        ds_waterpx = _create_ds(s, "valid_water_px", (), np.int32)
        ds_icepx   = _create_ds(s, "valid_ice_px",   (), np.int32)
        ds_dt     = _create_ds(s, "max_time_diff_min",    ())
        ds_dt_mean = _create_ds(s, "mean_time_diff_min",   ())
        ds_dist   = _create_ds(s, "p95_match_dist_km",     ())
        ds_ovlp   = _create_ds(s, "mean_overlap_frac",    ())
        ds_wt     = _create_ds(s, "mean_sample_weight",   ())
        ds_cf     = _create_ds(s, "mean_cloud_frac",      ())
        ds_pc     = _create_ds(s, "mean_phase_consist",   ())

        for i, j, n_clp, n_cld in positions:
            agri_p = agri["BT"][i:i+ph, j:j+pw, :].transpose(2,0,1).astype(np.float32)
            geo_p  = np.stack([
                agri["lat"][i:i+ph, j:j+pw],
                agri["lon"][i:i+ph, j:j+pw],
                agri["VZA"][i:i+ph, j:j+pw],
                agri["SZA"][i:i+ph, j:j+pw],
            ], axis=0).astype(np.float32)
            lbl_p  = np.stack([
                labels["CLP"][i:i+ph, j:j+pw],
                labels["CTH"][i:i+ph, j:j+pw],
            ], axis=0).astype(np.float32)

            def _pmax(k):
                a = labels.get(k)
                if a is None: return np.nan
                v = a[i:i+ph, j:j+pw]
                return float(np.nanmax(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan

            def _pmean(k):
                a = labels.get(k)
                if a is None: return np.nan
                v = a[i:i+ph, j:j+pw]
                return float(np.nanmean(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan

            _append(ds_agri,  agri_p)
            _append(ds_geo,   geo_p)
            _append(ds_lbl,   lbl_p)
            _append(ds_row,   np.int32(i))
            _append(ds_col,   np.int32(j))
            _append(ds_clppx, np.int32(n_clp))
            _append(ds_cldpx, np.int32(n_cld))
            patch_clp = labels["CLP"][i:i+ph, j:j+pw]
            _append(ds_clearpx, np.int32((np.isfinite(patch_clp) & (patch_clp == 0)).sum()))
            _append(ds_waterpx, np.int32((np.isfinite(patch_clp) & (patch_clp == 1)).sum()))
            _append(ds_icepx,   np.int32((np.isfinite(patch_clp) & (patch_clp == 2)).sum()))
            _append(ds_dt,    np.float32(_pmax("MATCH_DT_MAX")))
            _append(ds_dt_mean, np.float32(_pmean("MATCH_DT_MEAN")))
            _append(ds_dist,  np.float32(_pmax("MATCH_DIST_P95_KM")))
            _append(ds_ovlp,  np.float32(_pmean("OVERLAP_FRACTION")))
            _append(ds_wt,    np.float32(_pmean("SAMPLE_WEIGHT")))
            _append(ds_cf,    np.float32(_pmean("CLOUD_FRACTION")))
            _append(ds_pc,    np.float32(_pmean("PHASE_CONSISTENCY")))

        f.attrs["num_samples"] = int(ds_agri.shape[0])

    # 校验并转正
    _validate_and_finalize(tmp, out_path, agri_dt, mode, thresh)
    return len(positions)


def _validate_and_finalize(tmp: Path, final: Path, agri_dt: datetime,
                            mode: str, thresh: dict):
    with h5py.File(tmp, "r") as f:
        assert f.attrs.get("format") == "samples_v2"
        assert f.attrs.get("agri_datetime") == agri_dt.strftime("%Y%m%d%H%M%S")
        s = f["Samples"]
        n = int(s["agri"].shape[0])
        assert n > 0
        assert s["labels"].shape[0] == s["geo"].shape[0] == n

        # 时间差上限校验（仅在有 MODIS 时间差数据时执行）
        dt_arr = s["max_time_diff_min"][()]
        dt_ok  = dt_arr[np.isfinite(dt_arr)]
        if dt_ok.size and float(np.nanmax(dt_ok)) > fc.TIME_LOW_Q_MIN + 1e-6:
            raise ValueError(f"Time diff exceeds TIME_LOW_Q_MIN in {tmp}")

    # 原子性重命名
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(final)
    log.info("Finalised %s (%d samples)", final.name, n)


def write_full_disk_hdf5(out_path: Path, agri: dict, labels: dict, agri_dt: datetime):
    """full_disk 模式（FUSION_OUTPUT_MODE='full_disk'）：写整景 HDF5。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["agri_datetime"] = agri_dt.strftime("%Y%m%d%H%M%S")
        f.attrs["agri_channels"] = cfg.AGRI_BT_CHANNEL_INDICES
        f.attrs["scan_time_sources"] = ",".join(labels.get("_scan_time_sources", []))
        f.attrs["geo_sources"] = ",".join(labels.get("_geo_sources", []))
        f.attrs["fallback_granules"] = int(labels.get("_fallback_granules", 0))

        g = f.create_group("AGRI/Geolocation")
        for k in ["lat", "lon", "VZA", "SZA"]:
            g.create_dataset(k, data=agri[k], compression="gzip", compression_opts=4)

        bt = f.create_group("AGRI/BT")
        for ci, ch in enumerate(cfg.AGRI_BT_CHANNEL_INDICES):
            bt.create_dataset(f"ch{ch+1:02d}", data=agri["BT"][..., ci],
                              compression="gzip", compression_opts=4)

        lbl = f.create_group("Labels")
        for k in ["CLP", "CTH"]:
            lbl.create_dataset(k, data=labels[k], compression="gzip", compression_opts=4)

        qa = f.create_group("QA")
        for k in ["MATCH_DT_MIN", "MATCH_DT_MEAN", "MATCH_DT_MAX",
                  "MATCH_DIST_MEAN_KM", "MATCH_DIST_P95_KM",
                  "OVERLAP_FRACTION", "VALID_PIX_1KM",
                  "VALID_PIX_5KM", "CLOUD_FRACTION", "PHASE_CONSISTENCY", "SAMPLE_WEIGHT"]:
            if k in labels:
                qa.create_dataset(k, data=labels[k], compression="gzip", compression_opts=4)

    log.debug("Wrote full-disk %s", out_path)
