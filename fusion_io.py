"""
fusion_io.py
============
AGRI 和 MYD06 文件读取 + HDF5 写出，与聚合逻辑完全解耦。

主要函数
--------
  read_agri_scene(path)   → dict(lat, lon, VZA, SZA, BT)
  read_myd06(path)        → dict(lat_5km, lon_5km, CLP_1km, CER_1km, COT_1km,
                                  CTH_5km, scan_time_1km, _dt_min, ...)
  write_fused_hdf5(...)   → 写出 samples_v2 格式 HDF5

注：本模块对 pyhdf / h5py 的依赖都在这里，fusion_core.py 只做纯数值计算。
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

    coff = 0.5 * (np.nanmin(col) + np.nanmax(col))
    loff = 0.5 * (np.nanmin(line) + np.nanmax(line))

    col_step  = float(np.nanmedian(np.diff(np.unique(col[0,  :][np.isfinite(col[0,  :])])))) if np.isfinite(col[0, :]).any()   else 1.0
    line_step = float(np.nanmedian(np.diff(np.unique(line[:, 0][np.isfinite(line[:, 0])])))) if np.isfinite(line[:, 0]).any() else 1.0

    x_pix = samp_ang * 1e-6 / (col_step  if col_step  > 1.5 else 1.0)
    y_pix = step_ang * 1e-6 / (line_step if line_step > 1.5 else 1.0)

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


def _read_scan_time_as_offset_min(sd: SD, file_dt: datetime) -> Optional[np.ndarray]:
    """
    读取 MYD06 Scan_Start_Time → 相对文件名时间的分钟偏移（像元级，1km 分辨率）。
    这是像元级时间的唯一来源；失败时返回 None（外部 fallback 到文件名时间）。
    """
    try:
        sst = sd.select("Scan_Start_Time")[:]          # (n_scans,) TAI93 秒
        tai_epoch = datetime(1993, 1, 1)
        offsets_min = np.array([
            (tai_epoch + timedelta(seconds=float(t)) - file_dt).total_seconds() / 60.0
            for t in sst
        ], dtype=np.float32)

        clp_shape = sd.select(cfg.MODIS_VARS["CLP"])[:].shape
        n_rows_1km, n_cols_1km = clp_shape

        # 每条 5km scan → 5 条 1km scan（重复，不插值）
        scan_1km = np.repeat(offsets_min, 5)
        if scan_1km.shape[0] < n_rows_1km:
            scan_1km = np.pad(scan_1km, (0, n_rows_1km - scan_1km.shape[0]),
                              mode="edge")
        scan_1km = scan_1km[:n_rows_1km]
        return np.tile(scan_1km[:, np.newaxis], (1, n_cols_1km)).astype(np.float32)
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
    当前配置下 4 个主监督量都按 1km 处理：
    - Cloud_Mask_1km 统一过滤 CLP/CER/COT/CTH
    - uncertainty 过滤 CER/COT
    - Optical phase 仅用于 CER/COT 质量约束
    - cloud-top 辅助变量仅用于 CTH 质量约束
    """
    clp = clp.astype(np.float32, copy=True)
    cer = cer.astype(np.float32, copy=True)
    cot = cot.astype(np.float32, copy=True)
    cth = cth.astype(np.float32, copy=True)
    clp[clp < 0] = np.nan

    if not cfg.MODIS_FILTER_WEAK_QUALITY:
        return dict(CLP=clp, CER=cer, COT=cot, CTH=cth)

    if cm_1km is not None and cm_1km.shape == clp.shape:
        ok = np.isin(cm_1km, np.asarray(cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_1KM))
        clp[~ok] = np.nan
        cer[~ok] = np.nan
        cot[~ok] = np.nan
        cth[~ok] = np.nan

    if cer_unc is not None and cer_unc.shape == cer.shape:
        cer[(~np.isfinite(cer_unc)) | (cer_unc > cfg.MODIS_MAX_CER_UNCERTAINTY_PCT)] = np.nan
    if cot_unc is not None and cot_unc.shape == cot.shape:
        cot[(~np.isfinite(cot_unc)) | (cot_unc > cfg.MODIS_MAX_COT_UNCERTAINTY_PCT)] = np.nan

    if bool(getattr(cfg, "MODIS_REQUIRE_OPTICAL_PHASE_FOR_COP", False)) and clp_opt_raw is not None and clp_opt_raw.shape == clp.shape:
        allowed = np.asarray(getattr(cfg, "MODIS_ALLOWED_OPTICAL_PHASES_FOR_COP", ()), dtype=np.int32)
        ok_opt = np.isin(clp_opt_raw.astype(np.int32), allowed)
        cer[~ok_opt] = np.nan
        cot[~ok_opt] = np.nan

        if bool(getattr(cfg, "MODIS_REQUIRE_PHASE_AGREEMENT", False)):
            opt_map = np.vectorize(lambda x: cfg.MODIS_OPTICAL_PHASE_MAP.get(int(x), -1))(clp_opt_raw).astype(np.float32)
            comparable = np.isfinite(clp) & np.isfinite(opt_map) & (clp >= 0) & (opt_map >= 0)
            agree = comparable & (clp == opt_map)
            cer[comparable & ~agree] = np.nan
            cot[comparable & ~agree] = np.nan

    if bool(getattr(cfg, "MODIS_REQUIRE_CTH_AUX", False)):
        cth_ok = np.isfinite(cth)
        if ctp_1km is not None and ctp_1km.shape == cth.shape:
            cth_ok &= np.isfinite(ctp_1km) & (ctp_1km > 0)
        if ctt_1km is not None and ctt_1km.shape == cth.shape:
            cth_ok &= np.isfinite(ctt_1km)
        if ctm_1km is not None and ctm_1km.shape == cth.shape:
            allowed_methods = np.asarray(getattr(cfg, "MODIS_ALLOWED_CLOUD_TOP_METHODS", ()), dtype=np.int32)
            cth_ok &= np.isin(ctm_1km.astype(np.int32), allowed_methods)
        cth[~cth_ok] = np.nan

    return dict(CLP=clp, CER=cer, COT=cot, CTH=cth)

    # 1km Cloud Mask -> CLP/CER/COT
    if cm_1km is not None and cm_1km.shape == clp.shape:
        ok = np.isin(cm_1km, np.asarray(cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_1KM))
        clp[~ok] = np.nan
        cer[~ok] = np.nan
        cot[~ok] = np.nan
    # 5km Cloud Mask -> CTH
    if cm_5km is not None and cm_5km.shape == cth.shape:
        ok5 = np.isin(cm_5km, np.asarray(cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_5KM))
        cth[~ok5] = np.nan
    # 不确定度过滤
    if cer_unc is not None and cer_unc.shape == cer.shape:
        cer[(~np.isfinite(cer_unc)) | (cer_unc > cfg.MODIS_MAX_CER_UNCERTAINTY_PCT)] = np.nan
    if cot_unc is not None and cot_unc.shape == cot.shape:
        cot[(~np.isfinite(cot_unc)) | (cot_unc > cfg.MODIS_MAX_COT_UNCERTAINTY_PCT)] = np.nan

    return dict(CLP=clp, CER=cer, COT=cot, CTH=cth)


def read_myd06(modis_file: Path, agri_dt: Optional[datetime] = None) -> Optional[dict]:
    """
    读取 MYD06 文件，返回保留 2D 空间形状的变量字典。

    当前配置下 4 个主监督量都按 1km SDS 处理；若产品只提供 5km 经纬度，
    则在聚合阶段按现有逻辑上采样到 1km 标签形状。
    """
    try:
        sd = SD(str(modis_file), SDC.READ)

        lat_5km = sd.select("Latitude")[:]
        lon_5km = sd.select("Longitude")[:]

        clp_raw = sd.select(cfg.MODIS_VARS["CLP"])[:].astype(np.int32)
        clp_1km = np.vectorize(lambda x: cfg.MODIS_PHASE_MAP.get(int(x), -1))(clp_raw).astype(np.float32)
        cer_1km = _sds_scaled(sd, cfg.MODIS_VARS["CER"], cfg.MODIS_SCALE["CER"])
        cot_1km = _sds_scaled(sd, cfg.MODIS_VARS["COT"], cfg.MODIS_SCALE["COT"])
        cth_1km = _sds_scaled(sd, cfg.MODIS_VARS["CTH"], cfg.MODIS_SCALE["CTH"])

        cm_1km = _decode_cloud_mask(_sds_optional(sd, "Cloud_Mask_1km"))
        cer_unc = _sds_optional(sd, "Cloud_Effective_Radius_Uncertainty", use_sds_sf=True)
        cot_unc = _sds_optional(sd, "Cloud_Optical_Thickness_Uncertainty", use_sds_sf=True)

        qc_vars = getattr(cfg, "MODIS_QC_VARS", {}) or {}
        clp_opt_raw = _sds_optional(sd, qc_vars.get("CLP_OPT", "")) if qc_vars.get("CLP_OPT") else None
        ctp_1km = _sds_optional(sd, qc_vars.get("CTP", ""), use_sds_sf=True) if qc_vars.get("CTP") else None
        ctt_1km = _sds_optional(sd, qc_vars.get("CTT", ""), use_sds_sf=True) if qc_vars.get("CTT") else None
        ctm_1km = _sds_optional(sd, qc_vars.get("CTM", "")) if qc_vars.get("CTM") else None

        file_dt = parse_modis_datetime(modis_file.name)
        ref_dt = agri_dt or file_dt
        scan_t = None
        fallback = True
        if ref_dt is not None:
            scan_t = _read_scan_time_as_offset_min(sd, ref_dt)
            fallback = (scan_t is None)

        sd.end()

        filt = _apply_qa_filter(
            clp_1km,
            cer_1km,
            cot_1km,
            cth_1km,
            cm_1km,
            cer_unc,
            cot_unc,
            clp_opt_raw=clp_opt_raw,
            ctp_1km=ctp_1km,
            ctt_1km=ctt_1km,
            ctm_1km=ctm_1km,
        )

        if fc.FUSION_LOG_PIXEL_STATS:
            log.info(
                "MYD06 qa | %s | clp=%d cer=%d cot=%d cth=%d | scantime=%s",
                modis_file.name,
                int(np.isfinite(filt["CLP"]).sum()),
                int(np.isfinite(filt["CER"]).sum()),
                int(np.isfinite(filt["COT"]).sum()),
                int(np.isfinite(filt["CTH"]).sum()),
                "pixel-level" if not fallback else "file-level-fallback",
            )

        return dict(
            lat_5km=lat_5km,
            lon_5km=lon_5km,
            CLP_1km=filt["CLP"],
            CER_1km=filt["CER"],
            COT_1km=filt["COT"],
            CTH_1km=filt["CTH"],
            scan_time_1km=scan_t,
            _scan_time_is_fallback=fallback,
        )

    except Exception as exc:
        log.warning("read_myd06 failed %s: %s", modis_file, exc)
        return None


# ---------------------------------------------------------------------------
# 质量后处理（apply_quality_filter）
# ---------------------------------------------------------------------------

def apply_quality_filter(agri: dict, labels: dict) -> dict:
    """
    融合后的最终质量过滤。
    - 时间差 > TIME_LOW_Q → NaN
    - overlap < OVERLAP_FRAC_MIN → NaN
    - phase_consistency < PHASE_CONSISTENCY_MIN → CLP NaN
    - VZA/SZA 几何过滤（可选，由 cfg 控制）
    """
    vza, sza = agri["VZA"], agri["SZA"]
    geo_ok = (
        np.isfinite(vza) & np.isfinite(sza) &
        (vza <= cfg.MAX_VZA_DEG) & (sza <= cfg.MAX_SZA_DEG)
    )

    dt   = labels.get("MATCH_DT_MIN")
    ovlp = labels.get("OVERLAP_FRACTION")
    phcon= labels.get("PHASE_CONSISTENCY")

    time_ok    = (np.isfinite(dt) & (dt <= fc.TIME_LOW_Q_MIN)) if dt   is not None else np.ones(labels["CLP"].shape, bool)
    overlap_ok = (np.isfinite(ovlp) & (ovlp >= fc.OVERLAP_FRAC_MIN))  if ovlp is not None else np.ones(labels["CLP"].shape, bool)
    phase_ok   = (~np.isfinite(phcon)) | (phcon >= fc.PHASE_CONSISTENCY_MIN) if phcon is not None else np.ones(labels["CLP"].shape, bool)

    clp_raw = labels["CLP"].copy()
    clp_ok  = (
        np.isfinite(clp_raw) & (clp_raw >= 0) & (clp_raw < cfg.CLP_CLASSES) &
        time_ok & overlap_ok & phase_ok
    )
    if cfg.CLP_USE_GEO_FILTER:
        clp_ok &= geo_ok
    labels["CLP"] = np.where(clp_ok, clp_raw, np.nan)

    cloudy = np.isfinite(labels["CLP"]) & (labels["CLP"] > 0)

    for k, lo, hi in [("CER", 0, 100), ("COT", 0, 200), ("CTH", 0, 25000)]:
        raw = labels[k].copy()
        ok  = cloudy & np.isfinite(raw) & (raw >= lo) & (raw <= hi) & time_ok & overlap_ok
        if cfg.REG_USE_GEO_FILTER:
            ok &= geo_ok
        labels[k] = np.where(ok, raw, np.nan)

    valid = np.isfinite(labels["CLP"])
    for k in ["MATCH_DT_MIN", "OVERLAP_FRACTION", "CLOUD_FRACTION", "PHASE_CONSISTENCY"]:
        if k in labels:
            labels[k] = np.where(valid, labels[k], np.nan)
    for k in ["VALID_PIX_1KM", "VALID_PIX_5KM"]:
        if k in labels:
            labels[k] = np.where(valid, labels[k], 0)
    if "SAMPLE_WEIGHT" in labels:
        labels["SAMPLE_WEIGHT"] = np.where(valid, labels["SAMPLE_WEIGHT"], 0.0)

    for k in ["CER", "COT", "CTH"]:
        labels[k][~cloudy] = np.nan

    if fc.FUSION_LOG_PIXEL_STATS:
        log.info(
            "post-qc | clp=%d cer=%d cot=%d cth=%d | "
            "clp_pct=%.1f%% wt_mean=%.3f dt_mean=%.2fmin",
            int(np.isfinite(labels["CLP"]).sum()),
            int(np.isfinite(labels["CER"]).sum()),
            int(np.isfinite(labels["COT"]).sum()),
            int(np.isfinite(labels["CTH"]).sum()),
            100.0 * np.isfinite(labels["CLP"]).mean(),
            float(np.nanmean(labels.get("SAMPLE_WEIGHT", np.array([np.nan])))),
            float(np.nanmean(labels.get("MATCH_DT_MIN",  np.array([np.nan])))),
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
    clp, cer, cot, cth = labels["CLP"], labels["CER"], labels["COT"], labels["CTH"]
    H, W = clp.shape
    sh, sw = (max(1, ph//2), max(1, pw//2)) if mode == "train" else (ph, pw)

    h_pos = list(range(0, H - ph + 1, sh))
    if h_pos and h_pos[-1] != H - ph:
        h_pos.append(H - ph)
    w_pos = list(range(0, W - pw + 1, sw))
    if w_pos and w_pos[-1] != W - pw:
        w_pos.append(W - pw)

    for i in h_pos:
        for j in w_pos:
            keep, counts, _ = patch_passes_supervision(
                clp[i:i+ph, j:j+pw], cer[i:i+ph, j:j+pw],
                cot[i:i+ph, j:j+pw], cth[i:i+ph, j:j+pw],
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
            "time_high_q_min": fc.TIME_HIGH_Q_MIN,
            "time_low_q_min":  fc.TIME_LOW_Q_MIN,
            "overlap_frac_min": fc.OVERLAP_FRAC_MIN,
            "cloud_frac_min":   fc.CLOUD_FRAC_MIN_CLOUDY,
            "phase_consist_min":fc.PHASE_CONSISTENCY_MIN,
            "min_valid_label_px": thresh["min_valid_label_pixels"],
            "min_valid_cloudy_px":thresh["min_valid_cloudy_pixels"],
        })

        s = f.create_group("Samples")
        ds_agri   = _create_ds(s, "agri",   (C, ph, pw))
        ds_geo    = _create_ds(s, "geo",    (4, ph, pw))
        ds_lbl    = _create_ds(s, "labels", (4, ph, pw))
        ds_row    = _create_ds(s, "row",    (), np.int32)
        ds_col    = _create_ds(s, "col",    (), np.int32)
        ds_clppx  = _create_ds(s, "valid_clp_px",   (), np.int32)
        ds_cldpx  = _create_ds(s, "valid_cloudy_px",(), np.int32)
        ds_dt     = _create_ds(s, "max_time_diff_min",    ())
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
                labels["CER"][i:i+ph, j:j+pw],
                labels["COT"][i:i+ph, j:j+pw],
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
            _append(ds_dt,    np.float32(_pmax("MATCH_DT_MIN")))
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

        # 时间差上限校验
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

        g = f.create_group("AGRI/Geolocation")
        for k in ["lat", "lon", "VZA", "SZA"]:
            g.create_dataset(k, data=agri[k], compression="gzip", compression_opts=4)

        bt = f.create_group("AGRI/BT")
        for ci, ch in enumerate(cfg.AGRI_BT_CHANNEL_INDICES):
            bt.create_dataset(f"ch{ch+1:02d}", data=agri["BT"][..., ci],
                              compression="gzip", compression_opts=4)

        lbl = f.create_group("Labels")
        for k in ["CLP", "CER", "COT", "CTH"]:
            lbl.create_dataset(k, data=labels[k], compression="gzip", compression_opts=4)

        qa = f.create_group("QA")
        for k in ["MATCH_DT_MIN", "OVERLAP_FRACTION", "VALID_PIX_1KM",
                  "VALID_PIX_5KM", "CLOUD_FRACTION", "PHASE_CONSISTENCY", "SAMPLE_WEIGHT"]:
            if k in labels:
                qa.create_dataset(k, data=labels[k], compression="gzip", compression_opts=4)

    log.debug("Wrote full-disk %s", out_path)
