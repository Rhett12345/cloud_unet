"""
fusion_core.py
==============
AGRI → GPM 空间匹配引擎（Tile 级输出）。

核心功能：
  1. GPM 双线性插值到 AGRI 经纬度网格
  2. 滑窗切 128×128 tile（stride=64）
  3. 区域过滤：tile 中心在 region 内
  4. 质量过滤：AGRI 有效像元 > 70%
  5. 夜间检测：可见光通道置零
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config as cfg
import fusion_config as fc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AGRI 有效圆盘掩膜
# ---------------------------------------------------------------------------

def compute_tight_disk_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    margin_deg: float = 5.0,
    sub_lon: float = 105.0,
) -> np.ndarray:
    """
    计算 AGRI 全圆盘缩紧后的有效像元 mask。
    以星下点 (0°N, sub_lon°E) 为圆心，剔除距圆盘边界 margin_deg 度以内的边缘像元。
    """
    valid = np.isfinite(lat) & np.isfinite(lon)
    if not valid.any():
        return valid

    sub_lat = 0.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    c_lat_r = np.deg2rad(sub_lat)
    c_lon_r = np.deg2rad(sub_lon)

    dlat = lat_r - c_lat_r
    dlon = lon_r - c_lon_r
    a = np.sin(dlat * 0.5) ** 2 + np.cos(c_lat_r) * np.cos(lat_r) * np.sin(dlon * 0.5) ** 2
    dist_deg = np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

    max_dist = float(dist_deg[valid].max())
    threshold = max(0.0, max_dist - margin_deg)

    return valid & (dist_deg <= threshold)


# ---------------------------------------------------------------------------
# 夜间判断
# ---------------------------------------------------------------------------

def is_nighttime_sza(sza: np.ndarray) -> bool:
    """判断整景是否为夜间（SZA 中位数 > 85°）。"""
    valid = np.isfinite(sza)
    if not valid.any():
        return False
    return bool(np.nanmedian(sza[valid]) > 85.0)


# ---------------------------------------------------------------------------
# 主匹配引擎
# ---------------------------------------------------------------------------

def extract_tiles_with_gpm(
    agri: dict,
    gpm_precip: np.ndarray,
    gpm_lat: np.ndarray,
    gpm_lon: np.ndarray,
    tile_size: tuple = (128, 128),
    stride: int = 64,
    region_lat_min: float = -10.0,
    region_lat_max: float = 20.0,
    region_lon_min: float = 100.0,
    region_lon_max: float = 130.0,
) -> List[dict]:
    """
    在 AGRI 全分辨率网格上滑窗切 tile，将 GPM 双线性插值到 AGRI 网格后裁切。

    Parameters
    ----------
    agri : dict with lat, lon, VZA, SZA, BT (all 2748×2748)
    gpm_precip : (N_lat, N_lon) GPM 降水率
    gpm_lat : (N_lat,) GPM 纬度轴
    gpm_lon : (N_lon,) GPM 经度轴

    Returns
    -------
    List of dict with:
        agri_tile   : (9, 128, 128) float32  — 7 BT + 2 geo (lat/90, lon/180)
        gpm_tile    : (1, 128, 128) float32  — 插值后降水量
        lat_center  : float                  — tile 中心纬度
        lon_center  : float                  — tile 中心经度
        has_rain    : bool                   — tile 内是否有降水 (>0.1mm/h)
    """
    th, tw = tile_size
    agri_lat = agri["lat"]
    agri_lon = agri["lon"]
    agri_bt = agri["BT"]
    agri_sza = agri["SZA"]
    H_a, W_a, C = agri_bt.shape

    # ── 判断是否为夜间 ──
    night = is_nighttime_sza(agri_sza)
    if night:
        log.debug("Nighttime scene detected, will zero visible channels")

    # ── GPM 插值器（双线性） ──
    precip_filled = np.where(np.isfinite(gpm_precip), gpm_precip, 0.0).astype(np.float32)
    gpm_interp = RegularGridInterpolator(
        (gpm_lat.astype(np.float64), gpm_lon.astype(np.float64)),
        precip_filled.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # ── 确定 AGRI 行列范围内的 region ──
    valid_agri = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if not valid_agri.any():
        return []

    # 在 AGRI 网格中找 region 对应的行列范围（加速）
    in_region = (
        valid_agri
        & (agri_lat >= region_lat_min) & (agri_lat <= region_lat_max)
        & (agri_lon >= region_lon_min) & (agri_lon <= region_lon_max)
    )
    region_rows, region_cols = np.where(in_region)
    if len(region_rows) == 0:
        log.debug("No AGRI pixels in region")
        return []

    r_min, r_max = region_rows.min(), region_rows.max()
    c_min, c_max = region_cols.min(), region_cols.max()

    # 扩展范围以包含跨越边界的 tile
    r_start = max(0, r_min - th)
    r_end = min(H_a, r_max + th)
    c_start = max(0, c_min - tw)
    c_end = min(W_a, c_max + tw)

    # ── 滑窗切 tile ──
    samples = []
    for r in range(r_start, r_end - th + 1, stride):
        for c in range(c_start, c_end - tw + 1, stride):
            r_end_t = r + th
            c_end_t = c + tw

            # Tile 中心坐标
            center_lat = float(np.nanmedian(agri_lat[r:r_end_t, c:c_end_t]))
            center_lon = float(np.nanmedian(agri_lon[r:r_end_t, c:c_end_t]))

            # 区域过滤：tile 中心在 region 内
            if not (region_lat_min <= center_lat <= region_lat_max
                    and region_lon_min <= center_lon <= region_lon_max):
                continue

            # 提取 AGRI BT
            patch_bt = agri_bt[r:r_end_t, c:c_end_t, :].copy()  # (128, 128, 7)
            patch_lat = agri_lat[r:r_end_t, c:c_end_t].copy()
            patch_lon = agri_lon[r:r_end_t, c:c_end_t].copy()

            # 有效性检查：AGRI 有效像元 > 70%
            valid_frac = np.isfinite(patch_bt).mean()
            if valid_frac < 0.7:
                continue

            # NaN → 0
            patch_bt = np.where(np.isfinite(patch_bt), patch_bt, 0.0).astype(np.float32)
            patch_lat = np.where(np.isfinite(patch_lat), patch_lat, 0.0).astype(np.float32)
            patch_lon = np.where(np.isfinite(patch_lon), patch_lon, 0.0).astype(np.float32)

            # 夜间：可见光通道置零
            if night:
                for vi in cfg.VIS_CHANNEL_INDICES:
                    patch_bt[:, :, vi] = 0.0

            # GPM 插值到 tile 网格
            interp_pts = np.stack([
                patch_lat.ravel().astype(np.float64),
                patch_lon.ravel().astype(np.float64),
            ], axis=-1)
            gpm_interp_flat = gpm_interp(interp_pts).astype(np.float32)
            gpm_tile = gpm_interp_flat.reshape(th, tw)  # (128, 128)

            has_rain = bool(np.any(gpm_tile > 0.1))

            # Transpose BT to (C, H, W)
            patch_bt = patch_bt.transpose(2, 0, 1)  # (7, 128, 128)

            # Geo: lat/90, lon/180
            geo = np.stack([
                patch_lat / 90.0,
                patch_lon / 180.0,
            ], axis=0).astype(np.float32)

            # Concat BT + geo → (9, 128, 128)
            agri_tile = np.concatenate([patch_bt, geo], axis=0)

            samples.append({
                "agri_tile": agri_tile,
                "gpm_tile": gpm_tile[np.newaxis, :, :],  # (1, 128, 128)
                "lat_center": center_lat,
                "lon_center": center_lon,
                "has_rain": has_rain,
            })

    return samples
