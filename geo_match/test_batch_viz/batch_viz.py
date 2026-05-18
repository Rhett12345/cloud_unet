"""
批量绘制 L2 vs MODIS / 模型外推 vs MODIS 差异可视化
====================================================
日期: 20190505
条件:
  1. 时间差 ≤ 5 min
  2. MODIS 范围完全落入 AGRI 全圆盘
  3. FY4A L2 CTH/CLP + MODIS MYD06/MYD03 + 模型检索结果三者同时存在

用法:
  conda run -n cloudunet python batch_viz.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import h5py
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────
L2_CTH_DIR = Path("/data/Data_yuq/FY4A_L2/CTH/20190505")
L2_CLP_DIR = Path("/data/Data_yuq/FY4A_L2/CLP/20190505")
MYD06_DIR  = Path("/data/Data_yuq/MYD06/20190505")
MYD03_DIR  = Path("/data/Data_yuq/MYD03/20190505")
RETRIEVAL_DIR = Path("/data/Data_yuq/unet_workdir/retrieval")
FDI_DIR = Path("/data/Data_yuq/FY4A/20190505")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

DAY = "20190505"
TIME_MAX_MIN = 5.0          # 最大时间差（分钟）
GRID_RES_DEG = 0.05         # ≈ 5 km，与 MODIS 5km 匹配
MIN_DISK_DIST_KM = 10.0     # MODIS 边缘到 AGRI 有效像元最远距离
MAP_RES = "50m"

# ─────────────────────────────────────────────────────────────────
# FY4A 坐标转换常量
# ─────────────────────────────────────────────────────────────────
_RES_PARAMS = {
    500:   dict(COFF=10991.5, CFAC=81865099, LOFF=10991.5, LFAC=81865099),
    1000:  dict(COFF=5495.5,  CFAC=40932549, LOFF=5495.5,  LFAC=40932549),
    2000:  dict(COFF=2747.5,  CFAC=20466274, LOFF=2747.5,  LFAC=20466274),
    4000:  dict(COFF=1373.5,  CFAC=10233137, LOFF=1373.5,  LFAC=10233137),
}
_EA    = 6378.137
_EB    = 6356.7523
_H     = 42164.0
_LAM_D = 104.7

# cartopy 投影（延迟导入避免无 cartopy 时崩溃）
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _PC = ccrs.PlateCarree()
    _HAS_CARTOPY = True
except ImportError:
    _PC = None
    _HAS_CARTOPY = False


# ═══════════════════════════════════════════════════════════════════
# 坐标转换
# ═══════════════════════════════════════════════════════════════════

def linecolumn_to_lonlat(l: np.ndarray, c: np.ndarray,
                         resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """FY4A 标称行列号 → 地理经纬度，地球外像素返回 NaN。"""
    p = _RES_PARAMS[resolution]
    COFF, CFAC, LOFF, LFAC = p['COFF'], p['CFAC'], p['LOFF'], p['LFAC']
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    x = np.deg2rad((c - COFF) / (2 ** -16 * CFAC))
    y = np.deg2rad((l - LOFF) / (2 ** -16 * LFAC))

    cos_x, cos_y = np.cos(x), np.cos(y)
    sin_x, sin_y = np.sin(x), np.sin(y)

    disc = ((_H * cos_x * cos_y)**2
            - (cos_y**2 + (_EA**2 / _EB**2) * sin_y**2) * (_H**2 - _EA**2))
    valid     = disc >= 0
    disc_safe = np.where(valid, disc, 0.0)

    sd  = np.sqrt(disc_safe)
    sn  = (_H * cos_x * cos_y - sd) / (cos_y**2 + (_EA**2 / _EB**2) * sin_y**2)
    s1  = _H - sn * cos_x * cos_y
    s2  = sn * sin_x * cos_y
    s3  = -sn * sin_y
    sxy = np.sqrt(s1**2 + s2**2)

    lon = np.rad2deg(np.arctan(s2 / s1)) + _LAM_D
    lat = np.rad2deg(np.arctan((_EA**2 / _EB**2) * (s3 / sxy)))
    lon = np.where(valid, lon, np.nan)
    lat = np.where(valid, lat, np.nan)
    return lon, lat


def _fy4a_full_lonlat(shape: Tuple[int, int],
                      resolution: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    l_arr, c_arr = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    return linecolumn_to_lonlat(l_arr, c_arr, resolution)


# ═══════════════════════════════════════════════════════════════════
# lat/lon → ECEF xyz
# ═══════════════════════════════════════════════════════════════════

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    cos_lat = np.cos(lat_r)
    return np.column_stack([
        cos_lat * np.cos(lon_r),
        cos_lat * np.sin(lon_r),
        np.sin(lat_r),
    ])


# ═══════════════════════════════════════════════════════════════════
# FY4A L2 NC 读取
# ═══════════════════════════════════════════════════════════════════

def read_fy4a_l2_nc(nc_path: str, var_name: str,
                    resolution: int = 4000) -> Dict[str, np.ndarray]:
    with nc.Dataset(nc_path, 'r') as ds:
        var = ds.variables[var_name]
        var.set_auto_mask(False)
        raw  = var[:]
        fill = getattr(var, '_FillValue', None) or getattr(var, 'FillValue', None)
        attrs = {k: getattr(var, k) for k in var.ncattrs()}

    data = raw.T.copy().astype(np.float32)

    if fill is not None:
        fill_val = np.array(fill).astype(raw.dtype)
        data[data == fill_val] = np.nan

    print(f"  [read] {Path(nc_path).name}  shape={data.shape}")
    lon, lat = _fy4a_full_lonlat(data.shape, resolution)
    return {'data': data, 'lon': lon, 'lat': lat, 'attrs': attrs}


# ═══════════════════════════════════════════════════════════════════
# MODIS HDF4 读取
# ═══════════════════════════════════════════════════════════════════

def _read_hdf4_sds(hdf_path: str, sds_name: str) -> Tuple[np.ndarray, dict]:
    hdf  = SD(hdf_path, SDC.READ)
    sds  = hdf.select(sds_name)
    data = sds.get().astype(np.float32)
    attr = sds.attributes()
    hdf.end()

    fill   = attr.get('_FillValue', None)
    scale  = attr.get('scale_factor', 1.0)
    offset = attr.get('add_offset',  0.0)

    if fill is not None:
        mask = data == float(fill)
    else:
        mask = np.zeros_like(data, dtype=bool)

    data = data * float(scale) + float(offset)
    data[mask] = np.nan
    return data, attr


def read_modis_myd06(myd06_path: str, myd03_path: str) -> Dict[str, object]:
    print(f"  [read] MYD06: {Path(myd06_path).name}")
    clp, clp_attrs = _read_hdf4_sds(myd06_path, 'Cloud_Phase_Infrared')
    cth, cth_attrs = _read_hdf4_sds(myd06_path, 'Cloud_Top_Height')
    ctp, ctp_attrs = _read_hdf4_sds(myd06_path, 'Cloud_Top_Pressure')

    print(f"  [read] MYD03: {Path(myd03_path).name}")
    hdf03 = SD(myd03_path, SDC.READ)
    lat1km = hdf03.select('Latitude').get().astype(np.float32)
    lon1km = hdf03.select('Longitude').get().astype(np.float32)
    hdf03.end()

    row_idx = np.arange(2, 2030, 5)[:406]
    col_idx = np.arange(2, 1354, 5)[:270]

    lat5km = lat1km[np.ix_(row_idx, col_idx)]
    lon5km = lon1km[np.ix_(row_idx, col_idx)]

    lat5km[lat5km < -90]  = np.nan
    lon5km[lon5km < -180] = np.nan

    return {
        'clp': clp, 'cth': cth, 'ctp': ctp,
        'lon': lon5km, 'lat': lat5km,
        'clp_attrs': clp_attrs, 'cth_attrs': cth_attrs,
    }


# ═══════════════════════════════════════════════════════════════════
# 模型外推结果读取
# ═══════════════════════════════════════════════════════════════════

def read_model_retrieval(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    cth_pred = d['CTH_pred'].astype(np.float32)
    clp_pred = d['CLP_pred'].astype(np.float32)
    lat = d['latitude'].astype(np.float32)
    lon = d['longitude'].astype(np.float32)

    # 模型 CLP: -1=invalid, 0=Clear, 1=Water, 2=Ice
    clp_pred[clp_pred < 0] = np.nan
    # 模型 CTH: 0 = invalid
    cth_pred[cth_pred <= 0] = np.nan

    print(f"  [read] Model: {Path(npz_path).name}  shape={cth_pred.shape}")
    return {
        'data_cth': cth_pred,
        'data_clp': clp_pred,
        'lon': lon,
        'lat': lat,
    }


# ═══════════════════════════════════════════════════════════════════
# FDI L1 RGB 读取
# ═══════════════════════════════════════════════════════════════════

def read_fdi_rgb(fdi_path: str,
                 r_ch: int = 2,
                 g_ch: int = 3,
                 b_ch: int = 1,
                 gamma: float = 1.5,
                 percentile: Tuple[float, float] = (2, 98)
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 FDI L1 HDF5 文件的三通道，做定标 + 百分位拉伸 + gamma 校正，
    返回 (R, G, B) 三通道 float32 数组，shape 均为 (2748, 2748)，值域 [0, 1]。

    Parameters
    ----------
    fdi_path   : FDI HDF5 文件路径
    r_ch/g_ch/b_ch : R/G/B 通道号 (默认 2/3/1 = 近似真彩色)
    gamma      : gamma 校正系数 (>1 增亮暗部)
    percentile : 拉伸百分位 (lo, hi)
    """
    def _load_channel(ch: int) -> np.ndarray:
        ch_name = f"NOMChannel{ch:02d}"
        cal_name = f"CALChannel{ch:02d}"
        with h5py.File(fdi_path, 'r') as f:
            raw = f[ch_name][:]
            fill = int(f[ch_name].attrs.get('FillValue', [65535])[0])
            if cal_name in f:
                cal_table = f[cal_name][:]
                idx = np.clip(raw.astype(np.int32), 0, len(cal_table) - 1)
                data = cal_table[idx].astype(np.float32)
            else:
                data = raw.astype(np.float32)
        data[raw == fill] = np.nan
        return data

    print(f"  [read] FDI RGB: {Path(fdi_path).name}")
    data_r = _load_channel(r_ch)
    data_g = _load_channel(g_ch)
    data_b = _load_channel(b_ch)

    # 逐通道百分位拉伸 + gamma
    for data in [data_r, data_g, data_b]:
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            continue
        lo, hi = np.nanpercentile(valid, percentile)
        data[:] = np.clip((data - lo) / (hi - lo + 1e-9), 0, 1) ** (1.0 / gamma)

    return data_r, data_g, data_b


# ═══════════════════════════════════════════════════════════════════
# AGRI 像素 → 规则经纬度网格重采样（参照 agri_viz.py pipeline）
# ═══════════════════════════════════════════════════════════════════

def _agri_pixels_to_scatter(data: np.ndarray,
                            subsample: int = 4
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AGRI 2748×2748 像素 → (lon, lat, val) 散点，NaN 已剔除。"""
    sl = slice(None, None, subsample)
    rows = np.arange(data.shape[0])[sl]
    cols = np.arange(data.shape[1])[sl]
    c2d, r2d = np.meshgrid(cols, rows)
    lon, lat = linecolumn_to_lonlat(r2d, c2d, resolution=4000)
    d_sub = data[sl, :][:, sl]
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(d_sub)
    return lon[valid], lat[valid], d_sub[valid]


def _scatter_to_grid(lon_pts: np.ndarray, lat_pts: np.ndarray,
                     val_pts: np.ndarray,
                     lon_range: Tuple[float, float],
                     lat_range: Tuple[float, float],
                     nx: int = 600, ny: int = 600,
                     method: str = 'linear'
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """散点 (lon, lat, val) → griddata 插值到规则网格。"""
    lon_grid = np.linspace(lon_range[0], lon_range[1], nx)
    lat_grid = np.linspace(lat_range[0], lat_range[1], ny)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    val2d = griddata(
        np.column_stack([lon_pts, lat_pts]), val_pts,
        (lon2d, lat2d), method=method,
    )
    return lon2d, lat2d, val2d


def _agri_rgb_resample(
    r: np.ndarray, g: np.ndarray, b: np.ndarray,
    subsample: int = 4, nx: int = 600, ny: int = 600,
    extent: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    AGRI 三通道 2748×2748 数据 → 规则 PlateCarree 网格 RGB。

    Returns
    -------
    lon2d, lat2d : 规则网格坐标
    R, G, B      : 规则网格上的三通道值，shape=(ny, nx)，[0,1]
    extent       : 实际使用的 [lon_min, lon_max, lat_min, lat_max]
    """
    print(f"    重采样 AGRI → 规则网格 (subsample={subsample}, {nx}×{ny})...")
    lon_r, lat_r, val_r = _agri_pixels_to_scatter(r, subsample)
    lon_g, lat_g, val_g = _agri_pixels_to_scatter(g, subsample)
    lon_b, lat_b, val_b = _agri_pixels_to_scatter(b, subsample)

    # ★ 从 linecolumn_to_lonlat 散点计算 Earth disk 范围（不是模型预计算的全球范围）
    if extent is None:
        extent = [float(np.nanmin(lon_r)), float(np.nanmax(lon_r)),
                  float(np.nanmin(lat_r)), float(np.nanmax(lat_r))]
    print(f"    Earth disk extent: lon=[{extent[0]:.2f}, {extent[1]:.2f}]  "
          f"lat=[{extent[2]:.2f}, {extent[3]:.2f}]")
    lon_range = (extent[0], extent[1])
    lat_range = (extent[2], extent[3])

    # 裁剪到 extent
    def _clip(lon, lat, val):
        mask = ((lon >= lon_range[0]) & (lon <= lon_range[1]) &
                (lat >= lat_range[0]) & (lat <= lat_range[1]))
        return lon[mask], lat[mask], val[mask]

    lon_r, lat_r, val_r = _clip(lon_r, lat_r, val_r)
    lon_g, lat_g, val_g = _clip(lon_g, lat_g, val_g)
    lon_b, lat_b, val_b = _clip(lon_b, lat_b, val_b)

    _, _, R = _scatter_to_grid(lon_r, lat_r, val_r, lon_range, lat_range, nx, ny)
    _, _, G = _scatter_to_grid(lon_g, lat_g, val_g, lon_range, lat_range, nx, ny)
    lon2d, lat2d, B = _scatter_to_grid(lon_b, lat_b, val_b, lon_range, lat_range, nx, ny)

    R = np.nan_to_num(R, nan=0.0).clip(0, 1)
    G = np.nan_to_num(G, nan=0.0).clip(0, 1)
    B = np.nan_to_num(B, nan=0.0).clip(0, 1)

    return lon2d, lat2d, R, G, B, list(extent)


# ═══════════════════════════════════════════════════════════════════
# MODIS 落入 AGRI 圆盘检测
# ═══════════════════════════════════════════════════════════════════

def check_modis_in_agri_disk(
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    max_dist_km: float = 10.0,
) -> bool:
    agri_valid = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if not agri_valid.any():
        return False

    agri_xyz = latlon_to_xyz(agri_lat[agri_valid], agri_lon[agri_valid])
    tree = cKDTree(agri_xyz)

    h, w = modis_lat.shape
    step = max(1, min(h, w) // 50)

    top_cols = np.arange(0, w, step, dtype=int)
    top_rows = np.zeros(len(top_cols), dtype=int)
    bot_cols = np.arange(0, w, step, dtype=int)
    bot_rows = np.full(len(bot_cols), h - 1, dtype=int)
    left_rows = np.arange(0, h, step, dtype=int)
    left_cols = np.zeros(len(left_rows), dtype=int)
    right_rows = np.arange(0, h, step, dtype=int)
    right_cols = np.full(len(right_rows), w - 1, dtype=int)

    edge_rows = np.concatenate([top_rows, bot_rows, left_rows, right_rows])
    edge_cols = np.concatenate([top_cols, bot_cols, left_cols, right_cols])

    sample_lat = modis_lat[edge_rows, edge_cols]
    sample_lon = modis_lon[edge_rows, edge_cols]
    valid_sample = np.isfinite(sample_lat) & np.isfinite(sample_lon)
    if not valid_sample.any():
        return False

    sample_xyz = latlon_to_xyz(sample_lat[valid_sample], sample_lon[valid_sample])
    dist, _ = tree.query(sample_xyz, k=1)
    dist_km = 2.0 * 6371.0 * np.arcsin(np.clip(dist * 0.5, 0.0, 1.0))

    return bool(np.all(dist_km <= max_dist_km))


# ═══════════════════════════════════════════════════════════════════
# 时间解析
# ═══════════════════════════════════════════════════════════════════

def extract_fy4a_time(filename: str) -> datetime:
    """从 FY4A L2 文件名中提取开始时间。"""
    m = re.search(r'(\d{14})', Path(filename).name)
    if m:
        return datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
    raise ValueError(f"Cannot parse time from {filename}")

def extract_modis_time(filename: str) -> datetime:
    """从 MODIS 文件名中提取时间 (HHMM UTC)。"""
    name = Path(filename).name
    m = re.search(r'A\d{7}\.(\d{4})\.', name)
    if m:
        hhmm = m.group(1)
        return datetime.strptime(f"20190505{hhmm}00", '%Y%m%d%H%M%S')
    raise ValueError(f"Cannot parse time from {filename}")

def extract_retrieval_time(filename: str) -> datetime:
    """从模型检索文件名中提取开始时间。"""
    m = re.search(r'(\d{14})', Path(filename).name)
    if m:
        return datetime.strptime(m.group(1), '%Y%m%d%H%M%S')
    raise ValueError(f"Cannot parse time from {filename}")


# ═══════════════════════════════════════════════════════════════════
# 地理匹配
# ═══════════════════════════════════════════════════════════════════

def geo_match(
    src_lon, src_lat, src_val,
    ref_lon, ref_lat, ref_val,
    grid_res_deg=0.05,
    interp_method='nearest',
    max_extrap_km=15.0,
):
    slon, slat, sval = src_lon.ravel(), src_lat.ravel(), src_val.ravel()
    rlon, rlat, rval = ref_lon.ravel(), ref_lat.ravel(), ref_val.ravel()

    def _valid(lon, lat, val):
        m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
        return lon[m], lat[m], val[m]

    slon, slat, sval = _valid(slon, slat, sval)
    rlon, rlat, rval = _valid(rlon, rlat, rval)

    if slon.size == 0 or rlon.size == 0:
        raise ValueError("有效数据点为空")

    lon_min = max(np.nanmin(slon), np.nanmin(rlon))
    lon_max = min(np.nanmax(slon), np.nanmax(rlon))
    lat_min = max(np.nanmin(slat), np.nanmin(rlat))
    lat_max = min(np.nanmax(slat), np.nanmax(rlat))

    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("两组数据无地理重叠区域")

    extent = [lon_min, lon_max, lat_min, lat_max]
    lon_grid = np.arange(lon_min, lon_max, grid_res_deg)
    lat_grid = np.arange(lat_min, lat_max, grid_res_deg)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    # ── 距离掩膜：格点距最近真实数据点超限则视为无数据 ──────────
    def _to_xyz(lat, lon):
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        return np.column_stack([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ])

    def _coverage_mask(data_lon, data_lat, max_km):
        tree = cKDTree(_to_xyz(data_lat, data_lon))
        chord, _ = tree.query(_to_xyz(lat2d.ravel(), lon2d.ravel()), k=1)
        km = 2.0 * 6371.0 * np.arcsin(np.clip(chord / 2.0, 0, 1))
        return (km <= max_km).reshape(lon2d.shape)

    src_coverage = _coverage_mask(slon, slat, max_extrap_km)
    ref_coverage = _coverage_mask(rlon, rlat, max_extrap_km)

    # ── 两者都有覆盖才是有效格点，先算出 overlap，插值只在这里做 ──
    overlap = src_coverage & ref_coverage          # ← 核心改动

    src_grid = np.full(lon2d.shape, np.nan, dtype=np.float32)
    ref_grid = np.full(lon2d.shape, np.nan, dtype=np.float32)

    if overlap.any():
        # 只对 overlap 区域内的格点做插值，彻底避免边缘外推
        overlap_pts = np.column_stack([lon2d[overlap], lat2d[overlap]])

        src_grid[overlap] = griddata(
            np.column_stack([slon, slat]), sval,
            overlap_pts, method=interp_method,
        )
        ref_grid[overlap] = griddata(
            np.column_stack([rlon, rlat]), rval,
            overlap_pts, method=interp_method,
        )

        # griddata 在凸包外仍可能返回 nan，再过滤一次
        still_nan = ~np.isfinite(src_grid) | ~np.isfinite(ref_grid)
        overlap[still_nan] = False
        src_grid[still_nan] = np.nan
        ref_grid[still_nan] = np.nan

    diff = np.where(overlap, src_grid - ref_grid, np.nan)

    print(f"    公共网格: {lon2d.shape}  分辨率={grid_res_deg}°")
    print(f"    重叠像素: {overlap.sum()}")

    return {
        'lon2d': lon2d, 'lat2d': lat2d,
        'src_grid': src_grid, 'ref_grid': ref_grid,
        'diff_grid': diff, 'overlap_mask': overlap, 'extent': extent,
    }


# ═══════════════════════════════════════════════════════════════════
# 绘图辅助
# ═══════════════════════════════════════════════════════════════════

def _make_ax(fig, pos, extent, land_color='lightgray', ocean_color='lightblue',
             map_res='50m', gridlines=True) -> plt.Axes:
    if not _HAS_CARTOPY:
        raise ImportError("cartopy 未安装，无法绘制地图")
    ax = fig.add_subplot(pos, projection=_PC)
    ax.set_extent(extent, crs=_PC)
    ax.add_feature(cfeature.OCEAN.with_scale(map_res),   facecolor=ocean_color, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(map_res),    facecolor=land_color,  zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale(map_res),
                   linewidth=0.7, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale(map_res),
                   linewidth=0.5, edgecolor='gray', linestyle='--', zorder=3)
    if gridlines:
        gl = ax.gridlines(crs=_PC, draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.6,
                          linestyle='--', zorder=4)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
    return ax


# ═══════════════════════════════════════════════════════════════════
# CLP 简化映射
# ═══════════════════════════════════════════════════════════════════

def _fy4a_clp_to_simple(v: np.ndarray) -> np.ndarray:
    """FY4A L2 CLP → 3 类: 0=Clear, 1=Water, 2=Ice/Mixed"""
    out = np.full_like(v, np.nan)
    out[v == 0] = 0
    out[v == 1] = 1   # Water
    out[v == 2] = 1   # SuperCooled → Water
    out[v == 3] = 2   # Mixed
    out[v == 4] = 2   # Ice
    return out

def _modis_clp_to_simple(v: np.ndarray) -> np.ndarray:
    """MODIS IR CLP → 3 类: 0=Clear, 1=Water, 2=Ice/Mixed"""
    out = np.full_like(v, np.nan)
    out[v == 0] = 0   # CloudFree
    out[v == 1] = 1   # Water
    out[v == 2] = 2   # Ice
    out[v == 3] = 2   # Mixed → Ice
    return out

def _model_clp_to_simple(v: np.ndarray) -> np.ndarray:
    """模型 CLP → 3 类: 0=Clear, 1=Water, 2=Ice"""
    out = np.full_like(v, np.nan)
    out[v == 0] = 0
    out[v == 1] = 1
    out[v == 2] = 2
    return out


# ═══════════════════════════════════════════════════════════════════
# 对比图
# ═══════════════════════════════════════════════════════════════════

def plot_cth_comparison(match: Dict, src_label: str, ref_label: str,
                        time_diff_min: float, scene_id: str,
                        save_dir: Path, prefix: str):
    """
    CTH 三联图: src | MODIS | 差值
    """
    lon2d, lat2d = match['lon2d'], match['lat2d']
    src_g = match['src_grid']
    ref_g = match['ref_grid']
    diff  = match['diff_grid']
    extent = match['extent']
    overlap = match['overlap_mask']

    vmin = np.nanpercentile(np.concatenate([src_g[overlap], ref_g[overlap]]), 2)
    vmax = np.nanpercentile(np.concatenate([src_g[overlap], ref_g[overlap]]), 98)
    diff_abs = np.nanpercentile(np.abs(diff[overlap]), 95) if overlap.any() else 100
    diff_abs = max(diff_abs, 100)

    # 差值统计
    if overlap.any():
        d_valid = diff[overlap]
        stats = (f"mean={np.nanmean(d_valid):.0f}m  "
                 f"std={np.nanstd(d_valid):.0f}m  "
                 f"RMSE={np.sqrt(np.nanmean(d_valid**2)):.0f}m")
    else:
        stats = "no overlap"

    fig = plt.figure(figsize=(18, 6), dpi=110)
    kw = dict(extent=extent, map_res=MAP_RES)

    ax1 = _make_ax(fig, 131, **kw)
    src_plot = np.where(overlap, src_g, np.nan)
    pcm1 = ax1.pcolormesh(lon2d, lat2d, src_plot,
                           cmap='plasma', vmin=vmin, vmax=vmax,
                           transform=_PC, shading='auto', zorder=1, alpha=0.92)
    fig.colorbar(pcm1, ax=ax1, fraction=0.046, pad=0.05, label='m')
    ax1.set_title(f"{src_label}\nΔt={time_diff_min:.1f}min", fontsize=11, fontweight='bold')

    ax2 = _make_ax(fig, 132, **kw)
    ref_plot = np.where(overlap, ref_g, np.nan)
    pcm2 = ax2.pcolormesh(lon2d, lat2d, ref_plot,
                           cmap='plasma', vmin=vmin, vmax=vmax,
                           transform=_PC, shading='auto', zorder=1, alpha=0.92)
    fig.colorbar(pcm2, ax=ax2, fraction=0.046, pad=0.05, label='m')
    ax2.set_title(f"{ref_label}", fontsize=11, fontweight='bold')

    ax3 = _make_ax(fig, 133, **kw)
    pcm3 = ax3.pcolormesh(lon2d, lat2d, diff,
                           cmap='RdBu_r', vmin=-diff_abs, vmax=diff_abs,
                           transform=_PC, shading='auto', zorder=1, alpha=0.92)
    fig.colorbar(pcm3, ax=ax3, fraction=0.046, pad=0.05, label='m')
    ax3.set_title(f"Diff ({src_label.split()[0]} − MODIS)\n{stats}",
                  fontsize=11, fontweight='bold')

    suptitle = f"CTH — {scene_id}  (Δt={time_diff_min:.1f} min)"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in [
                'png',
                # 'svg',
                # 'pdf'
                ]:
        save_path = save_dir / f"{prefix}_cth.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] {prefix}_cth.{{png,svg,pdf}}")
    plt.close(fig)


def plot_clp_comparison(match: Dict, src_label: str, ref_label: str,
                        clp_from: str,  # 'l2' or 'model'
                        time_diff_min: float, scene_id: str,
                        save_dir: Path, prefix: str):
    """
    CLP 三联图: src | MODIS | 一致性
    """
    lon2d, lat2d = match['lon2d'], match['lat2d']
    src_g = match['src_grid']
    ref_g = match['ref_grid']
    overlap = match['overlap_mask']
    extent = match['extent']

    # 简化到 3 类
    if clp_from == 'l2':
        src_simple = _fy4a_clp_to_simple(np.round(src_g))
    else:
        src_simple = _model_clp_to_simple(np.round(src_g))

    ref_simple = _modis_clp_to_simple(np.round(ref_g))

    # 一致性: 0=Clear一致, 1=Water一致, 2=Ice一致, 3=不一致
    agree = np.where(
        overlap & (src_simple == ref_simple),
        src_simple,
        np.where(overlap, 3, np.nan),
    )

    phase_colors = {0: '#A8D5BA', 1: '#5B9BD5', 2: '#C4A0E8', 3: '#E8736A'}
    phase_labels = {0: 'Clear', 1: 'Water', 2: 'Ice/Mixed', 3: 'Disagree'}
    cmap_phase = mcolors.ListedColormap(
        ['#A8D5BA', '#5B9BD5', '#C4A0E8', '#E8736A'])
    norm_phase = mcolors.BoundaryNorm([0, 1, 2, 3, 4], 4)

    src_plot = np.where(overlap, src_simple, np.nan)
    ref_plot = np.where(overlap, ref_simple, np.nan)

    fig = plt.figure(figsize=(18, 6), dpi=110)
    kw = dict(extent=extent, map_res=MAP_RES)

    for idx, (data, title) in enumerate(
            [(src_plot, f"{src_label}\nΔt={time_diff_min:.1f}min"),
             (ref_plot, ref_label),
             (agree, "Agreement")]):
        ax = _make_ax(fig, int(f"13{idx+1}"), **kw)
        ax.pcolormesh(lon2d, lat2d, data,
                      cmap=cmap_phase, norm=norm_phase,
                      transform=_PC, shading='auto', zorder=1, alpha=0.92)
        if idx == 2 and overlap.any():
            total  = overlap.sum()
            agree_n = int(np.sum((agree >= 0) & (agree <= 2) & np.isfinite(agree)))
            pct = 100 * agree_n / total if total > 0 else 0
            title = f"Agreement (rate={pct:.1f}%)"
        ax.set_title(title, fontsize=11, fontweight='bold')

    patches = [mpatches.Patch(color=phase_colors[k], label=phase_labels[k])
               for k in [0, 1, 2, 3]]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.06))

    suptitle = f"CLP — {scene_id}  (Δt={time_diff_min:.1f} min)"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in ['png',
                # 'svg',
                # 'pdf'
                ]:
        save_path = save_dir / f"{prefix}_clp.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] {prefix}_clp.{{png,svg,pdf}}")
    plt.close(fig)


def plot_cth_scatter(match: Dict, src_label: str,
                     time_diff_min: float, scene_id: str,
                     save_dir: Path, prefix: str):
    """CTH 散点密度图"""
    overlap = match['overlap_mask']
    src_v = match['src_grid'][overlap]
    ref_v = match['ref_grid'][overlap]
    valid  = np.isfinite(src_v) & np.isfinite(ref_v)
    src_v, ref_v = src_v[valid], ref_v[valid]

    if src_v.size == 0:
        print("  [warn] 无有效配对点，跳过散点图")
        return

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)

    vmin = min(np.percentile(src_v, 1), np.percentile(ref_v, 1))
    vmax = max(np.percentile(src_v, 99), np.percentile(ref_v, 99))

    h, xe, ye = np.histogram2d(ref_v, src_v, bins=80,
                                range=[[vmin, vmax], [vmin, vmax]])
    ax.pcolormesh(xe, ye, h.T, cmap='hot_r', shading='auto')

    ax.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='1:1')

    coeffs = np.polyfit(ref_v, src_v, 1)
    x_fit  = np.linspace(vmin, vmax, 200)
    ax.plot(x_fit, np.polyval(coeffs, x_fit),
            'r-', linewidth=1.5,
            label=f'fit: y={coeffs[0]:.2f}x+{coeffs[1]:.0f}m')

    corr = float(np.corrcoef(ref_v, src_v)[0, 1])
    rmse = float(np.sqrt(np.mean((src_v - ref_v)**2)))
    bias = float(np.mean(src_v - ref_v))
    ax.set_xlabel("MODIS MYD06 CTH [m]", fontsize=11)
    ax.set_ylabel(f"{src_label} CTH [m]", fontsize=11)
    ax.set_title(f"CTH Scatter — {scene_id}  N={src_v.size}\n"
                 f"R={corr:.3f}  RMSE={rmse:.0f}m  Bias={bias:.0f}m  "
                 f"Δt={time_diff_min:.1f}min",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for fmt in ['png',
                # 'svg',
                # 'pdf'
                ]:
        save_path = save_dir / f"{prefix}_cth_scatter.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] {prefix}_cth_scatter.{{png,svg,pdf}}")
    plt.close(fig)


def plot_cth_histogram(match: Dict, src_label: str,
                       time_diff_min: float, scene_id: str,
                       save_dir: Path, prefix: str):
    """CTH 差值直方图 + CDF"""
    overlap = match['overlap_mask']
    diff    = match['diff_grid']
    d_valid = diff[overlap]
    d_valid = d_valid[np.isfinite(d_valid)]

    if d_valid.size == 0:
        print("  [warn] 无有效差值，跳过直方图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)

    bins = np.linspace(np.percentile(d_valid, 1), np.percentile(d_valid, 99), 60)
    ax1.hist(d_valid, bins=bins, color='steelblue', edgecolor='white',
             linewidth=0.3, alpha=0.85)
    ax1.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero')
    ax1.axvline(np.nanmean(d_valid), color='orange', linewidth=1.5,
                linestyle='-', label=f"Mean={np.nanmean(d_valid):.0f}m")
    ax1.set_xlabel(f"CTH diff ({src_label.split()[0]} − MODIS) [m]", fontsize=11)
    ax1.set_ylabel("Pixels", fontsize=11)
    ax1.set_title(f"CTH Diff Histogram — {scene_id}", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    stats_text = (f"N = {d_valid.size}\n"
                  f"Mean = {np.nanmean(d_valid):.0f} m\n"
                  f"Std  = {np.nanstd(d_valid):.0f} m\n"
                  f"RMSE = {np.sqrt(np.nanmean(d_valid**2)):.0f} m\n"
                  f"Bias = {np.nanmean(d_valid):.0f} m\n"
                  f"Δt   = {time_diff_min:.1f} min")
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
             va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    sorted_d = np.sort(d_valid)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax2.plot(sorted_d, cdf * 100, color='steelblue', linewidth=1.5)
    ax2.axvline(0, color='red', linewidth=1.5, linestyle='--')
    for q in [10, 25, 50, 75, 90]:
        qv = np.percentile(d_valid, q)
        ax2.axvline(qv, color='gray', linewidth=0.8, linestyle=':')
        ax2.text(qv, q + 1, f'P{q}={qv:.0f}m', fontsize=7, ha='center')
    ax2.set_xlabel(f"CTH diff ({src_label.split()[0]} − MODIS) [m]", fontsize=11)
    ax2.set_ylabel("Cumulative [%]", fontsize=11)
    ax2.set_title(f"CTH Diff CDF — {scene_id}", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()

    for fmt in ['png',
                # 'svg',
                # 'pdf'
                ]:
        save_path = save_dir / f"{prefix}_cth_hist.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] {prefix}_cth_hist.{{png,svg,pdf}}")
    plt.close(fig)


def plot_overlap_check(fy4a_lon, fy4a_lat, fy4a_mask,
                       model_lon, model_lat, model_mask,
                       modis_lon, modis_lat, modis_mask,
                       time_diff_min, scene_id, save_dir):
    """空间覆盖总览: FY4A L2 / 模型 / MODIS 三者散点"""
    if not _HAS_CARTOPY:
        return

    all_lons = []
    all_lats = []
    for lon, lat, mask, sl in [
        (fy4a_lon, fy4a_lat, fy4a_mask, 20),
        (model_lon, model_lat, model_mask, 20),
        (modis_lon, modis_lat, modis_mask, 1),
    ]:
        m = mask.ravel() & np.isfinite(lon.ravel()) & np.isfinite(lat.ravel())
        all_lons.append(lon.ravel()[m][::sl])
        all_lats.append(lat.ravel()[m][::sl])

    if not all_lons[0].size:
        return

    combo_lon = np.concatenate([a for a in all_lons if len(a) > 0])
    combo_lat = np.concatenate([a for a in all_lats if len(a) > 0])
    pad = 1.5
    extent = [float(np.nanmin(combo_lon)) - pad, float(np.nanmax(combo_lon)) + pad,
              float(np.nanmin(combo_lat)) - pad, float(np.nanmax(combo_lat)) + pad]

    fig = plt.figure(figsize=(12, 7), dpi=110)
    ax = _make_ax(fig, 111, extent, map_res=MAP_RES)

    colors = ['steelblue', 'green', 'tomato']
    labels = ['FY4A L2', 'Model', 'MODIS MYD06']
    for i, (lons, lats, c, lbl) in enumerate(zip(all_lons, all_lats, colors, labels)):
        if len(lons) > 0:
            ax.scatter(lons, lats, s=0.5 if i < 2 else 8, c=c, alpha=0.5,
                       transform=_PC, zorder=2, label=lbl)

    ax.legend(loc='lower right', fontsize=10, markerscale=5)
    ax.set_title(f"Coverage — {scene_id}  (Δt={time_diff_min:.1f} min)\n"
                 "(blue=FY4A L2  green=Model  red=MODIS)",
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    for fmt in ['png',
                # 'svg',
                # 'pdf'
                ]:
        save_path = save_dir / f"overlap_coverage.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] overlap_coverage.{{png,svg,pdf}}")
    plt.close(fig)


def plot_rgb_with_modis(rgb_r: np.ndarray, rgb_g: np.ndarray, rgb_b: np.ndarray,
                        agri_lon: np.ndarray, agri_lat: np.ndarray,
                        modis_lon: np.ndarray, modis_lat: np.ndarray,
                        time_diff_min: float, scene_id: str,
                        save_dir: Path):
    """
    AGRI FDI RGB 真彩色图 + MODIS 条带范围叠加。

    先将 AGRI 像素重采样到规则 PlateCarree 网格（参照 agri_viz.py pipeline），
    确保 RGB 与 cartopy 底图陆地对齐。再用红色半透明标示 MODIS 条带范围。
    """
    # ★ 关键：重采样到规则经纬度网格（全盘，降采样加速）
    _, _, R, G, B, extent = _agri_rgb_resample(
        rgb_r, rgb_g, rgb_b, subsample=4, nx=700, ny=700)
    rgb = np.stack([R, G, B], axis=-1)

    fig = plt.figure(figsize=(14, 10), dpi=110)
    ax = _make_ax(fig, 111, extent, map_res=MAP_RES,
                  land_color='none', ocean_color='none')

    # 重采样后的 RGB 在规则网格上，imshow 可与海岸线对齐
    ax.imshow(rgb,
              origin='lower',
              extent=extent,
              transform=_PC,
              interpolation='bilinear',
              zorder=1, alpha=0.95)

    # MODIS 条带凸包
    m_valid = np.isfinite(modis_lon) & np.isfinite(modis_lat)
    if m_valid.any():
        from scipy.spatial import ConvexHull
        pts = np.column_stack([modis_lon[m_valid].ravel()[::5],
                               modis_lat[m_valid].ravel()[::5]])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) > 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color='red', linewidth=1.5, linestyle='-',
                        transform=_PC, zorder=10)
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        color='red', alpha=0.12, transform=_PC, zorder=9)
            except Exception:
                pass

    # 海岸线 / 国界在 MODIS 层之上
    ax.add_feature(cfeature.COASTLINE.with_scale(MAP_RES),
                   linewidth=0.7, edgecolor='black', zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale(MAP_RES),
                   linewidth=0.5, edgecolor='gray', linestyle='--', zorder=11)

    ax.set_title(f"AGRI RGB + MODIS — {scene_id}  (Δt={time_diff_min:.1f} min)\n"
                 "Red outline/fill = MODIS MYD06 swath",
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    for fmt in ['png']:
        save_path = save_dir / f"rgb_modis_overlap.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] rgb_modis_overlap.{{png}}")
    plt.close(fig)


def plot_rgb_clp_overlap(rgb_r: np.ndarray, rgb_g: np.ndarray, rgb_b: np.ndarray,
                         agri_lon: np.ndarray, agri_lat: np.ndarray,
                         modis_lon: np.ndarray, modis_lat: np.ndarray,
                         modis_clp: np.ndarray,
                         time_diff_min: float, scene_id: str,
                         save_dir: Path):
    """
    重合区域 RGB + MODIS CLP 并排对比。

    左侧：AGRI RGB 重采样到规则网格后裁剪到 MODIS 条带范围
    右侧：MODIS CLP 相态分类（同区域）
    """
    # MODIS 条带有效范围
    m_valid = np.isfinite(modis_lon) & np.isfinite(modis_lat)
    if not m_valid.any():
        print("  [warn] MODIS 无有效坐标，跳过 RGB+CLP 重合图")
        return

    pad = 0.5
    m_lon_min = float(np.nanmin(modis_lon[m_valid])) - pad
    m_lon_max = float(np.nanmax(modis_lon[m_valid])) + pad
    m_lat_min = float(np.nanmin(modis_lat[m_valid])) - pad
    m_lat_max = float(np.nanmax(modis_lat[m_valid])) + pad
    overlap_extent = [m_lon_min, m_lon_max, m_lat_min, m_lat_max]

    # ★ 重采样：指定 extent 为 MODIS 范围，减少无效区域插值
    nx = max(400, int((m_lon_max - m_lon_min) / 0.04))
    ny = max(300, int((m_lat_max - m_lat_min) / 0.04))
    _, _, R, G, B, _ = _agri_rgb_resample(
        rgb_r, rgb_g, rgb_b,
        subsample=2, nx=nx, ny=ny, extent=overlap_extent)
    rgb = np.stack([R, G, B], axis=-1)

    # ── MODIS CLP → 简化 3 类 ──
    clp_simple = np.full_like(modis_clp, np.nan, dtype=np.float32)
    clp_simple[modis_clp == 0] = 0
    clp_simple[modis_clp == 1] = 1
    clp_simple[modis_clp == 2] = 2
    clp_simple[modis_clp == 3] = 2

    phase_colors = {0: '#A8D5BA', 1: '#5B9BD5', 2: '#C4A0E8'}
    phase_labels = {0: 'Clear', 1: 'Water', 2: 'Ice/Mixed'}
    cmap_phase = mcolors.ListedColormap(
        ['#A8D5BA', '#5B9BD5', '#C4A0E8'])
    norm_phase = mcolors.BoundaryNorm([0, 1, 2, 3], 3)

    fig = plt.figure(figsize=(18, 8), dpi=110)

    # ── 左：重采样后的 RGB（已在 overlap_extent 范围内）──
    ax1 = _make_ax(fig, 121, overlap_extent, map_res=MAP_RES,
                   land_color='none', ocean_color='none')
    ax1.imshow(rgb,
               origin='lower',
               extent=overlap_extent,
               transform=_PC,
               interpolation='bilinear',
               zorder=1, alpha=0.95)
    ax1.set_title(f"AGRI RGB — overlap region\n{scene_id}  Δt={time_diff_min:.1f}min",
                  fontsize=11, fontweight='bold')

    # ── 右：MODIS CLP ──
    ax2 = _make_ax(fig, 122, overlap_extent, map_res=MAP_RES)
    ax2.pcolormesh(modis_lon, modis_lat, clp_simple,
                   cmap=cmap_phase, norm=norm_phase,
                   transform=_PC, shading='auto', zorder=1, alpha=0.92)
    ax2.set_title("MODIS MYD06 CLP (IR)",
                  fontsize=11, fontweight='bold')

    patches = [mpatches.Patch(color=phase_colors[k], label=phase_labels[k])
               for k in [0, 1, 2]]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f"RGB vs MODIS CLP — overlap region  ({scene_id})",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in ['png']:
        save_path = save_dir / f"rgb_clp_overlap.{fmt}"
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
    print(f"  [saved] rgb_clp_overlap.{{png}}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 单场景处理
# ═══════════════════════════════════════════════════════════════════

def process_one_scene(
    scene_id: str,
    l2_cth_path: str,
    l2_clp_path: str,
    myd06_path: str,
    myd03_path: str,
    retrieval_path: str,
    l2_time: datetime,
    modis_time: datetime,
) -> bool:
    """处理单个匹配场景，生成所有对比图。"""
    time_diff_min = abs((l2_time - modis_time).total_seconds()) / 60.0
    # save_dir = OUTPUT_DIR / scene_id
    # save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene: {scene_id}  Δt={time_diff_min:.2f} min")
    print(f"  L2 CTH:  {Path(l2_cth_path).name}")
    print(f"  L2 CLP:  {Path(l2_clp_path).name}")
    print(f"  MYD06:   {Path(myd06_path).name}")
    print(f"  MYD03:   {Path(myd03_path).name}")
    print(f"  Model:   {Path(retrieval_path).name}")

    # ── 1. 读取数据 ──
    print("--- 读取数据 ---")
    l2_cth = read_fy4a_l2_nc(l2_cth_path, 'CTH')
    l2_clp = read_fy4a_l2_nc(l2_clp_path, 'CLP')
    modis  = read_modis_myd06(myd06_path, myd03_path)
    model  = read_model_retrieval(retrieval_path)

    # ── 2. MODIS 落入 AGRI 检查 ──
    print("--- MODIS-in-AGRI 检查 ---")
    agri_lat = model['lat']   # 全圆盘经纬度 (2748,2748)
    agri_lon = model['lon']
    in_disk = check_modis_in_agri_disk(
        modis['lat'], modis['lon'], agri_lat, agri_lon, max_dist_km=MIN_DISK_DIST_KM)
    print(f"  MODIS in AGRI disk: {in_disk}")
    if not in_disk:
        print("  [skip] MODIS 不完全在 AGRI 圆盘内")
        return False
    save_dir = OUTPUT_DIR / scene_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. 空间覆盖总览 ──
    print("--- 空间覆盖总览 ---")
    try:
        plot_overlap_check(
            l2_cth['lon'], l2_cth['lat'],
            np.isfinite(l2_cth['data']),
            model['lon'], model['lat'],
            np.isfinite(model['data_cth']),
            modis['lon'], modis['lat'],
            np.isfinite(modis['cth']),
            time_diff_min, scene_id, save_dir,
        )
    except Exception as e:
        print(f"  [warn] 总览图失败: {e}")

    # ── 3.5 RGB + MODIS 叠加 （全盘 + 重合区域） ──
    print("--- RGB + MODIS 叠加 ---")
    try:
        fdi_name = Path(retrieval_path).name.replace('_retrieval.npz', '.HDF')
        fdi_path = str(FDI_DIR / fdi_name)
        if Path(fdi_path).exists():
            r, g, b = read_fdi_rgb(fdi_path)
            plot_rgb_with_modis(r, g, b,
                                agri_lon, agri_lat,
                                modis['lon'], modis['lat'],
                                time_diff_min, scene_id, save_dir)
            plot_rgb_clp_overlap(r, g, b,
                                 agri_lon, agri_lat,
                                 modis['lon'], modis['lat'],
                                 modis['clp'],
                                 time_diff_min, scene_id, save_dir)
        else:
            print(f"  [warn] FDI 文件不存在: {fdi_name}")
    except Exception as e:
        print(f"  [warn] RGB + MODIS 图失败: {e}")

    # ── 4. L2 vs MODIS: CTH ──
    print("--- L2 vs MODIS CTH ---")
    try:
        match_l2_cth = geo_match(
            l2_cth['lon'], l2_cth['lat'], l2_cth['data'],
            modis['lon'], modis['lat'], modis['cth'],
            grid_res_deg=GRID_RES_DEG, interp_method='nearest',
        )
        plot_cth_comparison(match_l2_cth, "FY4A L2 CTH", "MODIS MYD06 CTH",
                            time_diff_min, scene_id, save_dir, "l2_vs_modis")
        plot_cth_scatter(match_l2_cth, "FY4A L2",
                         time_diff_min, scene_id, save_dir, "l2_vs_modis")
        plot_cth_histogram(match_l2_cth, "FY4A L2",
                           time_diff_min, scene_id, save_dir, "l2_vs_modis")
    except ValueError as e:
        print(f"  [warn] L2 vs MODIS CTH 匹配失败: {e}")

    # ── 5. L2 vs MODIS: CLP ──
    print("--- L2 vs MODIS CLP ---")
    try:
        match_l2_clp = geo_match(
            l2_clp['lon'], l2_clp['lat'], l2_clp['data'],
            modis['lon'], modis['lat'], modis['clp'],
            grid_res_deg=GRID_RES_DEG, interp_method='nearest',
        )
        plot_clp_comparison(match_l2_clp, "FY4A L2 CLP", "MODIS MYD06 CLP",
                            clp_from='l2',
                            time_diff_min=time_diff_min, scene_id=scene_id,
                            save_dir=save_dir, prefix="l2_vs_modis")
    except ValueError as e:
        print(f"  [warn] L2 vs MODIS CLP 匹配失败: {e}")

    # ── 6. 模型 vs MODIS: CTH ──
    print("--- Model vs MODIS CTH ---")
    try:
        match_model_cth = geo_match(
            model['lon'], model['lat'], model['data_cth'],
            modis['lon'], modis['lat'], modis['cth'],
            grid_res_deg=GRID_RES_DEG, interp_method='nearest',
        )
        plot_cth_comparison(match_model_cth, "Model CTH", "MODIS MYD06 CTH",
                            time_diff_min, scene_id, save_dir, "model_vs_modis")
        plot_cth_scatter(match_model_cth, "Model",
                         time_diff_min, scene_id, save_dir, "model_vs_modis")
        plot_cth_histogram(match_model_cth, "Model",
                           time_diff_min, scene_id, save_dir, "model_vs_modis")
    except ValueError as e:
        print(f"  [warn] Model vs MODIS CTH 匹配失败: {e}")

    # ── 7. 模型 vs MODIS: CLP ──
    print("--- Model vs MODIS CLP ---")
    try:
        match_model_clp = geo_match(
            model['lon'], model['lat'], model['data_clp'],
            modis['lon'], modis['lat'], modis['clp'],
            grid_res_deg=GRID_RES_DEG, interp_method='nearest',
        )
        plot_clp_comparison(match_model_clp, "Model CLP", "MODIS MYD06 CLP",
                            clp_from='model',
                            time_diff_min=time_diff_min, scene_id=scene_id,
                            save_dir=save_dir, prefix="model_vs_modis")
    except ValueError as e:
        print(f"  [warn] Model vs MODIS CLP 匹配失败: {e}")

    # ── 8. 写场景信息 ──
    info_path = save_dir / "scene_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Scene: {scene_id}\n")
        f.write(f"Time diff: {time_diff_min:.2f} min\n")
        f.write(f"L2 CTH: {l2_cth_path}\n")
        f.write(f"L2 CLP: {l2_clp_path}\n")
        f.write(f"MYD06:  {myd06_path}\n")
        f.write(f"MYD03:  {myd03_path}\n")
        f.write(f"Model:  {retrieval_path}\n")

    print(f"  [done] Scene {scene_id} 完成")
    return True


# ═══════════════════════════════════════════════════════════════════
# 批量匹配主逻辑
# ═══════════════════════════════════════════════════════════════════

def find_matches() -> List[Dict]:
    """查找 20190505 所有三源匹配的场景。"""
    # 收集所有文件
    l2_cth_files = sorted(glob.glob(str(L2_CTH_DIR / "*.NC")))
    l2_clp_files = sorted(glob.glob(str(L2_CLP_DIR / "*.NC")))
    myd06_files  = sorted(glob.glob(str(MYD06_DIR / "*.hdf")))
    myd03_files  = sorted(glob.glob(str(MYD03_DIR / "*.hdf")))
    retrieval_files = sorted(glob.glob(str(RETRIEVAL_DIR / "*20190505*.npz")))

    print(f"L2 CTH files: {len(l2_cth_files)}")
    print(f"L2 CLP files: {len(l2_clp_files)}")
    print(f"MYD06 files:  {len(myd06_files)}")
    print(f"MYD03 files:  {len(myd03_files)}")
    print(f"Retrieval:    {len(retrieval_files)}")

    # 构建时间索引
    l2_cth_map = {}
    for f in l2_cth_files:
        try:
            t = extract_fy4a_time(f)
            l2_cth_map[t] = f
        except ValueError:
            pass

    l2_clp_map = {}
    for f in l2_clp_files:
        try:
            t = extract_fy4a_time(f)
            l2_clp_map[t] = f
        except ValueError:
            pass

    myd06_map = {}
    for f in myd06_files:
        try:
            t = extract_modis_time(f)
            myd06_map[t] = f
        except ValueError:
            pass

    myd03_map = {}
    for f in myd03_files:
        try:
            t = extract_modis_time(f)
            myd03_map[t] = f
        except ValueError:
            pass

    retrieval_map = {}
    for f in retrieval_files:
        try:
            t = extract_retrieval_time(f)
            retrieval_map[t] = f
        except ValueError:
            pass

    # 匹配: FY4A L2 时间 ±5min 内找最近的 MODIS
    matches = []
    for l2_time in sorted(l2_cth_map.keys()):
        if l2_time not in l2_clp_map:
            continue

        # 找最近的 MODIS
        best_modis_time = None
        best_dt = float('inf')
        for modis_time in sorted(myd06_map.keys()):
            dt = abs((l2_time - modis_time).total_seconds()) / 60.0
            if dt < best_dt and dt <= TIME_MAX_MIN and modis_time in myd03_map:
                best_dt = dt
                best_modis_time = modis_time

        if best_modis_time is None:
            continue

        # 找最近的模型检索结果
        best_retrieval_time = None
        best_rdt = float('inf')
        for ret_time in sorted(retrieval_map.keys()):
            rdt = abs((l2_time - ret_time).total_seconds()) / 60.0
            if rdt < best_rdt:
                best_rdt = rdt
                best_retrieval_time = ret_time

        if best_retrieval_time is None:
            continue

        scene_id = f"{DAY}_{l2_time.strftime('%H%M%S')}"

        matches.append({
            'scene_id': scene_id,
            'l2_cth_path': l2_cth_map[l2_time],
            'l2_clp_path': l2_clp_map[l2_time],
            'myd06_path': myd06_map[best_modis_time],
            'myd03_path': myd03_map[best_modis_time],
            'retrieval_path': retrieval_map[best_retrieval_time],
            'l2_time': l2_time,
            'modis_time': best_modis_time,
            'time_diff_min': best_dt,
        })

    print(f"\n找到 {len(matches)} 个潜在匹配 (time ≤ {TIME_MAX_MIN} min)")
    return matches


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def main():
    if not _HAS_CARTOPY:
        print("[ERROR] cartopy 未安装，请执行: pip install cartopy")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Batch Viz: L2 vs MODIS | Model vs MODIS")
    print(f"Date: {DAY}   Time max: {TIME_MAX_MIN} min")
    print("=" * 60)

    matches = find_matches()

    if not matches:
        print("No matching scenes found!")
        return

    print(f"\n开始处理 {len(matches)} 个匹配场景")
    print(f"(MODIS-in-AGRI-disk 检查将在每个场景中执行)")

    # 处理每个匹配（场景内部有 MODIS-in-AGRI 检查）
    n_ok = 0
    n_skip = 0
    for m in matches:
        try:
            ok = process_one_scene(**{k: m[k] for k in [
                'scene_id', 'l2_cth_path', 'l2_clp_path',
                'myd06_path', 'myd03_path', 'retrieval_path',
                'l2_time', 'modis_time',
            ]})
            if ok:
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            print(f"  [error] Scene {m['scene_id']} 处理失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"完成! 成功 {n_ok}  跳过 {n_skip}  总计 {len(matches)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
