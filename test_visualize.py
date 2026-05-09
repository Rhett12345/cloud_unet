"""
tools/test_visualize.py — MODIS–AGRI 时空对齐诊断
====================================================
诊断 MODIS MYD06 与 FY4A AGRI 之间是否存在时空错位。

只需指定日期，脚本自动找到时间最接近的 AGRI–MODIS 配对，然后运行
5 维诊断：

  D1  坐标叠加图       — MODIS 像元 lat/lon 散点 vs AGRI 网格
  D2  CTH 空间分布对比 — 各自坐标系 + 投影后对比
  D3  时间差热图       — MODIS 像元级扫描时间偏移
  D4  2D 互相关        — 检测系统性空间偏移（最有诊断力）
  D5  散点图           — 同网格像元 CTH 直接对比

用法：
  # 自动匹配当天第一个 AGRI 场景与最近的 MODIS
  python test_visualize.py --day 20190505

  # 指定具体时刻
  python test_visualize.py --day 20190505 --time 040000

  # 先看当天有哪些场景和 MODIS 配对
  python test_visualize.py --day 20190505 --list

  # 选择第 N 个场景（0-indexed）
  python test_visualize.py --day 20190505 --scene_index 2

输出（在 --out_dir 下）：
  diag_D1_coord_overlay.png
  diag_D2_cth_maps.png
  diag_D3_time_offset.png
  diag_D4_crosscorr.png
  diag_D5_scatter.png
  diag_summary.txt
"""

from __future__ import annotations
import argparse
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
import config as cfg
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom
from scipy.signal import correlate2d
from scipy.spatial import cKDTree
from fusion_io import _derive_latlon

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 底层读取
# ---------------------------------------------------------------------------

def _read_agri_geo_raw(geo_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """直接从 AGRI GEO HDF5 读取 lat/lon，若不存在则从轨道参数反算。"""
    import h5py
    with h5py.File(geo_file, "r") as f:
        for lat_key in ["Geolocation/NOMLatitude", "NOMLatitude", "Latitude"]:
            if lat_key in f:
                lat = f[lat_key][()].astype(np.float32)
                break
        else:
            return _derive_latlon(f)
        for lon_key in ["Geolocation/NOMLongitude", "Geolocation/NOMlongitude",
                         "NOMLongitude", "Longitude"]:
            if lon_key in f:
                lon = f[lon_key][()].astype(np.float32)
                break
        else:
            raise KeyError(f"Cannot find lon in {geo_file}")

    lat[lat > 1e4] = np.nan
    lon[lon > 1e4] = np.nan
    lon = (((lon + 180.0) % 360.0) - 180.0)
    return lat, lon


def _read_agri_cth_raw(nc_file: Path) -> Optional[np.ndarray]:
    """直接从 AGRI L2 CTH NetCDF 读取，不做 valid_range 过滤以外的处理。"""
    try:
        import netCDF4 as nc4
        ds = nc4.Dataset(str(nc_file), "r")
        v = ds.variables["CTH"]
        v.set_auto_mask(False)
        arr = np.asarray(v[:], dtype=np.float32)
        ds.close()
        arr[(arr <= 0) | (arr >= 65500)] = np.nan
        return arr
    except Exception as e:
        log.warning("Cannot read AGRI CTH %s: %s", nc_file, e)
        return None


def _read_myd03_latlon(myd03_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从 MYD03 读取 1km lat/lon。"""
    from pyhdf.SD import SD, SDC
    sd = SD(str(myd03_file), SDC.READ)
    lat = sd.select("Latitude")[:].astype(np.float32)
    lon = sd.select("Longitude")[:].astype(np.float32)
    sd.end()
    lat[~np.isfinite(lat)] = np.nan
    lon[~np.isfinite(lon)] = np.nan
    return lat, lon


def _read_myd06_5km_latlon(myd06_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从 MYD06 读取 5km lat/lon。"""
    from pyhdf.SD import SD, SDC
    sd = SD(str(myd06_file), SDC.READ)
    lat = sd.select("Latitude")[:].astype(np.float32)
    lon = sd.select("Longitude")[:].astype(np.float32)
    sd.end()
    return lat, lon


def _read_myd06_cth(myd06_file: Path) -> Optional[np.ndarray]:
    """
    从 MYD06 读取 CTH（Cloud_Top_Height，5km）。
    scale_factor / add_offset 来自 SDS 属性。
    """
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(myd06_file), SDC.READ)
        ds = sd.select("Cloud_Top_Height")
        raw = ds[:].astype(np.float32)
        attr = ds.attributes()
        fv = attr.get("_FillValue", -9999)
        raw[raw == fv] = np.nan
        sf = float(attr.get("scale_factor", 1.0))
        ao = float(attr.get("add_offset", 0.0))
        cth = raw * sf + ao          # 单位：m（通常为 m 或 hPa，需确认）
        # 若结果量级在 100~1100 范围，可能是 hPa（气压顶），跳过范围过滤
        if np.nanmedian(cth[np.isfinite(cth)]) < 2000:
            log.warning("MODIS CTH median=%.1f — may be in hPa, not meters. "
                        "Check MODIS_VARS cfg.", np.nanmedian(cth[np.isfinite(cth)]))
        sd.end()
        return cth
    except Exception as e:
        log.warning("Cannot read MYD06 CTH: %s", e)
        return None


def _read_myd06_clp(myd06_file: Path) -> Optional[np.ndarray]:
    """从 MYD06 读取 Cloud_Phase_Optical_Properties（1km）。"""
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(myd06_file), SDC.READ)
        ds = sd.select("Cloud_Phase_Optical_Properties")
        raw = ds[:].astype(np.float32)
        attr = ds.attributes()
        fv = attr.get("_FillValue", 255)
        raw[raw == fv] = np.nan
        sd.end()
        return raw
    except Exception as e:
        log.warning("Cannot read MYD06 CLP: %s", e)
        return None


def _read_myd06_scan_time(myd06_file: Path) -> Optional[np.ndarray]:
    """读取 MYD06 Scan_Start_Time（TAI93，5km）。"""
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(myd06_file), SDC.READ)
        t = sd.select("Scan_Start_Time")[:].astype(np.float64)
        sd.end()
        t[t < -1e9] = np.nan
        return t
    except Exception as e:
        log.warning("Cannot read Scan_Start_Time: %s", e)
        return None


def _read_myd03_scan_time(myd03_file: Path) -> Optional[np.ndarray]:
    """读取 MYD03 EV_start_time 或等效字段（TAI93，1km）。"""
    from pyhdf.SD import SD, SDC
    sd = SD(str(myd03_file), SDC.READ)
    for name in ("EV start time", "EV_start_time", "EV_Start_Time",
                 "EV center time", "EV_center_time"):
        try:
            t = sd.select(name)[:].astype(np.float64)
            sd.end()
            t[t < -1e9] = np.nan
            return t
        except Exception:
            continue
    sd.end()
    return None


# ---------------------------------------------------------------------------
# 坐标投影：MODIS → AGRI 网格
# ---------------------------------------------------------------------------

def project_modis_to_agri_grid(
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
    modis_data: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    search_radius_km: float = 5.0,
) -> np.ndarray:
    """
    将任意 MODIS 2D 数组（可以是 5km 或 1km）投影到 AGRI 网格。
    使用 KD-tree 最近邻，搜索半径 search_radius_km。
    返回与 agri_lat 形状相同的数组，无对应像元处为 NaN。
    """
    H_a, W_a = agri_lat.shape

    def xyz(lat, lon):
        lr, lo = np.deg2rad(lat), np.deg2rad(lon)
        return np.column_stack([
            np.cos(lr) * np.cos(lo),
            np.cos(lr) * np.sin(lo),
            np.sin(lr),
        ])

    chord = 2.0 * np.sin(search_radius_km / (2.0 * 6371.0))

    # MODIS 像元（展平，去 NaN）
    m_lat_f = modis_lat.ravel()
    m_lon_f = modis_lon.ravel()
    m_dat_f = modis_data.ravel()
    valid_m = np.isfinite(m_lat_f) & np.isfinite(m_lon_f) & np.isfinite(m_dat_f)
    if not valid_m.any():
        return np.full((H_a, W_a), np.nan, np.float32)

    m_xyz = xyz(m_lat_f[valid_m], m_lon_f[valid_m])
    m_val = m_dat_f[valid_m]
    tree = cKDTree(m_xyz)

    # AGRI 像元（展平，去 NaN）
    a_lat_f = agri_lat.ravel()
    a_lon_f = agri_lon.ravel()
    valid_a = np.isfinite(a_lat_f) & np.isfinite(a_lon_f)
    a_xyz = xyz(a_lat_f[valid_a], a_lon_f[valid_a])

    dist, idx = tree.query(a_xyz, k=1, distance_upper_bound=chord, workers=-1)

    out = np.full(H_a * W_a, np.nan, np.float32)
    found = idx < len(m_val)
    out_idx = np.where(valid_a)[0]
    out[out_idx[found]] = m_val[idx[found]]
    return out.reshape(H_a, W_a)


# ---------------------------------------------------------------------------
# 诊断 D4：2D 互相关
# ---------------------------------------------------------------------------

def compute_2d_crosscorr(
    arr_ref: np.ndarray,
    arr_test: np.ndarray,
    max_shift_px: int = 15,
    downsample: int = 4,
) -> dict:
    """
    计算 arr_ref 与 arr_test（两者在同一网格上）之间的 2D 归一化互相关。
    返回峰值偏移 (row_shift, col_shift) 和峰值相关系数。

    downsample : 先降采样，加速计算，对亚像元偏移的灵敏度有限制。
    max_shift_px : 仅在此范围内搜索峰值。
    """
    # 用有限像元的公共掩码
    valid = np.isfinite(arr_ref) & np.isfinite(arr_test)
    if valid.sum() < 100:
        return {"status": "insufficient_data", "row_shift": 0, "col_shift": 0, "peak_r": np.nan}

    # 降采样后填 0（互相关对均值敏感）
    def prep(arr):
        a = arr.copy()
        a[~valid] = np.nan
        # 降采样
        a_ds = a[::downsample, ::downsample]
        v_ds = valid[::downsample, ::downsample]
        mn = np.nanmean(a_ds[v_ds])
        std = np.nanstd(a_ds[v_ds])
        if std < 1e-6:
            return None
        a_ds = np.where(v_ds, (a_ds - mn) / std, 0.0)
        return a_ds

    r = prep(arr_ref)
    t = prep(arr_test)
    if r is None or t is None:
        return {"status": "no_variance", "row_shift": 0, "col_shift": 0, "peak_r": np.nan}

    corr = correlate2d(r, t, mode="full", boundary="fill", fillvalue=0)
    # 归一化
    norm = float(np.sqrt((r**2).sum() * (t**2).sum()))
    if norm > 0:
        corr /= norm

    # 搜索范围：仅看中心 ±max_shift_px 像元（降采样坐标）
    mid_r, mid_c = np.array(corr.shape) // 2
    max_ds = max(1, max_shift_px // downsample)
    r0, r1 = max(0, mid_r - max_ds), min(corr.shape[0], mid_r + max_ds + 1)
    c0, c1 = max(0, mid_c - max_ds), min(corr.shape[1], mid_c + max_ds + 1)
    sub = corr[r0:r1, c0:c1]
    peak_pos = np.unravel_index(np.argmax(sub), sub.shape)
    peak_r = float(sub[peak_pos])

    # 换算回像元偏移（×downsample）
    row_shift = int((peak_pos[0] - (r1 - r0) // 2) * downsample)
    col_shift = int((peak_pos[1] - (c1 - c0) // 2) * downsample)

    # 用真实坐标估计偏移距离（km）
    return {
        "status": "ok",
        "row_shift": row_shift,
        "col_shift": col_shift,
        "peak_r": peak_r,
        "corr_map": sub,
        "downsample": downsample,
    }


# ---------------------------------------------------------------------------
# 时间差计算（MODIS scan time → 分钟偏移）
# ---------------------------------------------------------------------------

def scan_time_to_offset_min(
    scan_tai93: np.ndarray,
    ref_dt: datetime,
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """TAI93 秒 → 相对 ref_dt 的分钟偏移，展开到 target_shape。"""
    tai_epoch = datetime(1993, 1, 1)
    ref_sec = (ref_dt - tai_epoch).total_seconds()
    arr = np.asarray(scan_tai93, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    offset_min = ((arr - ref_sec) / 60.0).astype(np.float32)

    H, W = target_shape
    if offset_min.ndim == 1:
        rpt = max(1, int(np.ceil(H / offset_min.shape[0])))
        row_vals = np.repeat(offset_min, rpt)[:H]
        return np.tile(row_vals[:, None], (1, W))
    if offset_min.ndim == 2:
        rh = max(1, int(np.ceil(H / offset_min.shape[0])))
        rw = max(1, int(np.ceil(W / offset_min.shape[1])))
        g = np.repeat(np.repeat(offset_min, rh, 0), rw, 1)
        g = g[:H, :W]
        return g
    return None


# ---------------------------------------------------------------------------
# 诊断绘图
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def _subsample(arr2d, n=40000):
    """随机抽 n 个非 NaN 像元，返回 (row, col) 索引。"""
    valid = np.isfinite(arr2d)
    idx = np.where(valid.ravel())[0]
    if len(idx) > n:
        idx = np.random.choice(idx, n, replace=False)
    rows, cols = np.unravel_index(idx, arr2d.shape)
    return rows, cols


def plot_D1_coord_overlay(
    agri_lat, agri_lon,
    modis_lat_1km, modis_lon_1km,
    modis_lat_5km, modis_lon_5km,
    out_path: Path,
):
    """D1：坐标叠加图。AGRI 网格灰点 + MODIS 1km 彩点（按行着色）+ 5km 轮廓。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (mlat, mlon, label, marker_size) in zip(
        axes,
        [
            (modis_lat_1km, modis_lon_1km, "MYD03 1km", 0.3),
            (modis_lat_5km, modis_lon_5km, "MYD06 5km", 4.0),
        ]
    ):
        # AGRI 网格背景
        ar, ac = _subsample(agri_lat, 30000)
        ax.scatter(agri_lon[ar, ac], agri_lat[ar, ac],
                   s=0.2, c="lightgrey", alpha=0.4, rasterized=True, label="AGRI grid")

        # MODIS 像元（按行着色，便于识别扫描条带方向）
        if mlat is not None and mlon is not None:
            nrow = mlat.shape[0]
            mr, mc = _subsample(mlat, 20000)
            colors = plt.cm.plasma(mr / max(nrow - 1, 1))
            ax.scatter(mlon[mr, mc], mlat[mr, mc],
                       s=marker_size, c=colors, alpha=0.7, rasterized=True,
                       label=label)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"D1 Coord Overlay — {label}\n"
                     f"AGRI灰色 / MODIS按行号着色（紫→黄=首行→末行）")
        ax.legend(markerscale=4, fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("D1: MODIS vs AGRI 坐标叠加\n"
                 "如果对应云区的彩点落在灰点覆盖范围内 → 坐标系一致", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("D1 saved: %s", out_path)


def plot_D2_cth_maps(
    agri_lat, agri_lon, agri_cth,
    modis_lat_5km, modis_lon_5km, modis_cth_5km,
    modis_cth_on_agri_grid,
    out_path: Path,
):
    """D2：CTH 空间分布对比（各自坐标系 + 投影到 AGRI 网格后差值图）。"""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    vmin, vmax = 0, 15000
    cmap = "RdYlBu_r"

    # --- 面板 1：AGRI L2 CTH（自身坐标系）---
    ax1 = fig.add_subplot(gs[0])
    valid_a = np.isfinite(agri_cth) & np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if valid_a.any():
        sc = ax1.scatter(agri_lon[valid_a], agri_lat[valid_a],
                         c=agri_cth[valid_a], s=0.3, vmin=vmin, vmax=vmax,
                         cmap=cmap, rasterized=True)
        plt.colorbar(sc, ax=ax1, label="CTH (m)", fraction=0.04)
    ax1.set_title("AGRI L2 CTH\n(自身坐标系)", fontsize=8)
    ax1.set_xlabel("Lon"); ax1.set_ylabel("Lat")

    # --- 面板 2：MODIS CTH（5km 自身坐标系）---
    ax2 = fig.add_subplot(gs[1])
    valid_m = np.isfinite(modis_cth_5km) & np.isfinite(modis_lat_5km) & np.isfinite(modis_lon_5km)
    if valid_m.any():
        sc = ax2.scatter(modis_lon_5km[valid_m], modis_lat_5km[valid_m],
                         c=modis_cth_5km[valid_m], s=2.0, vmin=vmin, vmax=vmax,
                         cmap=cmap, rasterized=True)
        plt.colorbar(sc, ax=ax2, label="CTH (m)", fraction=0.04)
    ax2.set_title("MODIS MYD06 CTH\n(5km 自身坐标系)", fontsize=8)
    ax2.set_xlabel("Lon"); ax2.set_ylabel("Lat")

    # --- 面板 3：MODIS CTH 投影到 AGRI 网格 ---
    ax3 = fig.add_subplot(gs[2])
    valid_p = np.isfinite(modis_cth_on_agri_grid) & np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if valid_p.any():
        sc = ax3.scatter(agri_lon[valid_p], agri_lat[valid_p],
                         c=modis_cth_on_agri_grid[valid_p], s=0.3, vmin=vmin, vmax=vmax,
                         cmap=cmap, rasterized=True)
        plt.colorbar(sc, ax=ax3, label="CTH (m)", fraction=0.04)
    ax3.set_title("MODIS CTH 投影到\nAGRI 4km 网格", fontsize=8)
    ax3.set_xlabel("Lon"); ax3.set_ylabel("Lat")

    # --- 面板 4：差值图（MODIS - AGRI） ---
    ax4 = fig.add_subplot(gs[3])
    both = np.isfinite(agri_cth) & np.isfinite(modis_cth_on_agri_grid)
    diff = np.full_like(agri_cth, np.nan)
    diff[both] = modis_cth_on_agri_grid[both] - agri_cth[both]
    bias = float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan
    rmse = float(np.sqrt(np.nanmean(diff**2))) if np.isfinite(diff).any() else np.nan
    valid_d = np.isfinite(diff) & np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if valid_d.any():
        lim = max(3000, float(np.nanpercentile(np.abs(diff[valid_d]), 90)))
        sc = ax4.scatter(agri_lon[valid_d], agri_lat[valid_d],
                         c=diff[valid_d], s=0.3, vmin=-lim, vmax=lim,
                         cmap="RdBu", rasterized=True)
        plt.colorbar(sc, ax=ax4, label="ΔCTH (m)", fraction=0.04)
    ax4.set_title(f"差值 (MODIS - AGRI)\n"
                  f"bias={bias:+.0f}m  RMSE={rmse:.0f}m", fontsize=8)
    ax4.set_xlabel("Lon"); ax4.set_ylabel("Lat")

    fig.suptitle("D2: CTH 空间分布对比\n"
                 "左两图形态若一致 → 产品本身正常；面板3若云型错位 → 投影出了问题",
                 fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("D2 saved: %s", out_path)


def plot_D3_time_offset(
    agri_lat, agri_lon, agri_dt,
    modis_scan_time_5km,     # shape: 5km grid, TAI93 秒
    modis_lat_5km, modis_lon_5km,
    out_path: Path,
):
    """D3：MODIS 像元级时间差热图（投影到 AGRI 网格）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 左图：MODIS scan_time 在 5km 自身坐标系 ──
    ax = axes[0]
    if modis_scan_time_5km is not None:
        offset = scan_time_to_offset_min(modis_scan_time_5km, agri_dt,
                                         modis_lat_5km.shape)
        valid_m = (np.isfinite(offset) & np.isfinite(modis_lat_5km)
                   & np.isfinite(modis_lon_5km))
        if valid_m.any():
            lim = max(1.0, float(np.nanpercentile(np.abs(offset[valid_m]), 95)))
            sc = ax.scatter(modis_lon_5km[valid_m], modis_lat_5km[valid_m],
                            c=offset[valid_m], s=3.0, vmin=-lim, vmax=lim,
                            cmap="RdBu_r", rasterized=True)
            plt.colorbar(sc, ax=ax, label="Δt (min, MODIS−AGRI)")
            ax.set_title(f"D3a MODIS scan_time offset\n"
                         f"(AGRI obs @ {agri_dt:%H:%M}UTC)", fontsize=8)
            median_dt = float(np.nanmedian(np.abs(offset[valid_m])))
            max_dt = float(np.nanmax(np.abs(offset[valid_m])))
            ax.set_xlabel(f"Lon   |  median |Δt|={median_dt:.1f}min  max={max_dt:.1f}min")
            ax.set_ylabel("Lat")
        else:
            ax.text(0.5, 0.5, "No valid scan time", transform=ax.transAxes, ha="center")
    else:
        ax.text(0.5, 0.5, "Scan_Start_Time unavailable", transform=ax.transAxes, ha="center")
    ax.set_aspect("equal", adjustable="datalim")

    # ── 右图：时间差直方图 ──
    ax2 = axes[1]
    if modis_scan_time_5km is not None and 'offset' in dir():
        vals = offset[np.isfinite(offset)]
        if vals.size:
            ax2.hist(vals, bins=60, color="#2E86AB", alpha=0.85, edgecolor="none")
            ax2.axvline(0, color="k", lw=1.2, linestyle="--", label="同时观测")
            ax2.axvline(np.nanmedian(vals), color="tomato", lw=1.2,
                        linestyle="--", label=f"中位数 {np.nanmedian(vals):+.1f}min")
            ax2.axvspan(-7.5, 7.5, alpha=0.12, color="green",
                        label="±7.5min 融合窗口")
            ax2.set_xlabel("Δt (min)")
            ax2.set_ylabel("像元数")
            ax2.set_title("D3b MODIS 时间差分布\n（若峰值偏离0则存在系统性时间错位）")
            ax2.legend(fontsize=7)

    fig.suptitle("D3: MODIS 像元级时间差（MODIS 观测时刻 − AGRI 观测时刻）\n"
                 "红色=MODIS更晚，蓝色=MODIS更早；若整体偏红/偏蓝说明时间窗口设置有问题", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("D3 saved: %s", out_path)


def plot_D4_crosscorr(
    agri_cth, modis_cth_on_agri_grid,
    agri_lat, agri_lon,
    ccr_result: dict,
    out_path: Path,
    pixel_size_km: float = 4.0,
):
    """D4：2D 互相关峰值图，直接显示系统性空间偏移。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── 左：AGRI CTH（用于互相关的输入）──
    ax = axes[0]
    valid = np.isfinite(agri_cth) & np.isfinite(agri_lat)
    if valid.any():
        im = ax.imshow(np.where(valid, agri_cth, np.nan),
                       origin="upper", cmap="RdYlBu_r",
                       vmin=0, vmax=15000, aspect="auto")
        plt.colorbar(im, ax=ax, label="CTH (m)", fraction=0.04)
    ax.set_title("AGRI CTH（参考）", fontsize=8)

    # ── 中：MODIS CTH 投影到 AGRI 网格 ──
    ax = axes[1]
    valid2 = np.isfinite(modis_cth_on_agri_grid) & np.isfinite(agri_lat)
    if valid2.any():
        im = ax.imshow(np.where(valid2, modis_cth_on_agri_grid, np.nan),
                       origin="upper", cmap="RdYlBu_r",
                       vmin=0, vmax=15000, aspect="auto")
        plt.colorbar(im, ax=ax, label="CTH (m)", fraction=0.04)
    ax.set_title("MODIS CTH → AGRI 网格", fontsize=8)

    # ── 右：互相关热图 ──
    ax = axes[2]
    if ccr_result.get("status") == "ok" and "corr_map" in ccr_result:
        cm = ccr_result["corr_map"]
        ds = ccr_result.get("downsample", 1)
        row_s = ccr_result["row_shift"]
        col_s = ccr_result["col_shift"]
        peak_r = ccr_result["peak_r"]
        # 轴标签换算成像元偏移（未降采样）
        n_r, n_c = cm.shape
        row_ticks = np.linspace(-(n_r // 2) * ds, (n_r // 2) * ds, min(5, n_r))
        col_ticks = np.linspace(-(n_c // 2) * ds, (n_c // 2) * ds, min(5, n_c))
        extent = [col_ticks[0], col_ticks[-1], row_ticks[-1], row_ticks[0]]
        im = ax.imshow(cm, cmap="hot", origin="upper", aspect="auto", extent=extent,
                       vmin=0, vmax=cm.max())
        plt.colorbar(im, ax=ax, label="归一化相关系数", fraction=0.04)
        ax.axhline(0, color="cyan", lw=0.8, linestyle="--")
        ax.axvline(0, color="cyan", lw=0.8, linestyle="--")
        ax.plot(col_s, row_s, "r+", markersize=12, markeredgewidth=2,
                label=f"峰值 ({col_s:+d},{row_s:+d}) px")
        ax.set_xlabel("列偏移 (AGRI像元)")
        ax.set_ylabel("行偏移 (AGRI像元)")
        dist_km = np.sqrt(row_s**2 + col_s**2) * pixel_size_km
        ax.set_title(
            f"D4: 2D 互相关峰值\n"
            f"峰值偏移: row={row_s:+d} col={col_s:+d} px"
            f"  ≈ {dist_km:.0f} km\n"
            f"峰值相关系数: {peak_r:.3f}\n"
            f"{'⚠ 存在系统性空间偏移!' if (abs(row_s)>1 or abs(col_s)>1) else '✓ 无显著空间偏移'}",
            fontsize=8, color="red" if (abs(row_s) > 1 or abs(col_s) > 1) else "green",
        )
        ax.legend(fontsize=7)
    elif ccr_result.get("status") == "insufficient_data":
        ax.text(0.5, 0.5, "数据重叠区域不足，无法计算互相关",
                transform=ax.transAxes, ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, f"互相关失败: {ccr_result.get('status', 'unknown')}",
                transform=ax.transAxes, ha="center", fontsize=8)

    fig.suptitle("D4: 2D 互相关 — 检测系统性空间偏移\n"
                 "峰值在中心(0,0)=无偏移；偏离中心=MODIS相对AGRI系统性平移",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("D4 saved: %s", out_path)


def plot_D5_scatter_collocated(
    agri_cth, modis_cth_on_agri_grid,
    out_path: Path,
    scene_id: str = "",
):
    """D5：AGRI vs MODIS CTH 散点图（同一网格像元对比）。"""
    both = np.isfinite(agri_cth) & np.isfinite(modis_cth_on_agri_grid)
    n = int(both.sum())
    if n < 10:
        log.warning("D5: too few collocated pixels (%d), skipping scatter", n)
        return

    x = agri_cth[both]
    y = modis_cth_on_agri_grid[both]

    # 相关系数
    xm, ym = x - x.mean(), y - y.mean()
    r = float((xm * ym).sum() / (np.sqrt((xm**2).sum() * (ym**2).sum()) + 1e-12))
    rmse = float(np.sqrt(np.mean((y - x)**2)))
    bias = float(np.mean(y - x))

    fig, ax = plt.subplots(figsize=(6, 6))
    # 密度散点（降采样）
    idx = np.random.choice(n, min(n, 10000), replace=False)
    ax.scatter(x[idx], y[idx], s=1.5, alpha=0.3, color="#2E86AB", rasterized=True)
    lim = max(float(np.nanpercentile(np.concatenate([x, y]), 99)), 1000)
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, alpha=0.5, label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("AGRI L2 CTH (m)")
    ax.set_ylabel("MODIS MYD06 CTH (m) → AGRI 网格")
    ax.set_title(
        f"D5: 同网格像元 CTH 对比  {scene_id}\n"
        f"n={n}  R={r:.3f}  RMSE={rmse:.0f}m  bias={bias:+.0f}m\n"
        f"{'⚠ R<0.5，很可能存在错位或产品定义差异' if r < 0.5 else '✓ R 尚可'}\n"
        f"{'⚠ |bias|>1000m，注意单位或定义差异' if abs(bias) > 1000 else ''}",
        fontsize=8,
        color="red" if r < 0.5 else "black",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("D5 saved: %s  R=%.3f RMSE=%.0f bias=%+.0f", out_path, r, rmse, bias)


# ---------------------------------------------------------------------------
# 文字摘要
# ---------------------------------------------------------------------------

def write_summary(
    out_path: Path,
    agri_dt: datetime,
    modis_file: str,
    myd03_file: Optional[str],
    agri_cth: np.ndarray,
    modis_cth_on_agri_grid: np.ndarray,
    ccr_result: dict,
    modis_scan_time_5km: Optional[np.ndarray],
    agri_lat: np.ndarray,
    pixel_size_km: float = 4.0,
):
    both = np.isfinite(agri_cth) & np.isfinite(modis_cth_on_agri_grid)
    n = int(both.sum())
    lines = [
        "=" * 60,
        "MODIS–AGRI 时空错位诊断摘要",
        "=" * 60,
        f"AGRI 观测时刻    : {agri_dt:%Y-%m-%d %H:%M:%S} UTC",
        f"MODIS 文件       : {Path(modis_file).name}",
        f"MYD03 文件       : {Path(myd03_file).name if myd03_file else '未提供'}",
        "",
        "── 数据覆盖 ──",
        f"AGRI CTH 有效像元: {int(np.isfinite(agri_cth).sum())}",
        f"MODIS CTH 有效像元: {int(np.isfinite(modis_cth_on_agri_grid).sum())}",
        f"两者重叠像元     : {n}",
        "",
    ]

    if n > 10:
        x = agri_cth[both]
        y = modis_cth_on_agri_grid[both]
        xm, ym = x - x.mean(), y - y.mean()
        r = float((xm * ym).sum() / (np.sqrt((xm**2).sum() * (ym**2).sum()) + 1e-12))
        rmse = float(np.sqrt(np.mean((y - x)**2)))
        bias = float(np.mean(y - x))
        lines += [
            "── CTH 对比统计（重叠像元）──",
            f"Pearson R        : {r:.4f}",
            f"RMSE             : {rmse:.0f} m",
            f"Bias (MODIS-AGRI): {bias:+.0f} m",
            "",
        ]
    else:
        lines += ["⚠ 重叠像元不足，跳过 CTH 统计", ""]

    if ccr_result.get("status") == "ok":
        rs = ccr_result["row_shift"]
        cs = ccr_result["col_shift"]
        dist_km = np.sqrt(rs**2 + cs**2) * pixel_size_km
        lines += [
            "── D4 空间互相关 ──",
            f"峰值行偏移       : {rs:+d} 像元  ({rs * pixel_size_km:+.0f} km)",
            f"峰值列偏移       : {cs:+d} 像元  ({cs * pixel_size_km:+.0f} km)",
            f"合计偏移距离     : {dist_km:.0f} km",
            f"峰值相关系数     : {ccr_result['peak_r']:.3f}",
            "",
            "诊断结论 (空间)  : " + (
                "⚠ 存在系统性空间偏移！偏移量已超过 1 像元（4 km）。"
                "可能原因：①坐标系定义差异 ②视差效应 ③聚合搜索半径过小"
                if (abs(rs) > 1 or abs(cs) > 1)
                else "✓ 未发现显著系统性空间偏移（偏移 ≤ 1 像元）"
            ),
            "",
        ]
    else:
        lines += [f"D4 互相关: {ccr_result.get('status')}", ""]

    if modis_scan_time_5km is not None:
        tai_epoch = datetime(1993, 1, 1)
        ref_sec = (agri_dt - tai_epoch).total_seconds()
        dt_vals = (modis_scan_time_5km.ravel() - ref_sec) / 60.0
        dt_vals = dt_vals[np.isfinite(dt_vals)]
        if dt_vals.size:
            lines += [
                "── D3 时间差统计 ──",
                f"|Δt| 中位数      : {float(np.median(np.abs(dt_vals))):.1f} min",
                f"|Δt| 最大        : {float(np.max(np.abs(dt_vals))):.1f} min",
                f"Δt 均值          : {float(np.mean(dt_vals)):+.1f} min",
                "诊断结论 (时间)  : " + (
                    f"⚠ 最大时间差 {float(np.max(np.abs(dt_vals))):.1f}min > 7.5min，"
                    "时间窗口可能过宽"
                    if float(np.max(np.abs(dt_vals))) > 7.5
                    else "✓ 时间差均在 7.5min 以内"
                ),
                "",
            ]
    else:
        lines += ["D3: Scan_Start_Time 不可用", ""]

    lines += ["=" * 60]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    log.info("Summary saved: %s", out_path)
    print(text)


# ---------------------------------------------------------------------------
# 文件自动发现与时间匹配
# ---------------------------------------------------------------------------

def _parse_agri_ts(filename: str) -> Optional[str]:
    m = re.search(r"(\d{8})(\d{6})", filename)
    return m.group(1) + m.group(2) if m else None


def _parse_modis_dt(filename: str) -> Optional[datetime]:
    """MYD06_L2.A2019125.0510.061.hdf → datetime(2019,5,5,5,10)"""
    m = re.search(r"A(\d{7})\.(\d{4})", filename)
    if not m:
        return None
    yday, hhmm = m.group(1), m.group(2)
    year, doy = int(yday[:4]), int(yday[4:])
    return datetime(year, 1, 1) + timedelta(
        days=doy - 1, hours=int(hhmm[:2]), minutes=int(hhmm[2:]))


def _find_agri_scenes(day: str) -> list[tuple[Path, str, datetime]]:
    """Return [(filepath, timestamp_str, datetime), ...] sorted by time."""
    agri_dir = cfg.AGRI_ROOT / day
    if not agri_dir.is_dir():
        return []
    scenes = []
    for f in sorted(agri_dir.glob("*_FDI-_*.HDF")):
        ts = _parse_agri_ts(f.name)
        if ts:
            scenes.append((f, ts, datetime.strptime(ts, "%Y%m%d%H%M%S")))
    return scenes


def _find_l2_cth(day: str, ts: str) -> Optional[Path]:
    l2_dir = cfg.FY4A_L2_ROOT / "CTH" / day
    if not l2_dir.is_dir():
        return None
    for f in sorted(l2_dir.glob("*_CTH-_*.NC")):
        if ts in f.name:
            return f
    return None


def _find_modis_list(day: str) -> list[tuple[Path, datetime]]:
    """Return [(filepath, datetime), ...] sorted by time."""
    d = cfg.MODIS_ROOT / day
    if not d.is_dir():
        return []
    files = []
    for f in sorted(list(d.glob("MYD06*.hdf")) + list(d.glob("MYD06*.HDF"))):
        dt = _parse_modis_dt(f.name)
        if dt:
            files.append((f, dt))
    return sorted(files, key=lambda x: x[1])


def _find_myd03_map(day: str) -> dict[datetime, Path]:
    """Return {datetime: filepath} keyed by parsed time."""
    d = cfg.MYD03_ROOT / day
    if not d.is_dir():
        return {}
    out = {}
    for f in sorted(list(d.glob("MYD03*.hdf")) + list(d.glob("MYD03*.HDF"))):
        dt = _parse_modis_dt(f.name)
        if dt:
            out[dt] = f
    return out


def _find_closest_modis(
    agri_dt: datetime,
    modis_files: list[tuple[Path, datetime]],
    myd03_map: dict[datetime, Path],
    max_dt_min: float = 15.0,
) -> tuple[Optional[Path], Optional[Path], float]:
    """Return (modis_path, myd03_path, dt_minutes) for closest match."""
    if not modis_files:
        return None, None, float("inf")
    best, best_dt = None, float("inf")
    for mp, mdt in modis_files:
        diff = abs((mdt - agri_dt).total_seconds()) / 60.0
        if diff < best_dt:
            best_dt = diff
            best = mp
    if best_dt > max_dt_min:
        return None, None, best_dt
    # Match MYD03 by closest time
    best_m03 = None
    best_m03_dt = float("inf")
    for mdt, mp in myd03_map.items():
        diff = abs((mdt - agri_dt).total_seconds()) / 60.0
        if diff < best_m03_dt:
            best_m03_dt = diff
            best_m03 = mp
    return best, best_m03, best_dt


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_diagnostics(
    agri_fdi_file: Path,
    agri_cth_nc: Optional[Path],
    modis_file: Path,
    myd03_file: Optional[Path],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 解析 AGRI 时刻 ──
    import re
    m = re.search(r"(\d{8})(\d{6})", agri_fdi_file.name)
    if m is None:
        raise ValueError(f"Cannot parse datetime from {agri_fdi_file.name}")
    agri_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    scene_id = f"{m.group(1)}_{m.group(2)}"
    log.info("AGRI datetime: %s", agri_dt)

    # ── 读取 AGRI 坐标 ──
    geo_file = Path(str(agri_fdi_file).replace("_FDI-_", "_GEO-_"))
    if not geo_file.exists():
        raise FileNotFoundError(f"AGRI GEO file not found: {geo_file}")
    log.info("Reading AGRI GEO: %s", geo_file)
    agri_lat, agri_lon = _read_agri_geo_raw(geo_file)
    log.info("  AGRI grid shape: %s", agri_lat.shape)

    # ── 读取 AGRI CTH ──
    agri_cth = None
    if agri_cth_nc is not None and agri_cth_nc.exists():
        log.info("Reading AGRI CTH: %s", agri_cth_nc)
        agri_cth = _read_agri_cth_raw(agri_cth_nc)
    if agri_cth is None:
        log.warning("AGRI CTH not available; D2/D4/D5 will be skipped")
        agri_cth = np.full_like(agri_lat, np.nan)

    # ── 读取 MODIS 坐标和产品 ──
    log.info("Reading MYD06 5km geo: %s", modis_file)
    modis_lat_5km, modis_lon_5km = _read_myd06_5km_latlon(modis_file)
    log.info("  MODIS 5km shape: %s", modis_lat_5km.shape)

    modis_lat_1km, modis_lon_1km = None, None
    if myd03_file is not None and myd03_file.exists():
        log.info("Reading MYD03 1km geo: %s", myd03_file)
        modis_lat_1km, modis_lon_1km = _read_myd03_latlon(myd03_file)
        log.info("  MYD03 1km shape: %s", modis_lat_1km.shape)
    else:
        log.warning("MYD03 not provided; D1 will only show 5km geo")

    log.info("Reading MYD06 CTH (5km)...")
    modis_cth_5km = _read_myd06_cth(modis_file)
    if modis_cth_5km is None:
        log.warning("MODIS CTH unavailable; D2/D4 will show empty")
        modis_cth_5km = np.full_like(modis_lat_5km, np.nan)

    log.info("Reading MYD06 Scan_Start_Time...")
    modis_scan_t_5km = _read_myd06_scan_time(modis_file)

    # ── 将 MODIS CTH 投影到 AGRI 网格（两种分辨率都试） ──
    log.info("Projecting MODIS 5km CTH → AGRI grid (radius=5km)...")
    modis_cth_on_agri = project_modis_to_agri_grid(
        modis_lat_5km, modis_lon_5km, modis_cth_5km,
        agri_lat, agri_lon, search_radius_km=5.0,
    )
    n_proj = int(np.isfinite(modis_cth_on_agri).sum())
    log.info("  Projected pixels: %d / %d AGRI pixels", n_proj, int(np.isfinite(agri_lat).sum()))

    if n_proj < 100:
        log.warning("Very few projected pixels (%d)! The MODIS granule may not overlap "
                    "with the AGRI scene, or the coordinate systems are completely mismatched.", n_proj)

    # ── D1：坐标叠加 ──
    log.info("Plotting D1 (coord overlay)...")
    plot_D1_coord_overlay(
        agri_lat, agri_lon,
        modis_lat_1km, modis_lon_1km,
        modis_lat_5km, modis_lon_5km,
        out_dir / "diag_D1_coord_overlay.png",
    )

    # ── D2：CTH 空间对比 ──
    log.info("Plotting D2 (CTH maps)...")
    plot_D2_cth_maps(
        agri_lat, agri_lon, agri_cth,
        modis_lat_5km, modis_lon_5km, modis_cth_5km,
        modis_cth_on_agri,
        out_dir / "diag_D2_cth_maps.png",
    )

    # ── D3：时间差 ──
    log.info("Plotting D3 (time offset)...")
    plot_D3_time_offset(
        agri_lat, agri_lon, agri_dt,
        modis_scan_t_5km,
        modis_lat_5km, modis_lon_5km,
        out_dir / "diag_D3_time_offset.png",
    )

    # ── D4：2D 互相关 ──
    log.info("Computing D4 (2D cross-correlation)...")
    ccr = compute_2d_crosscorr(agri_cth, modis_cth_on_agri,
                                max_shift_px=20, downsample=4)
    plot_D4_crosscorr(agri_cth, modis_cth_on_agri, agri_lat, agri_lon, ccr,
                      out_dir / "diag_D4_crosscorr.png")

    # ── D5：散点图 ──
    log.info("Plotting D5 (collocation scatter)...")
    plot_D5_scatter_collocated(
        agri_cth, modis_cth_on_agri,
        out_dir / "diag_D5_scatter.png",
        scene_id=scene_id,
    )

    # ── 文字摘要 ──
    write_summary(
        out_dir / "diag_summary.txt",
        agri_dt, str(modis_file), str(myd03_file) if myd03_file else None,
        agri_cth, modis_cth_on_agri, ccr, modis_scan_t_5km,
        agri_lat, pixel_size_km=4.0,
    )

    log.info("All diagnostics written to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="诊断 MODIS MYD06 与 AGRI 之间的时空错位")
    parser.add_argument("--day", required=True, help="日期 YYYYMMDD")
    parser.add_argument("--time", default=None, help="AGRI 时刻 HHMMSS（默认使用第一个场景）")
    parser.add_argument("--scene_index", type=int, default=0, help="使用当天第 N 个 AGRI 场景（0-indexed）")
    parser.add_argument("--list", action="store_true", help="仅列出当天可用场景和 MODIS 配对")
    parser.add_argument("--out_dir", default="diag_alignment", help="诊断图输出目录")
    parser.add_argument("--max_dt_min", type=float, default=15.0, help="AGRI-MODIS 最大允许时间差（分钟）")
    args = parser.parse_args()

    day = args.day

    # ── 发现文件 ──
    agri_scenes = _find_agri_scenes(day)
    modis_files = _find_modis_list(day)
    myd03_map = _find_myd03_map(day)

    log.info("Day %s: %d AGRI scenes, %d MODIS, %d MYD03",
             day, len(agri_scenes), len(modis_files), len(myd03_map))

    if not agri_scenes:
        log.error("No AGRI FDI files found for day %s in %s", day, cfg.AGRI_ROOT / day)
        sys.exit(1)

    # ── --list 模式 ──
    if args.list:
        print(f"\nAGRI scenes for {day}:\n")
        for i, (f, ts, dt) in enumerate(agri_scenes):
            modis_f, _, dt_min = _find_closest_modis(dt, modis_files, myd03_map, args.max_dt_min)
            cth = _find_l2_cth(day, ts)
            cth_flag = "✓" if cth else "✗"
            modis_name = modis_f.name if modis_f else "NONE"
            print(f"  [{i}] {ts}  CTH={cth_flag}  →  {modis_name}  (Δ{dt_min:.0f}min)")
        print(f"\nMODIS granules for {day}:\n")
        for f, dt in modis_files:
            m03 = None
            for mdt, mp in myd03_map.items():
                if abs((mdt - dt).total_seconds()) < 120:
                    m03 = mp.name
                    break
            print(f"  {f.name}  ({dt:%Y-%m-%d %H:%M} UTC)  MYD03={'✓' if m03 else '✗'}")
        return

    # ── 选择 AGRI 场景 ──
    if args.time:
        target_ts = day + args.time
        match = None
        for f, ts, dt in agri_scenes:
            if ts == target_ts:
                match = (f, ts, dt)
                break
        if match is None:
            log.error("No AGRI scene at %s. Use --list to see available scenes.", target_ts)
            sys.exit(1)
        agri_fdi_file, ts, agri_dt = match
    else:
        if args.scene_index >= len(agri_scenes):
            log.error("Scene index %d out of range (0–%d)", args.scene_index, len(agri_scenes) - 1)
            sys.exit(1)
        agri_fdi_file, ts, agri_dt = agri_scenes[args.scene_index]

    log.info("AGRI scene:  %s", agri_fdi_file.name)

    # ── 找 L2 CTH ──
    agri_cth_nc = _find_l2_cth(day, ts)
    if agri_cth_nc:
        log.info("L2 CTH:      %s", agri_cth_nc.name)
    else:
        log.warning("No L2 CTH found for %s — CTH diagnostics will be skipped", ts)

    # ── 找最近 MODIS ──
    modis_file, myd03_file, dt_min = _find_closest_modis(
        agri_dt, modis_files, myd03_map, args.max_dt_min)
    if modis_file is None:
        log.error("No MODIS granule within %.0f min of AGRI %s", args.max_dt_min, ts)
        log.error("Closest dt=%.1f min. Use --max_dt_min to relax or --list to inspect.", dt_min)
        sys.exit(1)
    log.info("MODIS:       %s  (Δ%.1f min)", modis_file.name, dt_min)
    if myd03_file:
        log.info("MYD03:       %s", myd03_file.name)
    else:
        log.warning("No MYD03 found — D1 will only show 5km geo")

    run_diagnostics(
        agri_fdi_file=agri_fdi_file,
        agri_cth_nc=agri_cth_nc,
        modis_file=modis_file,
        myd03_file=myd03_file,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()