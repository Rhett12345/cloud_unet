"""
tools/geoloc_offset_diag.py — AGRI–MODIS 地理位置系统性偏移诊断
================================================================

检测 AGRI 预测场（或 AGRI L2 CTH）与 MODIS CTH 之间是否存在系统性
空间平移偏移，并量化偏移校正前后的验证指标差异。

诊断方法：
  1. 2D 归一化互相关（NCC）搜索最优平移
  2. 多场景聚合确认偏移是否具有系统性（方向 + 幅度一致）
  3. 偏移前后 RMSE / Bias / Correlation 对比
  4. 可视化：互相关热图、偏移矢量图、散点对比

用法：
  # 单场景诊断（使用 AGRI FDI + L2 CTH + MODIS MYD06）
  python tools/geoloc_offset_diag.py --day 20190505

  # 多场景批量诊断
  python tools/geoloc_offset_diag.py --days 20190501 20190502 20190503

  # 使用模型推理 npz 输出与 MODIS 对比
  python tools/geoloc_offset_diag.py --npz_dir /path/to/npz --modis_dir /path/to/modis

  # 指定搜索范围（像元数）
  python tools/geoloc_offset_diag.py --day 20190505 --max_shift 20

输出（在 --out_dir 下）：
  geoloc_offset_report.txt        — 汇总报告
  geoloc_offset_single_scene.png  — 单场景诊断图
  geoloc_offset_multi_scene.png   — 多场景聚合图
  geoloc_offset_before_after.png  — 偏移校正前后散点对比
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import correlate2d
from scipy.spatial import cKDTree

# ── 项目导入 ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 核心：2D 互相关偏移检测
# ═════════════════════════════════════════════════════════════════════════════

def compute_offset_ncc(
    ref: np.ndarray,
    test: np.ndarray,
    max_shift: int = 15,
    downsample: int = 2,
    min_overlap: int = 100,
) -> Dict:
    """
    计算 ref 与 test 之间的最优平移偏移（2D 归一化互相关）。

    Parameters
    ----------
    ref : 参考场（如 MODIS CTH），2D
    test : 待检测场（如 AGRI CTH），2D
    max_shift : 最大搜索偏移（像元，降采样前）
    downsample : 降采样因子（加速计算）
    min_overlap : 最少重叠像元数

    Returns
    -------
    dict with keys:
      status, row_shift, col_shift, peak_r, corr_map,
      rmse_before, rmse_after, bias_before, bias_after,
      r_before, r_after, n_pixels
    """
    valid = np.isfinite(ref) & np.isfinite(test)
    if valid.sum() < min_overlap:
        return {"status": "insufficient_data", "row_shift": 0, "col_shift": 0}

    # ── 降采样 + Z-score 标准化 ──
    def prepare(arr):
        a = arr.copy()
        a[~valid] = np.nan
        a_ds = a[::downsample, ::downsample]
        v_ds = valid[::downsample, ::downsample]
        mn = np.nanmean(a_ds[v_ds])
        std = np.nanstd(a_ds[v_ds])
        if std < 1e-6:
            return None, None
        a_norm = np.where(v_ds, (a_ds - mn) / std, 0.0)
        return a_norm, v_ds

    r_norm, r_valid = prepare(ref)
    t_norm, t_valid = prepare(test)
    if r_norm is None or t_norm is None:
        return {"status": "no_variance", "row_shift": 0, "col_shift": 0}

    # ── 2D 互相关 ──
    corr = correlate2d(r_norm, t_norm, mode="full", boundary="fill", fillvalue=0)
    norm = float(np.sqrt((r_norm ** 2).sum() * (t_norm ** 2).sum()))
    if norm > 0:
        corr /= norm

    # ── 搜索峰值（仅中心 ±max_shift 区域）──
    mid_r, mid_c = np.array(corr.shape) // 2
    max_ds = max(1, max_shift // downsample)
    r0 = max(0, mid_r - max_ds)
    r1 = min(corr.shape[0], mid_r + max_ds + 1)
    c0 = max(0, mid_c - max_ds)
    c1 = min(corr.shape[1], mid_c + max_ds + 1)
    sub = corr[r0:r1, c0:c1]

    peak_pos = np.unravel_index(np.argmax(sub), sub.shape)
    peak_r = float(sub[peak_pos])

    # 换算回原始像元偏移
    row_shift = int((peak_pos[0] - (r1 - r0) // 2) * downsample)
    col_shift = int((peak_pos[1] - (c1 - c0) // 2) * downsample)

    # ── 过滤虚假偏移：NCC < 0 或偏移到搜索边界 ──
    if peak_r < 0.05:
        # NCC 太低，无法可靠检测偏移
        row_shift, col_shift = 0, 0
    if abs(row_shift) >= max_shift * downsample or abs(col_shift) >= max_shift * downsample:
        # 偏移到搜索边界，可能是虚假峰值
        row_shift, col_shift = 0, 0

    # ── 计算偏移前后的指标 ──
    metrics_before = _compute_metrics(ref, test, valid)
    metrics_after = _compute_metrics_shifted(ref, test, valid, row_shift, col_shift)

    return {
        "status": "ok",
        "row_shift": row_shift,
        "col_shift": col_shift,
        "peak_r": peak_r,
        "corr_map": sub,
        "downsample": downsample,
        **metrics_before,
        **{f"{k}_after": v for k, v in metrics_after.items()},
    }


def _compute_metrics(
    ref: np.ndarray, test: np.ndarray, valid: np.ndarray
) -> Dict:
    """计算逐像元 RMSE / Bias / R。"""
    r = ref[valid]
    t = test[valid]
    n = len(r)
    if n < 10:
        return {"rmse_before": np.nan, "bias_before": np.nan,
                "r_before": np.nan, "n_pixels": 0}
    diff = t - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    if r.std() > 0 and t.std() > 0:
        r_val = float(np.corrcoef(r, t)[0, 1])
    else:
        r_val = 0.0
    return {"rmse_before": rmse, "bias_before": bias, "r_before": r_val, "n_pixels": n}


def _compute_metrics_shifted(
    ref: np.ndarray, test: np.ndarray, valid: np.ndarray,
    row_shift: int, col_shift: int,
) -> Dict:
    """将 test 场平移 (row_shift, col_shift) 后计算指标。
    返回 keys: rmse, bias, r（无后缀，由调用方添加 _after）。"""
    if row_shift == 0 and col_shift == 0:
        r = ref[valid]
        t = test[valid]
        n = len(r)
        if n < 10:
            return {"rmse": np.nan, "bias": np.nan, "r": np.nan}
        diff = t - r
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        r_val = float(np.corrcoef(r, t)[0, 1]) if r.std() > 0 and t.std() > 0 else 0.0
        return {"rmse": rmse, "bias": bias, "r": r_val}

    H, W = ref.shape
    # ref 区域
    r_r0 = max(0, row_shift)
    r_r1 = min(H, H + row_shift)
    r_c0 = max(0, col_shift)
    r_c1 = min(W, W + col_shift)
    # test 区域（反向偏移）
    t_r0 = max(0, -row_shift)
    t_r1 = min(H, H - row_shift)
    t_c0 = max(0, -col_shift)
    t_c1 = min(W, W - col_shift)

    overlap_h = min(r_r1 - r_r0, t_r1 - t_r0)
    overlap_w = min(r_c1 - r_c0, t_c1 - t_c0)
    if overlap_h <= 0 or overlap_w <= 0:
        return {"rmse": np.nan, "bias": np.nan, "r": np.nan}

    r_slice = ref[r_r0:r_r0 + overlap_h, r_c0:r_c0 + overlap_w]
    t_slice = test[t_r0:t_r0 + overlap_h, t_c0:t_c0 + overlap_w]
    v_slice = valid[r_r0:r_r0 + overlap_h, r_c0:r_c0 + overlap_w]

    # 计算指标（返回无后缀的 key）
    rv = r_slice[v_slice]
    tv = t_slice[v_slice]
    n = len(rv)
    if n < 10:
        return {"rmse": np.nan, "bias": np.nan, "r": np.nan}
    diff = tv - rv
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    r_val = float(np.corrcoef(rv, tv)[0, 1]) if rv.std() > 0 and tv.std() > 0 else 0.0
    return {"rmse": rmse, "bias": bias, "r": r_val}


# ═════════════════════════════════════════════════════════════════════════════
# 高级：分区域偏移检测（判断偏移是否空间均匀）
# ═════════════════════════════════════════════════════════════════════════════

def compute_regional_offsets(
    ref: np.ndarray,
    test: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    n_regions: int = 4,
    max_shift: int = 10,
    downsample: int = 2,
) -> List[Dict]:
    """
    将 AGRI 盘面分成 n_regions x n_regions 子区域，分别检测偏移。
    用于判断偏移是否空间均匀（系统性）还是随位置变化。
    """
    H, W = ref.shape
    rh, rw = H // n_regions, W // n_regions
    results = []

    for i in range(n_regions):
        for j in range(n_regions):
            r0, r1 = i * rh, (i + 1) * rh
            c0, c1 = j * rw, (j + 1) * rw
            sub_ref = ref[r0:r1, c0:c1]
            sub_test = test[r0:r1, c0:c1]
            sub_lat = lat[r0:r1, c0:c1]
            sub_lon = lon[r0:r1, c0:c1]

            valid = np.isfinite(sub_ref) & np.isfinite(sub_test)
            if valid.sum() < 50:
                continue

            res = compute_offset_ncc(sub_ref, sub_test,
                                     max_shift=max_shift,
                                     downsample=downsample,
                                     min_overlap=50)
            if res["status"] != "ok":
                continue

            results.append({
                "region": (i, j),
                "center_lat": float(np.nanmean(sub_lat)),
                "center_lon": float(np.nanmean(sub_lon)),
                "row_shift": res["row_shift"],
                "col_shift": res["col_shift"],
                "peak_r": res["peak_r"],
            })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MODIS → AGRI 网格投影（复用 test_visualize.py 逻辑）
# ═════════════════════════════════════════════════════════════════════════════

def project_modis_to_agri_grid(
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
    modis_data: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    search_radius_km: float = 5.0,
) -> np.ndarray:
    """KD-tree 最近邻投影 MODIS → AGRI 网格。"""
    H_a, W_a = agri_lat.shape

    def xyz(lat, lon):
        lr, lo = np.deg2rad(lat), np.deg2rad(lon)
        return np.column_stack([
            np.cos(lr) * np.cos(lo),
            np.cos(lr) * np.sin(lo),
            np.sin(lr),
        ])

    chord = 2.0 * np.sin(search_radius_km / (2.0 * 6371.0))

    m_lat_f = modis_lat.ravel()
    m_lon_f = modis_lon.ravel()
    m_dat_f = modis_data.ravel()
    valid_m = np.isfinite(m_lat_f) & np.isfinite(m_lon_f) & np.isfinite(m_dat_f)
    if not valid_m.any():
        return np.full((H_a, W_a), np.nan, np.float32)

    m_xyz = xyz(m_lat_f[valid_m], m_lon_f[valid_m])
    m_val = m_dat_f[valid_m]

    # 预过滤 AGRI 像元：只查询 MODIS 覆盖范围附近的像元（加速 10-100x）
    m_lat_valid = m_lat_f[valid_m]
    m_lon_valid = m_lon_f[valid_m]
    lat_margin = search_radius_km / 111.0 + 0.5  # km → degree + 余量
    lon_margin = lat_margin + 0.5
    lat_min, lat_max = m_lat_valid.min() - lat_margin, m_lat_valid.max() + lat_margin
    lon_min, lon_max = m_lon_valid.min() - lon_margin, m_lon_valid.max() + lon_margin

    a_lat_f = agri_lat.ravel()
    a_lon_f = agri_lon.ravel()
    valid_a = (
        np.isfinite(a_lat_f) & np.isfinite(a_lon_f) &
        (a_lat_f >= lat_min) & (a_lat_f <= lat_max) &
        (a_lon_f >= lon_min) & (a_lon_f <= lon_max)
    )

    if not valid_a.any():
        return np.full((H_a, W_a), np.nan, np.float32)

    tree = cKDTree(m_xyz)
    a_xyz = xyz(a_lat_f[valid_a], a_lon_f[valid_a])
    dist, idx = tree.query(a_xyz, k=1, distance_upper_bound=chord, workers=-1)

    out = np.full(H_a * W_a, np.nan, np.float32)
    found = idx < len(m_val)
    out_idx = np.where(valid_a)[0]
    out[out_idx[found]] = m_val[idx[found]]
    return out.reshape(H_a, W_a)


# ═════════════════════════════════════════════════════════════════════════════
# 文件发现（复用 test_visualize.py 逻辑）
# ═════════════════════════════════════════════════════════════════════════════

def _find_agri_scenes(day: str):
    """返回 [(fdi_path, timestamp_str, datetime), ...]"""
    import glob
    day_dir = cfg.AGRI_ROOT / day
    if not day_dir.exists():
        return []
    patterns = [
        str(day_dir / "**/AGRI_L1_FDI*.HDF"),
        str(day_dir / "**/FY4A-_AGRI*_FDI-_*.HDF"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # 排除 GEO 文件，只保留 FDI
    files = [f for f in files if "_FDI-_" in Path(f).name]
    scenes = []
    for f in sorted(files):
        fname = Path(f).name
        # 提取时间戳
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d_%H%M%S"):
            try:
                idx = fname.find(day)
                if idx >= 0:
                    ts_str = fname[idx:idx + 14]
                    dt = datetime.strptime(ts_str, fmt)
                    scenes.append((Path(f), ts_str, dt))
                    break
            except (ValueError, IndexError):
                continue
    scenes.sort(key=lambda x: x[2])
    return scenes


def _find_modis_list(day: str):
    """返回 [(path, datetime), ...]"""
    import glob
    from datetime import timedelta as td
    day_dir = cfg.MODIS_ROOT / day
    if not day_dir.exists():
        day_dir = cfg.MODIS_ROOT
    # MODIS 文件使用儒略日格式：MYD06_L2.A2019005.0000.061...
    dt = datetime.strptime(day, "%Y%m%d")
    jday = dt.timetuple().tm_yday
    julian_str = f"A{dt.year}{jday:03d}"
    files = sorted(glob.glob(str(day_dir / f"*{julian_str}*.hdf")))
    if not files:
        files = sorted(glob.glob(str(day_dir / f"*{day}*.hdf")))
    if not files:
        files = sorted(glob.glob(str(day_dir / f"*{day}*")))
    result = []
    for f in files:
        fname = Path(f).name
        # MODIS 文件名格式：MYD06_L2.A2019125.0410.061...
        try:
            parts = fname.split(".")
            jday_str = parts[1]  # A2019125
            time_str = parts[2]  # 0410
            year = int(jday_str[1:5])
            jday = int(jday_str[5:8])
            from datetime import timedelta as td
            dt = datetime(year, 1, 1) + td(days=jday - 1)
            hour, minute = int(time_str[:2]), int(time_str[2:4])
            dt = dt.replace(hour=hour, minute=minute)
            result.append((Path(f), dt))
        except (IndexError, ValueError):
            continue
    result.sort(key=lambda x: x[1])
    return result


def _find_l2_cth(day: str, ts: str) -> Optional[Path]:
    """查找 AGRI L2 CTH 文件。"""
    import glob
    # 优先从 FY4A_L2_ROOT/CTH 查找
    l2_dir = cfg.FY4A_L2_ROOT / "CTH" / day
    if l2_dir.exists():
        for f in sorted(l2_dir.glob("*_CTH-_*.NC")):
            if ts in f.name:
                return f
        # 如果没有精确匹配，返回最接近的
        files = sorted(l2_dir.glob("*_CTH-_*.NC"))
        if files:
            return files[0]
    # 备选：从 AGRI_ROOT 查找
    day_dir = cfg.AGRI_ROOT / day
    patterns = [
        str(day_dir / f"**/AGRI_L2_CTH*{ts}*.NC"),
        str(day_dir / f"**/AGRI_L2_CTH*{ts}*.nc"),
        str(day_dir / f"**/*CTH*{ts}*"),
    ]
    for p in patterns:
        files = glob.glob(p, recursive=True)
        if files:
            return Path(sorted(files)[0])
    return None


def _find_closest_modis(agri_dt, modis_files, max_dt_min=15.0):
    """找时间最近的 MODIS granule。"""
    best = None
    best_dt = float("inf")
    for mf, mdt in modis_files:
        dt = abs((mdt - agri_dt).total_seconds()) / 60.0
        if dt < best_dt:
            best_dt = dt
            best = (mf, mdt, dt)
    if best and best[2] <= max_dt_min:
        return best
    return (None, None, best_dt) if best else (None, None, float("inf"))


# ═════════════════════════════════════════════════════════════════════════════
# MODIS CTH 读取
# ═════════════════════════════════════════════════════════════════════════════

def read_myd06_cth(modis_file: Path) -> Optional[Dict]:
    """读取 MYD06 CTH + lat/lon（5km）。"""
    try:
        from pyhdf.SD import SD, SDC
        sd = SD(str(modis_file), SDC.READ)

        lat = sd.select("Latitude")[:].astype(np.float32)
        lon = sd.select("Longitude")[:].astype(np.float32)

        ds = sd.select("Cloud_Top_Height")
        raw = ds[:].astype(np.float32)
        attr = ds.attributes()
        fv = attr.get("_FillValue", -9999)
        raw[raw == fv] = np.nan
        sf = float(attr.get("scale_factor", 1.0))
        ao = float(attr.get("add_offset", 0.0))
        cth = raw * sf + ao

        sd.end()
        return {"lat": lat, "lon": lon, "CTH": cth}
    except Exception as e:
        log.warning("Cannot read MYD06 %s: %s", modis_file, e)
        return None


def read_agri_geo(geo_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 AGRI GEO lat/lon。"""
    from test_visualize import _read_agri_geo_raw
    return _read_agri_geo_raw(geo_file)


def read_agri_cth(nc_file: Path) -> Optional[np.ndarray]:
    """读取 AGRI L2 CTH。"""
    from test_visualize import _read_agri_cth_raw
    return _read_agri_cth_raw(nc_file)


# ═════════════════════════════════════════════════════════════════════════════
# 单场景诊断
# ═════════════════════════════════════════════════════════════════════════════

def diagnose_single_scene(
    agri_cth: np.ndarray,
    modis_cth_on_grid: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    max_shift: int = 15,
    downsample: int = 2,
    pixel_size_km: float = 4.0,
) -> Dict:
    """单场景偏移诊断，返回完整结果字典。"""
    # ── 全场互相关 ──
    result = compute_offset_ncc(
        modis_cth_on_grid, agri_cth,
        max_shift=max_shift, downsample=downsample,
    )

    # ── 分区域偏移检测 ──
    regional = []
    if result["status"] == "ok":
        regional = compute_regional_offsets(
            modis_cth_on_grid, agri_cth, agri_lat, agri_lon,
            n_regions=3, max_shift=max_shift, downsample=downsample,
        )

    result["regional"] = regional
    result["pixel_size_km"] = pixel_size_km

    # ── 偏移距离 ──
    if result["status"] == "ok":
        rs, cs = result["row_shift"], result["col_shift"]
        result["offset_km"] = float(np.sqrt(rs ** 2 + cs ** 2) * pixel_size_km)
        result["offset_azimuth_deg"] = float(np.rad2deg(np.arctan2(cs, rs)))

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 可视化
# ═════════════════════════════════════════════════════════════════════════════

def plot_single_scene(
    agri_cth: np.ndarray,
    modis_cth_on_grid: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    result: Dict,
    out_path: Path,
    scene_id: str = "",
):
    """单场景诊断图：4 面板。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── 1) AGRI CTH ──
    ax = axes[0, 0]
    v = np.isfinite(agri_cth) & np.isfinite(agri_lat)
    if v.any():
        im = ax.imshow(np.where(v, agri_cth, np.nan),
                       origin="upper", cmap="RdYlBu_r",
                       vmin=0, vmax=15000, aspect="auto")
        plt.colorbar(im, ax=ax, label="CTH (m)", fraction=0.04)
    ax.set_title("AGRI CTH (test field)", fontsize=9)

    # ── 2) MODIS CTH on AGRI grid ──
    ax = axes[0, 1]
    v2 = np.isfinite(modis_cth_on_grid)
    if v2.any():
        im = ax.imshow(np.where(v2, modis_cth_on_grid, np.nan),
                       origin="upper", cmap="RdYlBu_r",
                       vmin=0, vmax=15000, aspect="auto")
        plt.colorbar(im, ax=ax, label="CTH (m)", fraction=0.04)
    ax.set_title("MODIS CTH → AGRI grid (reference)", fontsize=9)

    # ── 3) 互相关热图 ──
    ax = axes[1, 0]
    if result.get("status") == "ok" and "corr_map" in result:
        cm = result["corr_map"]
        ds = result.get("downsample", 1)
        row_s = result["row_shift"]
        col_s = result["col_shift"]
        peak_r = result["peak_r"]
        n_r, n_c = cm.shape
        row_ticks = np.linspace(-(n_r // 2) * ds, (n_r // 2) * ds, min(5, n_r))
        col_ticks = np.linspace(-(n_c // 2) * ds, (n_c // 2) * ds, min(5, n_c))
        extent = [col_ticks[0], col_ticks[-1], row_ticks[-1], row_ticks[0]]
        im = ax.imshow(cm, cmap="hot", origin="upper", aspect="auto",
                       extent=extent, vmin=0, vmax=cm.max())
        plt.colorbar(im, ax=ax, label="NCC", fraction=0.04)
        ax.axhline(0, color="cyan", lw=0.8, ls="--")
        ax.axvline(0, color="cyan", lw=0.8, ls="--")
        ax.plot(col_s, row_s, "r+", ms=12, mew=2,
                label=f"peak ({col_s:+d},{row_s:+d}) px")
        ax.set_xlabel("Column shift (AGRI pixels)")
        ax.set_ylabel("Row shift (AGRI pixels)")
        dist_km = result.get("offset_km", 0)
        ax.set_title(
            f"2D Cross-Correlation\n"
            f"shift: row={row_s:+d} col={col_s:+d} px  ≈ {dist_km:.1f} km\n"
            f"peak NCC={peak_r:.3f}  "
            f"{'OFFSET DETECTED' if (abs(row_s) > 1 or abs(col_s) > 1) else 'No significant offset'}",
            fontsize=8,
            color="red" if (abs(row_s) > 1 or abs(col_s) > 1) else "green",
        )
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, f"Cross-correlation failed: {result.get('status', 'unknown')}",
                transform=ax.transAxes, ha="center", fontsize=9)

    # ── 4) 指标对比 ──
    ax = axes[1, 1]
    ax.axis("off")
    if result.get("status") == "ok":
        lines = [
            "=== Geolocation Offset Diagnostic ===",
            "",
            f"Optimal shift: row={result['row_shift']:+d}  col={result['col_shift']:+d} pixels",
            f"Offset distance: {result.get('offset_km', 0):.1f} km",
            f"Offset azimuth: {result.get('offset_azimuth_deg', 0):.1f}°",
            f"Peak NCC: {result['peak_r']:.4f}",
            "",
            "--- Before correction ---",
            f"  RMSE = {result['rmse_before']:.1f} m",
            f"  Bias = {result['bias_before']:.1f} m",
            f"  R    = {result['r_before']:.4f}",
            f"  N    = {result['n_pixels']}",
            "",
            "--- After correction ---",
            f"  RMSE = {result['rmse_after']:.1f} m",
            f"  Bias = {result['bias_after']:.1f} m",
            f"  R    = {result['r_after']:.4f}",
            "",
        ]
        # 改善幅度
        if np.isfinite(result["rmse_before"]) and np.isfinite(result["rmse_after"]):
            rmse_imp = result["rmse_before"] - result["rmse_after"]
            rmse_pct = rmse_imp / result["rmse_before"] * 100 if result["rmse_before"] > 0 else 0
            lines.append(f"RMSE improvement: {rmse_imp:+.1f} m ({rmse_pct:+.1f}%)")
        if np.isfinite(result["r_before"]) and np.isfinite(result["r_after"]):
            r_imp = result["r_after"] - result["r_before"]
            lines.append(f"R improvement: {r_imp:+.4f}")

        # 分区域偏移
        regional = result.get("regional", [])
        if regional:
            lines.extend(["", "--- Regional offsets ---"])
            shifts = [(r["row_shift"], r["col_shift"]) for r in regional]
            rows = [s[0] for s in shifts]
            cols = [s[1] for s in shifts]
            lines.append(f"  Row shift: {min(rows):+d} to {max(rows):+d}  "
                         f"(std={np.std(rows):.1f})")
            lines.append(f"  Col shift: {min(cols):+d} to {max(cols):+d}  "
                         f"(std={np.std(cols):.1f})")
            if np.std(rows) < 1.5 and np.std(cols) < 1.5:
                lines.append("  → Offset is SPATIALLY UNIFORM (systematic)")
            else:
                lines.append("  → Offset VARIES across scene (not purely systematic)")

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"Geolocation Offset Diagnostic — {scene_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Single-scene plot saved: %s", out_path)


def plot_multi_scene(
    scene_results: List[Dict],
    out_path: Path,
):
    """多场景聚合诊断图。"""
    valid_results = [r for r in scene_results if r.get("status") == "ok"]
    if not valid_results:
        log.warning("No valid scenes for multi-scene plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    row_shifts = [r["row_shift"] for r in valid_results]
    col_shifts = [r["col_shift"] for r in valid_results]
    peak_rs = [r["peak_r"] for r in valid_results]
    offsets_km = [r.get("offset_km", 0) for r in valid_results]

    # ── 1) 偏移矢量图 ──
    ax = axes[0, 0]
    ax.scatter(col_shifts, row_shifts, c=peak_rs, cmap="RdYlGn",
               s=60, edgecolors="k", linewidths=0.5, vmin=0.3, vmax=1.0)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    mean_row = np.mean(row_shifts)
    mean_col = np.mean(col_shifts)
    ax.plot(mean_col, mean_row, "r*", ms=15, label=f"Mean ({mean_col:+.1f},{mean_row:+.1f})")
    ax.set_xlabel("Column shift (pixels)")
    ax.set_ylabel("Row shift (pixels)")
    ax.set_title("Offset vectors (color = peak NCC)")
    ax.legend(fontsize=8)
    plt.colorbar(ax.collections[0], ax=ax, label="Peak NCC", fraction=0.04)

    # ── 2) 偏移方向直方图 ──
    ax = axes[0, 1]
    azimuths = [r.get("offset_azimuth_deg", 0) for r in valid_results]
    ax.hist(azimuths, bins=36, range=(-180, 180), color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(azimuths), color="red", ls="--", lw=1.5,
               label=f"Mean azimuth: {np.mean(azimuths):.1f}°")
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Offset direction distribution")
    ax.legend(fontsize=8)

    # ── 3) 偏移幅度分布 ──
    ax = axes[1, 0]
    ax.hist(offsets_km, bins=20, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(offsets_km), color="red", ls="--", lw=1.5,
               label=f"Mean: {np.mean(offsets_km):.1f} km")
    ax.set_xlabel("Offset distance (km)")
    ax.set_ylabel("Count")
    ax.set_title("Offset magnitude distribution")
    ax.legend(fontsize=8)

    # ── 4) 汇总统计 ──
    ax = axes[1, 1]
    ax.axis("off")
    rmse_b = [r["rmse_before"] for r in valid_results if np.isfinite(r.get("rmse_before", np.nan))]
    rmse_a = [r["rmse_after"] for r in valid_results if np.isfinite(r.get("rmse_after", np.nan))]
    r_b = [r["r_before"] for r in valid_results if np.isfinite(r.get("r_before", np.nan))]
    r_a = [r["r_after"] for r in valid_results if np.isfinite(r.get("r_after", np.nan))]

    lines = [
        "=== Multi-Scene Offset Summary ===",
        "",
        f"Scenes analyzed: {len(valid_results)}",
        "",
        "--- Systematic Offset ---",
        f"Mean row shift: {mean_row:+.2f} ± {np.std(row_shifts):.2f} pixels",
        f"Mean col shift: {mean_col:+.2f} ± {np.std(col_shifts):.2f} pixels",
        f"Mean offset: {np.mean(offsets_km):.1f} ± {np.std(offsets_km):.1f} km",
        f"Mean azimuth: {np.mean(azimuths):.1f}° ± {np.std(azimuths):.1f}°",
        "",
    ]

    # 判断系统性偏移
    if np.std(row_shifts) < 2 and np.std(col_shifts) < 2:
        lines.append("CONCLUSION: Systematic offset detected")
        lines.append(f"  Direction: col={mean_col:+.1f} px, row={mean_row:+.1f} px")
        if abs(mean_col) > abs(mean_row):
            direction = "EAST" if mean_col > 0 else "WEST"
        else:
            direction = "SOUTH" if mean_row > 0 else "NORTH"
        lines.append(f"  Primary direction: {direction}")
    else:
        lines.append("CONCLUSION: Offset is variable (not systematic)")

    lines.extend(["", "--- Metrics Before vs After ---"])
    if rmse_b and rmse_a:
        lines.append(f"RMSE: {np.mean(rmse_b):.1f} → {np.mean(rmse_a):.1f} m  "
                     f"({(np.mean(rmse_b) - np.mean(rmse_a)) / np.mean(rmse_b) * 100:+.1f}%)")
    if r_b and r_a:
        lines.append(f"R:    {np.mean(r_b):.4f} → {np.mean(r_a):.4f}  "
                     f"({np.mean(r_a) - np.mean(r_b):+.4f})")

    # 偏移来源分析
    lines.extend(["", "--- Source Attribution ---"])
    if np.mean(offsets_km) > 10:
        lines.append("Offset > 10 km: likely AGRI GEO center error")
        lines.append("  → Check _derive_latlon() center calculation")
    elif np.mean(offsets_km) > 5:
        lines.append("Offset 5-10 km: likely 5km→1km upsampling or parallax")
        lines.append("  → Check upsample_5km_to_1km_coords()")
        lines.append("  → Check parallax correction for high clouds")
    elif np.mean(offsets_km) > 2:
        lines.append("Offset 2-5 km: likely sub-pixel registration error")
        lines.append("  → Check MODIS geolocation accuracy")
        lines.append("  → Check AGRI pixel geolocation")
    else:
        lines.append("Offset < 2 km: within expected noise")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Multi-Scene Geolocation Offset Analysis", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Multi-scene plot saved: %s", out_path)


def plot_before_after_scatter(
    ref: np.ndarray,
    test: np.ndarray,
    test_shifted: np.ndarray,
    valid: np.ndarray,
    result: Dict,
    out_path: Path,
    scene_id: str = "",
):
    """偏移校正前后散点对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (data, title, metrics) in enumerate([
        (test, "Before correction", {
            "RMSE": result.get("rmse_before", np.nan),
            "Bias": result.get("bias_before", np.nan),
            "R": result.get("r_before", np.nan),
        }),
        (test_shifted, "After correction", {
            "RMSE": result.get("rmse_after", np.nan),
            "Bias": result.get("bias_after", np.nan),
            "R": result.get("r_after", np.nan),
        }),
    ]):
        ax = axes[idx]
        both = valid & np.isfinite(data)
        if both.sum() < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        x = ref[both]
        y = data[both]
        n = len(x)
        sub = np.random.choice(n, min(n, 10000), replace=False)
        ax.scatter(x[sub], y[sub], s=1, alpha=0.2, color="steelblue", rasterized=True)
        lim = max(float(np.nanpercentile(np.concatenate([x, y]), 99)), 1000)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="1:1")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("MODIS CTH (m)")
        ax.set_ylabel("AGRI CTH (m)")
        ax.set_title(f"{title}\nRMSE={metrics['RMSE']:.0f}m  "
                     f"Bias={metrics['Bias']:.0f}m  R={metrics['R']:.3f}",
                     fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"Before/After Offset Correction — {scene_id}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Before/after scatter saved: %s", out_path)


# ═════════════════════════════════════════════════════════════════════════════
# 报告生成
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(
    scene_results: List[Dict],
    out_path: Path,
):
    """生成文本汇总报告。"""
    valid = [r for r in scene_results if r.get("status") == "ok"]

    lines = [
        "=" * 70,
        "AGRI–MODIS Geolocation Offset Diagnostic Report",
        "=" * 70,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Scenes analyzed: {len(scene_results)} total, {len(valid)} valid",
        "",
    ]

    if not valid:
        lines.append("No valid scenes found.")
        out_path.write_text("\n".join(lines))
        return

    row_shifts = [r["row_shift"] for r in valid]
    col_shifts = [r["col_shift"] for r in valid]
    offsets_km = [r.get("offset_km", 0) for r in valid]
    peak_rs = [r["peak_r"] for r in valid]

    mean_row = np.mean(row_shifts)
    mean_col = np.mean(col_shifts)
    mean_km = np.mean(offsets_km)

    lines.extend([
        "--- Systematic Offset ---",
        f"Mean row shift: {mean_row:+.2f} ± {np.std(row_shifts):.2f} pixels",
        f"Mean col shift: {mean_col:+.2f} ± {np.std(col_shifts):.2f} pixels",
        f"Mean offset distance: {mean_km:.1f} ± {np.std(offsets_km):.1f} km",
        f"Mean peak NCC: {np.mean(peak_rs):.4f}",
        "",
    ])

    # 判断偏移方向
    if abs(mean_col) > abs(mean_row):
        direction = "EAST" if mean_col > 0 else "WEST"
    else:
        direction = "SOUTH" if mean_row > 0 else "NORTH"
    lines.append(f"Primary offset direction: {direction}")

    # 判断系统性
    if np.std(row_shifts) < 2 and np.std(col_shifts) < 2:
        lines.append("Offset is SYSTEMATIC (consistent across scenes)")
    else:
        lines.append("Offset is VARIABLE (not consistent across scenes)")
    lines.append("")

    # 偏移来源分析
    lines.extend([
        "--- Source Attribution ---",
    ])
    if mean_km > 10:
        lines.extend([
            "Likely source: AGRI GEO center calculation error",
            "  The _derive_latlon() function uses Begin/End Pixel/Line Number",
            "  from metadata to compute the image center. If these attributes",
            "  are incorrect or missing, the entire geolocation shifts by",
            "  tens of km.",
            "  → Verify AGRI GEO HDF5 metadata attributes.",
        ])
    elif mean_km > 5:
        lines.extend([
            "Likely source: MODIS 5km→1km coordinate upsampling or parallax",
            "  The upsample_5km_to_1km_coords() uses nearest-neighbor repeat,",
            "  which can introduce ~2.5 km error at block edges. Additionally,",
            "  high clouds viewed from geostationary orbit have parallax",
            "  displacement that is not corrected.",
            "  → Check if MYD03 1km geolocation is available.",
            "  → Consider parallax correction for CTH > 6 km.",
        ])
    elif mean_km > 2:
        lines.extend([
            "Likely source: Sub-pixel registration error",
            "  Both AGRI and MODIS have inherent geolocation accuracy of",
            "  ~1-2 pixels. The KD-tree search radius (2.5 km) may be too",
            "  tight to absorb this.",
            "  → Consider increasing FUSION_AGRI_SEARCH_RADIUS_KM.",
        ])
    else:
        lines.extend([
            "Offset is within expected noise (< 2 km)",
            "  This is consistent with normal sensor geolocation accuracy.",
        ])
    lines.append("")

    # 指标对比
    rmse_b = [r["rmse_before"] for r in valid if np.isfinite(r.get("rmse_before", np.nan))]
    rmse_a = [r["rmse_after"] for r in valid if np.isfinite(r.get("rmse_after", np.nan))]
    r_b = [r["r_before"] for r in valid if np.isfinite(r.get("r_before", np.nan))]
    r_a = [r["r_after"] for r in valid if np.isfinite(r.get("r_after", np.nan))]

    lines.extend(["--- Validation Metrics (Before vs After) ---"])
    if rmse_b:
        lines.append(f"RMSE: {np.mean(rmse_b):.1f} m", )
    if rmse_a:
        improvement = (np.mean(rmse_b) - np.mean(rmse_a)) / np.mean(rmse_b) * 100 if rmse_b else 0
        lines.append(f"  → {np.mean(rmse_a):.1f} m after correction ({improvement:+.1f}%)")
    if r_b:
        lines.append(f"R: {np.mean(r_b):.4f}")
    if r_a:
        lines.append(f"  → {np.mean(r_a):.4f} after correction ({np.mean(r_a) - np.mean(r_b):+.4f})")
    lines.append("")

    # 逐场景详情
    lines.extend(["--- Per-Scene Results ---"])
    for i, r in enumerate(scene_results):
        sid = r.get("scene_id", f"scene_{i}")
        if r.get("status") != "ok":
            lines.append(f"  {sid}: {r.get('status', 'unknown')}")
            continue
        lines.append(f"  {sid}: shift=({r['row_shift']:+d},{r['col_shift']:+d}) "
                     f"NCC={r['peak_r']:.3f} "
                     f"RMSE_before={r.get('rmse_before', np.nan):.0f} "
                     f"RMSE_after={r.get('rmse_after', np.nan):.0f}")

    out_path.write_text("\n".join(lines))
    log.info("Report saved: %s", out_path)


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

def run_single_day(
    day: str,
    out_dir: Path,
    max_shift: int = 15,
    downsample: int = 2,
    max_dt_min: float = 15.0,
    scene_index: int = 0,
) -> Optional[Dict]:
    """单天诊断。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 发现文件
    agri_scenes = _find_agri_scenes(day)
    modis_files = _find_modis_list(day)

    if not agri_scenes:
        log.error("No AGRI scenes for day %s", day)
        return None
    if not modis_files:
        log.error("No MODIS files for day %s", day)
        return None

    # 选择场景
    if scene_index >= len(agri_scenes):
        log.error("Scene index %d out of range (0-%d)", scene_index, len(agri_scenes) - 1)
        return None
    agri_fdi, ts, agri_dt = agri_scenes[scene_index]

    # 找 L2 CTH
    agri_cth_nc = _find_l2_cth(day, ts)
    if agri_cth_nc is None:
        log.warning("No L2 CTH for %s, skipping", ts)
        return None

    # 找最近 MODIS
    modis_file, modis_dt, dt_min = _find_closest_modis(agri_dt, modis_files, max_dt_min)
    if modis_file is None:
        log.warning("No MODIS within %.0f min for %s (closest=%.1f min)", max_dt_min, ts, dt_min)
        return None

    log.info("Scene: %s  MODIS: %s  Δt=%.1f min", ts, modis_file.name, dt_min)

    # 读取数据
    geo_file = Path(str(agri_fdi).replace("_FDI-_", "_GEO-_"))
    if not geo_file.exists():
        # 尝试其他命名模式
        for pattern in ["_FDI_", "_FDI-", "FDI"]:
            alt = Path(str(agri_fdi).replace(pattern, "GEO"))
            if alt.exists():
                geo_file = alt
                break
    try:
        agri_lat, agri_lon = read_agri_geo(geo_file)
    except Exception as e:
        log.error("Cannot read AGRI GEO %s: %s", geo_file, e)
        return None

    agri_cth = read_agri_cth(agri_cth_nc)
    if agri_cth is None:
        return None

    modis_data = read_myd06_cth(modis_file)
    if modis_data is None:
        return None

    # ── 找 AGRI/MODIS 重叠区域（大幅加速投影）──
    m_lat = modis_data["lat"]
    m_lon = modis_data["lon"]
    m_cth = modis_data["CTH"]
    m_valid = np.isfinite(m_lat) & np.isfinite(m_lon) & np.isfinite(m_cth)
    if not m_valid.any():
        log.warning("No valid MODIS CTH pixels")
        return None

    margin = 1.0  # degree margin
    lat_min, lat_max = float(m_lat[m_valid].min()) - margin, float(m_lat[m_valid].max()) + margin
    lon_min, lon_max = float(m_lon[m_valid].min()) - margin, float(m_lon[m_valid].max()) + margin

    a_valid = (
        np.isfinite(agri_lat) & np.isfinite(agri_lon) &
        (agri_lat >= lat_min) & (agri_lat <= lat_max) &
        (agri_lon >= lon_min) & (agri_lon <= lon_max)
    )
    if not a_valid.any():
        log.warning("No AGRI pixels overlap with MODIS swath")
        return None

    rows, cols = np.where(a_valid)
    r0, r1 = max(0, rows.min() - 10), min(agri_lat.shape[0], rows.max() + 11)
    c0, c1 = max(0, cols.min() - 10), min(agri_lat.shape[1], cols.max() + 11)

    log.info("Overlap region: rows [%d:%d], cols [%d:%d] (%dx%d)",
             r0, r1, c0, c1, r1 - r0, c1 - c0)

    agri_lat_sub = agri_lat[r0:r1, c0:c1]
    agri_lon_sub = agri_lon[r0:r1, c0:c1]
    agri_cth_sub = agri_cth[r0:r1, c0:c1]

    # 投影 MODIS → AGRI 子网格
    modis_cth_on_grid = project_modis_to_agri_grid(
        m_lat, m_lon, m_cth,
        agri_lat_sub, agri_lon_sub,
    )

    # 诊断
    result = diagnose_single_scene(
        agri_cth_sub, modis_cth_on_grid, agri_lat_sub, agri_lon_sub,
        max_shift=max_shift, downsample=downsample,
    )
    result["scene_id"] = f"{day}_{ts}"

    # 可视化
    plot_single_scene(
        agri_cth_sub, modis_cth_on_grid, agri_lat_sub, agri_lon_sub, result,
        out_dir / f"geoloc_offset_{day}_{ts}.png",
        scene_id=f"{day}_{ts}",
    )

    # 偏移校正后散点图（仅当检测到非零偏移时）
    if result["status"] == "ok" and (result["row_shift"] != 0 or result["col_shift"] != 0):
        shifted = _shift_field(agri_cth_sub, result["row_shift"], result["col_shift"])
        valid = np.isfinite(modis_cth_on_grid) & np.isfinite(agri_cth_sub)
        plot_before_after_scatter(
            modis_cth_on_grid, agri_cth_sub, shifted, valid, result,
            out_dir / f"geoloc_scatter_{day}_{ts}.png",
            scene_id=f"{day}_{ts}",
        )

    return result


def _shift_field(arr: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    """将 2D 数组平移 (row_shift, col_shift) 像元，空位填 NaN。"""
    H, W = arr.shape
    out = np.full_like(arr, np.nan)
    # 源区域
    sr0 = max(0, -row_shift)
    sr1 = min(H, H - row_shift)
    sc0 = max(0, -col_shift)
    sc1 = min(W, W - col_shift)
    # 目标区域
    dr0 = max(0, row_shift)
    dr1 = min(H, H + row_shift)
    dc0 = max(0, col_shift)
    dc1 = min(W, W + col_shift)
    # 复制
    h = min(sr1 - sr0, dr1 - dr0)
    w = min(sc1 - sc0, dc1 - dc0)
    if h > 0 and w > 0:
        out[dr0:dr0 + h, dc0:dc0 + w] = arr[sr0:sr0 + h, sc0:sc0 + w]
    return out


def run_multi_day(
    days: List[str],
    out_dir: Path,
    max_shift: int = 15,
    downsample: int = 2,
    max_dt_min: float = 15.0,
) -> List[Dict]:
    """多天批量诊断。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for day in days:
        log.info("=== Processing day %s ===", day)
        # 对每天尝试多个场景
        scenes = _find_agri_scenes(day)
        for si in range(min(len(scenes), 6)):  # 每天最多 6 个场景
            result = run_single_day(day, out_dir, max_shift, downsample,
                                    max_dt_min, scene_index=si)
            if result:
                all_results.append(result)

    # 聚合分析
    if len(all_results) > 1:
        plot_multi_scene(all_results, out_dir / "geoloc_offset_multi_scene.png")
        generate_report(all_results, out_dir / "geoloc_offset_report.txt")
    elif all_results:
        generate_report(all_results, out_dir / "geoloc_offset_report.txt")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="AGRI–MODIS 地理位置系统性偏移诊断")
    parser.add_argument("--day", type=str, default=None,
                        help="单天诊断 (YYYYMMDD)")
    parser.add_argument("--days", nargs="+", default=None,
                        help="多天批量诊断 (YYYYMMDD YYYYMMDD ...)")
    parser.add_argument("--out_dir", type=str, default="diag_geoloc_offset",
                        help="输出目录")
    parser.add_argument("--max_shift", type=int, default=15,
                        help="最大搜索偏移（像元数）")
    parser.add_argument("--downsample", type=int, default=2,
                        help="降采样因子（加速计算）")
    parser.add_argument("--max_dt_min", type=float, default=15.0,
                        help="AGRI-MODIS 最大允许时间差（分钟）")
    parser.add_argument("--scene_index", type=int, default=0,
                        help="单天模式下使用第 N 个场景（0-indexed）")
    args = parser.parse_args()

    if args.day:
        result = run_single_day(
            args.day, Path(args.out_dir),
            args.max_shift, args.downsample,
            args.max_dt_min, args.scene_index,
        )
        if result:
            generate_report([result], Path(args.out_dir) / "geoloc_offset_report.txt")
            log.info("Done. Report: %s", Path(args.out_dir) / "geoloc_offset_report.txt")
        else:
            log.error("No valid result for day %s", args.day)
            sys.exit(1)
    elif args.days:
        results = run_multi_day(
            args.days, Path(args.out_dir),
            args.max_shift, args.downsample, args.max_dt_min,
        )
        log.info("Done. %d scenes processed.", len(results))
    else:
        parser.error("Please specify --day or --days")


if __name__ == "__main__":
    main()
