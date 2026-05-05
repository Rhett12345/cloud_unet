"""
fusion_core.py
==============
MYD06 → AGRI 空间匹配引擎（单像元最近邻）。

本模块与 IO 解耦，只负责：
  1. 对每个 MODIS 条带，用 KD-tree 将 1km 像元匹配到 AGRI 4km 网格
  2. 跨文件选时间差最小的候选，直接取用其值（不聚合）
  3. 输出质量字段（时间差、距离、样本权重等）

设计原则
--------
- MODIS → AGRI 方向（绝不反向）
- k=1 最近邻，2.5 km 搜索半径
- 时间优先：多文件候选时选时间差最小的
- 全程向量化 + scipy KD-tree，关键路径无 Python 逐像元循环
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

import config as cfg
import fusion_config as fc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 球面坐标转换
# ---------------------------------------------------------------------------

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) 度 -> 单位球面 XYZ，用于 KD-tree 弦长距离。"""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def km_to_chord(km: float) -> float:
    """地球表面距离 (km) -> 单位球弦长。"""
    return 2.0 * np.sin(km / (2.0 * 6371.0))


# ---------------------------------------------------------------------------
# MODIS 完整落入 AGRI 圆盘检测
# ---------------------------------------------------------------------------

def check_modis_in_agri_disk(
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    max_dist_km: float = 10.0,
) -> bool:
    """
    检测 MODIS 条带是否完整落入 AGRI 全圆盘有效范围内。
    采样 MODIS 边缘像元，若任一边缘点到最近有效 AGRI 像元的距离超过阈值，
    判定为未完整落入，返回 False。

    若调用前已将 AGRI 边缘像元设为 NaN（如通过 compute_tight_disk_mask），
    则本函数自动以收紧后的范围做检查。
    """
    agri_valid = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if not agri_valid.any():
        return False

    agri_xyz = latlon_to_xyz(agri_lat[agri_valid], agri_lon[agri_valid])
    tree = cKDTree(agri_xyz)

    h, w = modis_lat.shape
    # 采样边缘像元：四边每隔 step 取一点，行列配对生成避免长度不匹配
    step = max(1, min(h, w) // 50)

    # 上边: row=0
    top_cols = np.arange(0, w, step, dtype=int)
    top_rows = np.zeros(len(top_cols), dtype=int)
    # 下边: row=h-1
    bot_cols = np.arange(0, w, step, dtype=int)
    bot_rows = np.full(len(bot_cols), h - 1, dtype=int)
    # 左边: col=0
    left_rows = np.arange(0, h, step, dtype=int)
    left_cols = np.zeros(len(left_rows), dtype=int)
    # 右边: col=w-1
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


def compute_tight_disk_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    margin_deg: float = 5.0,
    sub_lon: float = 105.0,
) -> np.ndarray:
    """
    计算 AGRI 全圆盘缩紧后的有效像元 mask。

    以星下点 (0°N, sub_lon°E) 为圆心，计算每个像元的角距离，
    剔除距圆盘边界 margin_deg 度以内的边缘像元。

    Parameters
    ----------
    lat, lon : 2D arrays
    margin_deg : 从圆盘边界向内收缩的度数。
    sub_lon : 静止卫星星下点经度。
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
# 时间权重函数
# ---------------------------------------------------------------------------

def time_weight(dt_min: float, is_fallback: bool = False) -> float:
    """
    时间差(分钟) -> 权重 [0, 1]。
    - dt <= TIME_HIGH_Q  : 1.0
    - dt in (HQ, LQ]    : 线性降到 0.5
    - dt > TIME_LOW_Q   : 0.0（外部已过滤，这里保险）
    - is_fallback        : 乘以 SCAN_TIME_FALLBACK_WEIGHT（文件名时间精度低）
    """
    if dt_min <= fc.TIME_HIGH_Q_MIN:
        w = 1.0
    elif dt_min <= fc.TIME_LOW_Q_MIN:
        w = 1.0 - 0.5 * (dt_min - fc.TIME_HIGH_Q_MIN) / (fc.TIME_LOW_Q_MIN - fc.TIME_HIGH_Q_MIN)
    else:
        w = 0.0
    if is_fallback:
        w *= fc.SCAN_TIME_FALLBACK_WEIGHT
    return w


# ---------------------------------------------------------------------------
# 视差修正占位 Hook
# ---------------------------------------------------------------------------

def parallax_correction_hook(
    lat: np.ndarray,
    lon: np.ndarray,
    cth_m: np.ndarray,
    vza_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    【视差修正占位函数 - HOOK】

    物理背景：静止卫星（FY-4）从固定位置观测，高云由于视差会在图像中产生
    水平位移。MYD06 是 Aqua 极轨卫星产品，其坐标是真实地面坐标。
    但当两者时间不同且云在高处时，FY-4 看到的云位置与 MYD06 坐标会有偏差。

    当前实现：标注高风险像元（高云 + 大视角），不做实际坐标修正。

    完整实现步骤（TODO）：
    1. 读取 FY-4 对每个 MYD06 像元的方位角和天顶角
    2. 计算视差位移：delta_x = CTH * tan(VZA) / R_earth (弧度)
    3. 按 FY-4 方位角方向将 MYD06 像元坐标平移 delta_x
    4. 用修正后坐标重新做 KD-tree 聚合

    返回
    ----
    lat_corr, lon_corr : 修正后坐标（当前版本 = 输入坐标）
    high_risk_mask     : bool 数组，True = 高云 + 大视角，需降权或过滤
    """
    high_cloud  = np.isfinite(cth_m)  & (cth_m  > fc.PARALLAX_HIGH_CTH_M)
    large_angle = np.isfinite(vza_deg) & (vza_deg > fc.PARALLAX_HIGH_VZA_DEG)
    high_risk   = high_cloud & large_angle

    n_risk = int(high_risk.sum())
    if n_risk > 0:
        log.debug(
            "parallax_hook: %d high-risk pixels (CTH>%.0fm & VZA>%.0fdeg). "
            "No coord correction applied – placeholder only.",
            n_risk, fc.PARALLAX_HIGH_CTH_M, fc.PARALLAX_HIGH_VZA_DEG,
        )
    return lat.copy(), lon.copy(), high_risk


# ---------------------------------------------------------------------------
# 5km 坐标上采样到 1km（最近邻重复，不插值）
# ---------------------------------------------------------------------------

def upsample_5km_to_1km_coords(
    lat_5km: np.ndarray,
    lon_5km: np.ndarray,
    target_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 5km MYD06 经纬度网格上采样到 1km 形状。
    使用行列方向重复（np.repeat），不做插值。
    物理上这等价于"每条 5km 扫描线包含 5 条 1km 扫描线，
    共享同一卫星轨道位置"的近似。
    """
    H_1km, W_1km = target_shape
    H_5km, W_5km = lat_5km.shape
    rh = max(1, round(H_1km / max(H_5km, 1)))
    rw = max(1, round(W_1km / max(W_5km, 1)))
    lat_up = np.repeat(np.repeat(lat_5km, rh, axis=0), rw, axis=1)
    lon_up = np.repeat(np.repeat(lon_5km, rh, axis=0), rw, axis=1)
    return lat_up[:H_1km, :W_1km], lon_up[:H_1km, :W_1km]


# ---------------------------------------------------------------------------
# 主聚合引擎（全向量化）
# ---------------------------------------------------------------------------

def aggregate_modis_to_agri(
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    modis_list: List[dict],
) -> Optional[dict]:
    """
    将 MYD06 1km 像元匹配到 AGRI 4km 网格。

    对每个 AGRI 4km 像元，在 2.5 km 半径内寻找距离最近的 MODIS 1km 像元；
    若多个 MODIS 文件提供候选，选时间差最小的那个，直接取其 CLP/CER/COT/CTH 值。
    """
    if not modis_list:
        return None

    H_a, W_a = agri_lat.shape
    N_a = H_a * W_a

    a_lat_f = agri_lat.ravel()
    a_lon_f = agri_lon.ravel()
    nan_mask = np.isnan(a_lat_f) | np.isnan(a_lon_f)
    valid_agri = np.where(~nan_mask)[0]

    if len(valid_agri) == 0:
        return _empty_output(H_a, W_a)

    a_xyz = latlon_to_xyz(a_lat_f[valid_agri], a_lon_f[valid_agri])
    chord_1km = km_to_chord(fc.AGRI_SEARCH_RADIUS_KM)
    # 5km 像元更大，搜索半径扩大到覆盖 5km 对角线的一半

    # -------------------------------------------------------------------
    # 收集缓冲区：每个有效 AGRI 像元存最近邻候选
    # -------------------------------------------------------------------

    def _make_buf():
        return [[] for _ in range(len(valid_agri))]

    buf_clp  = _make_buf()
    buf_cer  = _make_buf()
    buf_cot  = _make_buf()
    buf_cth  = _make_buf()
    buf_dt   = _make_buf()
    buf_dist = _make_buf()

    # -------------------------------------------------------------------
    # 遍历每个 MYD06 文件
    # -------------------------------------------------------------------
    for m in modis_list:
        file_dt_min = float(m.get("_dt_min", np.inf))

        _collect_1km(
            m, a_xyz, valid_agri, chord_1km,
            file_dt_min,
            buf_clp, buf_cer, buf_cot, buf_cth,
            buf_dt, buf_dist,
        )

    # -------------------------------------------------------------------
    # 汇总阶段：对每个有效 AGRI 像元做聚合 + 质量控制
    # 使用 numpy 向量化尽量减少 Python 循环开销
    # -------------------------------------------------------------------
    out_clp   = np.full(N_a, np.nan, np.float32)
    out_cer   = np.full(N_a, np.nan, np.float32)
    out_cot   = np.full(N_a, np.nan, np.float32)
    out_cth   = np.full(N_a, np.nan, np.float32)
    out_dt    = np.full(N_a, np.nan, np.float32)
    out_dt_mean = np.full(N_a, np.nan, np.float32)
    out_dt_max  = np.full(N_a, np.nan, np.float32)
    out_dist_mean = np.full(N_a, np.nan, np.float32)
    out_dist_p95  = np.full(N_a, np.nan, np.float32)
    out_ovlp  = np.zeros(N_a, np.float32)
    out_vpx1  = np.zeros(N_a, np.int32)
    out_vpx5  = np.zeros(N_a, np.int32)
    out_cfrac = np.full(N_a, np.nan, np.float32)
    out_phcon = np.full(N_a, np.nan, np.float32)
    out_wt    = np.zeros(N_a, np.float32)

    for k, agri_idx in enumerate(valid_agri):
        clp_v  = np.array(buf_clp[k],  dtype=np.float32) if buf_clp[k]  else np.array([], np.float32)
        cer_v  = np.array(buf_cer[k],  dtype=np.float32) if buf_cer[k]  else np.array([], np.float32)
        cot_v  = np.array(buf_cot[k],  dtype=np.float32) if buf_cot[k]  else np.array([], np.float32)
        cth_v  = np.array(buf_cth[k],  dtype=np.float32) if buf_cth[k]  else np.array([], np.float32)
        dt_v   = np.array(buf_dt[k],   dtype=np.float32) if buf_dt[k]   else np.array([], np.float32)
        dist_v = np.array(buf_dist[k], dtype=np.float32) if buf_dist[k] else np.array([], np.float32)

        n_candidates = len(clp_v)
        out_vpx1[agri_idx] = n_candidates

        # 时间差过滤
        if n_candidates > 0:
            keep_t = (np.isfinite(clp_v) & np.isfinite(dt_v) & (dt_v <= fc.TIME_LOW_Q_MIN))
            clp_v  = clp_v[keep_t]
            cer_v  = cer_v[keep_t]
            cot_v  = cot_v[keep_t]
            cth_v  = cth_v[keep_t]
            dt_v   = dt_v[keep_t]
            dist_v = dist_v[keep_t]
            n_candidates = len(clp_v)

        if n_candidates == 0:
            continue

        # 选时间差最小的候选
        best_idx  = int(np.argmin(dt_v))
        best_clp  = float(clp_v[best_idx])
        best_cer  = float(cer_v[best_idx])
        best_cot  = float(cot_v[best_idx])
        best_cth  = float(cth_v[best_idx])
        best_dt   = float(dt_v[best_idx])
        best_dist = float(dist_v[best_idx])

        if not (np.isfinite(best_clp) and best_clp >= 0 and best_clp < cfg.CLP_CLASSES):
            continue

        out_clp[agri_idx]       = best_clp
        out_dt[agri_idx]        = best_dt
        out_dt_mean[agri_idx]   = best_dt
        out_dt_max[agri_idx]    = best_dt
        out_dist_mean[agri_idx] = best_dist
        out_dist_p95[agri_idx]  = best_dist
        out_ovlp[agri_idx]      = 1.0
        out_phcon[agri_idx]     = 1.0
        is_cloudy = best_clp > 0
        out_cfrac[agri_idx]     = 1.0 if is_cloudy else 0.0
        out_wt[agri_idx]        = float(time_weight(best_dt))

        if is_cloudy:
            max_cth = getattr(cfg, "MAX_CTH_M", 18000)
            if np.isfinite(best_cer) and 0 <= best_cer <= 100:
                out_cer[agri_idx] = best_cer
            if np.isfinite(best_cot) and 0 <= best_cot <= 200:
                out_cot[agri_idx] = best_cot
            if np.isfinite(best_cth) and 0 <= best_cth <= max_cth:
                out_cth[agri_idx] = best_cth
                out_vpx5[agri_idx] = 1

    scan_time_sources = sorted({str(m.get("_scan_time_source", "unknown")) for m in modis_list})
    geo_sources = sorted({str(m.get("_geo_source", "unknown")) for m in modis_list})
    return {
        "CLP":               out_clp.reshape(H_a, W_a),
        "CTH":               out_cth.reshape(H_a, W_a),
        "MATCH_DT_MIN":      out_dt.reshape(H_a, W_a),
        "MATCH_DT_MEAN":     out_dt_mean.reshape(H_a, W_a),
        "MATCH_DT_MAX":      out_dt_max.reshape(H_a, W_a),
        "MATCH_DIST_MEAN_KM": out_dist_mean.reshape(H_a, W_a),
        "MATCH_DIST_P95_KM": out_dist_p95.reshape(H_a, W_a),
        "OVERLAP_FRACTION":  out_ovlp.reshape(H_a, W_a),
        "VALID_PIX_1KM":     out_vpx1.reshape(H_a, W_a),
        "VALID_PIX_5KM":     out_vpx5.reshape(H_a, W_a),
        "VALID_PIX_CTH_1KM": out_vpx5.reshape(H_a, W_a),
        "CLOUD_FRACTION":    out_cfrac.reshape(H_a, W_a),
        "PHASE_CONSISTENCY": out_phcon.reshape(H_a, W_a),
        "SAMPLE_WEIGHT":     out_wt.reshape(H_a, W_a),
        "_scan_time_sources": scan_time_sources,
        "_geo_sources": geo_sources,
        "_fallback_granules": int(sum(bool(m.get("_scan_time_is_fallback", True)) for m in modis_list)),
    }


# ---------------------------------------------------------------------------
# 内部辅助：收集 1km 最近邻到缓冲区（KD-tree k=1）
# ---------------------------------------------------------------------------

def _collect_1km(
    m: dict,
    a_xyz: np.ndarray,
    valid_agri: np.ndarray,
    chord: float,
    file_dt_min: float,
    buf_clp, buf_cer, buf_cot, buf_cth,
    buf_dt, buf_dist,
):
    clp_2d = m.get("CLP_1km")
    cer_2d = m.get("CER_1km")
    cot_2d = m.get("COT_1km")
    cth_2d = m.get("CTH_1km")
    lat_5  = m.get("lat_5km")
    lon_5  = m.get("lon_5km")
    lat_1  = m.get("lat_1km")
    lon_1  = m.get("lon_1km")
    scan_t = m.get("scan_time_1km")

    if clp_2d is None or lat_5 is None:
        return

    H_1km, W_1km = clp_2d.shape

    if lat_1 is not None and lon_1 is not None and lat_1.shape == (H_1km, W_1km) and lon_1.shape == (H_1km, W_1km):
        lat_1km, lon_1km = lat_1, lon_1
    else:
        lat_1km, lon_1km = upsample_5km_to_1km_coords(lat_5, lon_5, (H_1km, W_1km))

    lat_f = lat_1km.ravel()
    lon_f = lon_1km.ravel()
    clp_f = clp_2d.ravel()
    cer_f = (cer_2d.ravel() if cer_2d is not None
             else np.full(lat_f.shape, np.nan, np.float32))
    cot_f = (cot_2d.ravel() if cot_2d is not None
             else np.full(lat_f.shape, np.nan, np.float32))
    cth_f = (cth_2d.ravel() if cth_2d is not None
             else np.full(lat_f.shape, np.nan, np.float32))

    # 像元级时间差
    if scan_t is not None:
        dt_f = np.abs(scan_t.ravel()[:len(lat_f)]).astype(np.float32)
    else:
        dt_f = np.full(len(lat_f), float(file_dt_min), np.float32)

    # 预过滤：地理坐标有效 + 时间差在可接受范围
    geo_ok  = np.isfinite(lat_f) & np.isfinite(lon_f)
    time_ok = dt_f <= fc.TIME_LOW_Q_MIN
    keep = geo_ok & time_ok
    if keep.sum() == 0:
        return

    idx_keep = np.where(keep)[0]
    lat_k = lat_f[idx_keep]
    lon_k = lon_f[idx_keep]
    clp_k = clp_f[idx_keep]
    cer_k = cer_f[idx_keep]
    cot_k = cot_f[idx_keep]
    cth_k = cth_f[idx_keep]
    dt_k  = dt_f[idx_keep]

    m_xyz = latlon_to_xyz(lat_k, lon_k)
    tree  = cKDTree(m_xyz)
    dist_chord, nn_idx = tree.query(a_xyz, k=1, distance_upper_bound=chord, workers=1)

    for k_a in range(len(valid_agri)):
        ii = nn_idx[k_a]
        if ii >= len(m_xyz):
            continue
        d_km = float(2.0 * 6371.0 * np.arcsin(np.clip(float(dist_chord[k_a]) * 0.5, 0.0, 1.0)))
        buf_clp[k_a].append(float(clp_k[ii]))
        buf_cer[k_a].append(float(cer_k[ii]))
        buf_cot[k_a].append(float(cot_k[ii]))
        buf_cth[k_a].append(float(cth_k[ii]))
        buf_dt[k_a].append(float(dt_k[ii]))
        buf_dist[k_a].append(d_km)


# ---------------------------------------------------------------------------
# 辅助：空输出
# ---------------------------------------------------------------------------

def _empty_output(H: int, W: int) -> dict:
    nan2d  = np.full((H, W), np.nan, np.float32)
    zero2d = np.zeros((H, W), np.float32)
    int2d  = np.zeros((H, W), np.int32)
    return {
        "CLP": nan2d.copy(), "CTH": nan2d.copy(),
        "MATCH_DT_MIN": nan2d.copy(), "MATCH_DT_MEAN": nan2d.copy(), "MATCH_DT_MAX": nan2d.copy(),
        "MATCH_DIST_MEAN_KM": nan2d.copy(), "MATCH_DIST_P95_KM": nan2d.copy(),
        "OVERLAP_FRACTION": zero2d.copy(),
        "VALID_PIX_1KM": int2d.copy(), "VALID_PIX_5KM": int2d.copy(),
        "VALID_PIX_CTH_1KM": int2d.copy(),
        "CLOUD_FRACTION": nan2d.copy(), "PHASE_CONSISTENCY": nan2d.copy(),
        "SAMPLE_WEIGHT": zero2d.copy(),
        "_scan_time_sources": [], "_geo_sources": [], "_fallback_granules": 0,
    }
