"""
fusion_core.py
==============
质量优先的 MYD06 → AGRI 聚合核心引擎。

本模块与 IO 解耦，只负责：
  1. 从已读取的 MYD06 字典列表中，将所有变量聚合到 AGRI 4km 像元上
  2. 先用 Phase 多数表决确定主相态，再在主相态候选内聚合
     COT/CER/CTH（COT=对数域中位数 / CER=加权均值 / CTH=加权均值）
  3. 输出完整质量字段（time_diff, overlap_fraction, cloud_fraction,
     phase_consistency, sample_weight 等）

设计原则
--------
- MYD06 → AGRI 聚合（绝不反向）
- 质量优先：宁可 NaN，不做伪标签
- 1km 变量（CLP/CER/COT/CTH_1km）共同收集；回归量按主相态分层聚合
- 像元级时间过滤：scan_time 优先，文件名时间作 fallback 并降权
- 全程向量化 + scipy KD-tree，无 Python 逐像元循环（关键路径）
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

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
# 变量类型专用聚合函数
# ---------------------------------------------------------------------------

def aggregate_cot(values: np.ndarray, weights: np.ndarray) -> float:
    """
    COT 聚合：对数域加权中位数再还原。
    物理背景：COT 高度右偏，对数域中位数比线性均值更稳健。
    只使用 COT > 0 的有效云像元。
    """
    mask = np.isfinite(values) & (values > 0) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    v = np.log(values[mask] + fc.COT_LOG_EPS)
    w = weights[mask]
    # 加权中位数
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cumw = np.cumsum(w_sorted)
    mid  = cumw[-1] / 2.0
    idx  = np.searchsorted(cumw, mid)
    idx  = min(idx, len(v_sorted) - 1)
    return float(np.exp(v_sorted[idx]))


def aggregate_cer(values: np.ndarray, weights: np.ndarray) -> float:
    """
    CER 聚合：加权均值（线性域）。
    CER 分布近似正态，线性均值物理意义清晰。
    只使用 CER > 0 的有效云像元。
    """
    mask = np.isfinite(values) & (values > 0) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    v, w = values[mask], weights[mask]
    return float(np.average(v, weights=w))


def aggregate_phase(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """
    Phase 聚合：加权多数表决（绝不线性插值）。
    返回 (dominant_class, consistency)。
    consistency < fc.PHASE_CONSISTENCY_MIN 时调用方应丢弃标签。
    """
    mask = np.isfinite(values) & (values >= 0) & (weights > 0)
    if mask.sum() == 0:
        return np.nan, 0.0
    v = values[mask].astype(np.int32)
    w = weights[mask]
    classes = np.unique(v)
    class_weights = np.array([w[v == c].sum() for c in classes])
    total_w = class_weights.sum()
    dominant_idx  = np.argmax(class_weights)
    consistency   = float(class_weights[dominant_idx]) / float(total_w)
    return float(classes[dominant_idx]), consistency


def aggregate_cth(values: np.ndarray, weights: np.ndarray) -> float:
    """CTH 聚合：加权均值（线性域）。"""
    mask = np.isfinite(values) & (values >= 0) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    v, w = values[mask], weights[mask]
    return float(np.average(v, weights=w))


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
    agri_lat: np.ndarray,   # (H_a, W_a) AGRI 像元纬度
    agri_lon: np.ndarray,   # (H_a, W_a) AGRI 像元经度
    modis_list: List[dict], # read_myd06() 输出的列表，每个 dict 含 '_dt_min'
) -> Optional[dict]:
    """
    核心聚合函数：将 MYD06 列表中所有文件的像元聚合到 AGRI 4km 网格。

    算法流程
    --------
    1. 对每个 MYD06 文件：
       a. 优先使用 MYD03 1km 坐标；缺失时上采样 5km 坐标 → 1km
       b. 构建 1km KD-tree
       c. query_ball_point：先找每个 AGRI 像元附近的 MYD06 像元，再保留最近的
          EXPECTED_1KM_PER_AGRI 个候选
       d. 逐 AGRI 像元收集候选值和时间权重（列表追加）
    2. 汇总阶段（全向量化）：
       a. Phase：加权多数表决
       b. COT：对数域加权中位数（只用主相态云像元）
       c. CER：加权均值（只用主相态云像元）
       d. CTH：加权均值
    3. 质量控制：对每个 AGRI 像元独立判断
       - 时间差 > TIME_LOW_Q → NaN
       - overlap_fraction < OVERLAP_FRAC_MIN → NaN
       - cloud_fraction < CLOUD_FRAC_MIN_CLOUDY → COT/CER NaN
       - phase_consistency < PHASE_CONSISTENCY_MIN → Phase NaN
    4. 输出质量字段

    参数
    ----
    agri_lat, agri_lon : AGRI 经纬度网格 (H_a, W_a)
    modis_list : 每个元素为 read_myd06() 返回的 dict，
                 额外字段: _dt_min (文件名级时间差, 分钟)

    返回
    ----
    dict，含：
      CLP, CER, COT, CTH          : (H_a, W_a) float32，NaN=无效
      MATCH_DT_MIN                 : 最优像元时间差
      OVERLAP_FRACTION             : 1km 空间覆盖率
      VALID_PIX_1KM, VALID_PIX_5KM      : 后者为历史字段名，当前记录 CTH_1km 候选数
      CLOUD_FRACTION               : 有效 1km 中云像元比例
      PHASE_CONSISTENCY            : Phase 多数表决一致性
      SAMPLE_WEIGHT                : 综合样本权重（时间 × 覆盖）
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
    # 收集缓冲区：用 numpy object array 存 list（避免 Python 逐像元循环）
    # -------------------------------------------------------------------
    # 每个有效 AGRI 像元一个位置，存 (value, time_weight) pairs
    _empty = [[] for _ in range(len(valid_agri))]

    def _make_buf():
        return [[] for _ in range(len(valid_agri))]

    buf_clp_v = _make_buf()   # CLP 值
    buf_clp_w = _make_buf()   # 对应权重
    buf_cer_v = _make_buf()
    buf_cer_w = _make_buf()
    buf_cot_v = _make_buf()
    buf_cot_w = _make_buf()
    buf_cth_v = _make_buf()
    buf_cth_w = _make_buf()
    buf_dt    = _make_buf()   # 时间差（用于输出 MATCH_DT_MIN）
    buf_dist  = _make_buf()   # AGRI 中心到 MODIS 像元中心的距离 km

    # -------------------------------------------------------------------
    # 遍历每个 MYD06 文件
    # -------------------------------------------------------------------
    for m in modis_list:
        file_dt_min  = float(m.get("_dt_min", np.inf))
        is_fallback  = m.get("_scan_time_is_fallback", True)

        # ---- 1km 聚合 (CLP / CER / COT) ----
        _collect_1km(
            m, a_xyz, valid_agri, chord_1km,
            file_dt_min, is_fallback,
            buf_clp_v, buf_clp_w,
            buf_cer_v, buf_cer_w,
            buf_cot_v, buf_cot_w,
            buf_cth_v, buf_cth_w,
            buf_dt,
            buf_dist,
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
        clp_v = np.array(buf_clp_v[k], dtype=np.float32) if buf_clp_v[k] else np.array([], np.float32)
        clp_w = np.array(buf_clp_w[k], dtype=np.float32) if buf_clp_w[k] else np.array([], np.float32)
        cer_v = np.array(buf_cer_v[k], dtype=np.float32) if buf_cer_v[k] else np.array([], np.float32)
        cer_w = np.array(buf_cer_w[k], dtype=np.float32) if buf_cer_w[k] else np.array([], np.float32)
        cot_v = np.array(buf_cot_v[k], dtype=np.float32) if buf_cot_v[k] else np.array([], np.float32)
        cot_w = np.array(buf_cot_w[k], dtype=np.float32) if buf_cot_w[k] else np.array([], np.float32)
        cth_v = np.array(buf_cth_v[k], dtype=np.float32) if buf_cth_v[k] else np.array([], np.float32)
        cth_w = np.array(buf_cth_w[k], dtype=np.float32) if buf_cth_w[k] else np.array([], np.float32)
        dt_v  = np.array(buf_dt[k],    dtype=np.float32) if buf_dt[k]    else np.array([], np.float32)
        dist_v = np.array(buf_dist[k], dtype=np.float32) if buf_dist[k] else np.array([], np.float32)

        n_1km = len(clp_v)
        out_vpx1[agri_idx] = n_1km

        # 时间差 -> 过滤 > TIME_LOW_Q 的像元
        if n_1km > 0 and dt_v.size > 0:
            keep_t = dt_v <= fc.TIME_LOW_Q_MIN
            clp_v, clp_w = clp_v[keep_t], clp_w[keep_t]
            cer_v, cer_w = cer_v[keep_t], cer_w[keep_t]
            cot_v, cot_w = cot_v[keep_t], cot_w[keep_t]
            cth_v, cth_w = cth_v[keep_t], cth_w[keep_t]
            dt_v = dt_v[keep_t]
            dist_v = dist_v[keep_t]
            n_1km = len(clp_v)

        # overlap_fraction (1km)
        ovlp = min(float(n_1km) / fc.EXPECTED_1KM_PER_AGRI, 1.0)
        out_ovlp[agri_idx] = ovlp

        # 最优时间差
        valid_dt = dt_v[np.isfinite(clp_v)] if n_1km > 0 else np.array([])
        best_dt  = float(np.min(valid_dt)) if valid_dt.size > 0 else np.nan
        out_dt[agri_idx] = best_dt
        if valid_dt.size > 0:
            out_dt_mean[agri_idx] = float(np.mean(valid_dt))
            out_dt_max[agri_idx] = float(np.max(valid_dt))
        valid_dist = dist_v[np.isfinite(clp_v)] if n_1km > 0 and dist_v.size > 0 else np.array([])
        if valid_dist.size > 0:
            out_dist_mean[agri_idx] = float(np.mean(valid_dist))
            out_dist_p95[agri_idx] = float(np.percentile(valid_dist, 95))

        # cloud_fraction
        n_valid_clp = int(np.isfinite(clp_v).sum())
        cloud_mask  = np.isfinite(clp_v) & (clp_v > 0)
        n_cloud     = int(cloud_mask.sum())
        cfrac       = float(n_cloud) / float(n_valid_clp) if n_valid_clp > 0 else np.nan
        out_cfrac[agri_idx] = cfrac

        # 通用质量门：overlap 不足则跳过
        if ovlp < fc.OVERLAP_FRAC_MIN or n_1km < fc.MIN_VALID_PIX:
            continue

        # ---- Phase（多数表决）----
        phase_val, phase_con = aggregate_phase(clp_v, clp_w)
        out_phcon[agri_idx] = phase_con
        phase_ok = np.isfinite(phase_val) and phase_con >= fc.PHASE_CONSISTENCY_MIN
        if phase_ok:
            out_clp[agri_idx] = phase_val

        # ---- COT / CER / CTH（只用主相态云像元）----
        phase_cloud_mask = cloud_mask & phase_ok & (clp_v == phase_val)
        n_phase_cloud = int(phase_cloud_mask.sum())
        cloud_ok = (
            n_phase_cloud >= fc.MIN_VALID_PIX and
            np.isfinite(cfrac) and cfrac >= fc.CLOUD_FRAC_MIN_CLOUDY and
            (not fc.PURE_CLOUD_ONLY or cfrac >= fc.PURE_CLOUD_FRAC)
        )
        if cloud_ok:
            out_cot[agri_idx] = aggregate_cot(
                cot_v[phase_cloud_mask], cot_w[phase_cloud_mask]
            )
            out_cer[agri_idx] = aggregate_cer(
                cer_v[phase_cloud_mask], cer_w[phase_cloud_mask]
            )

        # ---- CTH（当前配置按 CTH_1km 与主相态云候选聚合）----
        n_cth = int(np.isfinite(cth_v[phase_cloud_mask]).sum()) if cloud_ok else 0
        out_vpx5[agri_idx] = n_cth
        if cloud_ok and n_cth > 0:
            cth_val = aggregate_cth(cth_v[phase_cloud_mask], cth_w[phase_cloud_mask])
            max_cth = getattr(cfg, "MAX_CTH_M", 18000)
            if np.isfinite(cth_val) and 0 <= cth_val <= max_cth:
                out_cth[agri_idx] = cth_val

        # ---- sample_weight ----
        tw = time_weight(best_dt) if np.isfinite(best_dt) else 0.0
        out_wt[agri_idx] = float(tw * ovlp)

    scan_time_sources = sorted({str(m.get("_scan_time_source", "unknown")) for m in modis_list})
    geo_sources = sorted({str(m.get("_geo_source", "unknown")) for m in modis_list})
    return {
        "CLP":               out_clp.reshape(H_a, W_a),
        "CER":               out_cer.reshape(H_a, W_a),
        "COT":               out_cot.reshape(H_a, W_a),
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
# 内部辅助：收集 1km 候选值到缓冲区（KD-tree query_ball_point）
# ---------------------------------------------------------------------------

def _collect_1km(
    m: dict,
    a_xyz: np.ndarray,
    valid_agri: np.ndarray,
    chord: float,
    file_dt_min: float,
    is_fallback: bool,
    buf_clp_v, buf_clp_w,
    buf_cer_v, buf_cer_w,
    buf_cot_v, buf_cot_w,
    buf_cth_v, buf_cth_w,
    buf_dt,
    buf_dist,
):
    clp_2d = m.get("CLP_1km")
    cer_2d = m.get("CER_1km")
    cot_2d = m.get("COT_1km")
    cth_2d = m.get("CTH_1km")
    lat_5  = m.get("lat_5km")
    lon_5  = m.get("lon_5km")
    lat_1  = m.get("lat_1km")
    lon_1  = m.get("lon_1km")
    scan_t = m.get("scan_time_1km")   # (H_1km, W_1km) float32 分钟偏移，或 None

    if clp_2d is None or lat_5 is None:
        return

    H_1km, W_1km = clp_2d.shape

    if lat_1 is not None and lon_1 is not None and lat_1.shape == (H_1km, W_1km) and lon_1.shape == (H_1km, W_1km):
        lat_1km, lon_1km = lat_1, lon_1
    else:
        # 5km 坐标上采样到 1km（重复，不插值）
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

    # 像元级时间权重
    if scan_t is not None:
        dt_f = np.abs(scan_t.ravel()[:len(lat_f)]).astype(np.float32)
        fallback = False
    else:
        dt_f = np.full(len(lat_f), float(file_dt_min), np.float32)
        fallback = True

    # 预过滤：地理坐标有效 + 时间差在可接受范围
    geo_ok = np.isfinite(lat_f) & np.isfinite(lon_f)
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

    # 时间权重向量（向量化，避免 Python 循环）
    w_k = np.where(
        dt_k <= fc.TIME_HIGH_Q_MIN,
        1.0,
        1.0 - 0.5 * (dt_k - fc.TIME_HIGH_Q_MIN) /
        (fc.TIME_LOW_Q_MIN - fc.TIME_HIGH_Q_MIN)
    ).astype(np.float32)
    if fallback:
        w_k *= fc.SCAN_TIME_FALLBACK_WEIGHT

    m_xyz = latlon_to_xyz(lat_k, lon_k)
    tree  = cKDTree(m_xyz)
    # query_ball_point 只做空间预筛；随后按距离保留约 4km footprint 对应的 16 个 1km 点。
    nbrs_list = tree.query_ball_point(a_xyz, r=chord, workers=1)
    max_candidates = max(1, int(round(float(fc.EXPECTED_1KM_PER_AGRI))))

    for k_a, nbrs in enumerate(nbrs_list):
        if len(nbrs) == 0:
            continue
        nbrs = np.asarray(nbrs, dtype=np.int64)
        d = np.linalg.norm(m_xyz[nbrs] - a_xyz[k_a], axis=1)
        d = 2.0 * 6371.0 * np.arcsin(np.clip(d * 0.5, 0.0, 1.0))
        if nbrs.size > max_candidates:
            order = np.argsort(d, kind="stable")[:max_candidates]
            nbrs = nbrs[order]
            d = d[order]
        buf_clp_v[k_a].extend(clp_k[nbrs].tolist())
        buf_clp_w[k_a].extend(w_k[nbrs].tolist())
        buf_cer_v[k_a].extend(cer_k[nbrs].tolist())
        buf_cer_w[k_a].extend(w_k[nbrs].tolist())
        buf_cot_v[k_a].extend(cot_k[nbrs].tolist())
        buf_cot_w[k_a].extend(w_k[nbrs].tolist())
        buf_cth_v[k_a].extend(cth_k[nbrs].tolist())
        buf_cth_w[k_a].extend(w_k[nbrs].tolist())
        buf_dt[k_a].extend(dt_k[nbrs].tolist())
        buf_dist[k_a].extend(d.astype(np.float32).tolist())


# ---------------------------------------------------------------------------
# 辅助：空输出
# ---------------------------------------------------------------------------

def _empty_output(H: int, W: int) -> dict:
    nan2d  = np.full((H, W), np.nan, np.float32)
    zero2d = np.zeros((H, W), np.float32)
    int2d  = np.zeros((H, W), np.int32)
    return {
        "CLP": nan2d.copy(), "CER": nan2d.copy(),
        "COT": nan2d.copy(), "CTH": nan2d.copy(),
        "MATCH_DT_MIN": nan2d.copy(), "MATCH_DT_MEAN": nan2d.copy(), "MATCH_DT_MAX": nan2d.copy(),
        "MATCH_DIST_MEAN_KM": nan2d.copy(), "MATCH_DIST_P95_KM": nan2d.copy(),
        "OVERLAP_FRACTION": zero2d.copy(),
        "VALID_PIX_1KM": int2d.copy(), "VALID_PIX_5KM": int2d.copy(),
        "VALID_PIX_CTH_1KM": int2d.copy(),
        "CLOUD_FRACTION": nan2d.copy(), "PHASE_CONSISTENCY": nan2d.copy(),
        "SAMPLE_WEIGHT": zero2d.copy(),
        "_scan_time_sources": [], "_geo_sources": [], "_fallback_granules": 0,
    }
