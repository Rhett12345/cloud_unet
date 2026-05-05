"""
data_fusion.py  (质量优先多进程版)
====================================
AGRI + MYD06 融合流水线的顶层调度器。

架构
----
  data_fusion.py   <- 本文件：调度、多进程、QC 图
  fusion_core.py   <- 纯数值聚合引擎（无 IO）
  fusion_io.py     <- 文件读写（AGRI / MYD06 / HDF5 输出）
  fusion_config.py <- 质量控制阈值（可被环境变量覆盖）

多进程策略
----------
- 每个 AGRI 文件作为一个独立任务
- ProcessPoolExecutor：N-1 个 worker 并行，主进程调度
- 子进程之间无共享状态

用法
----
  python data_fusion.py --split train --day 20190105
  python data_fusion.py --split train --workers 8
"""
from __future__ import annotations
import argparse, csv, json, logging, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np

import config as cfg
import fusion_config as fc
from fusion_core import (aggregate_modis_to_agri, check_modis_in_agri_disk,
                         compute_tight_disk_mask, latlon_to_xyz)
from fusion_io import (
    apply_quality_filter, find_day_folders, find_matching_modis, find_matching_myd03,
    parse_agri_datetime, parse_modis_datetime,
    read_agri_scene, read_myd06, read_modis_geo_quick,
    write_fused_samples, write_full_disk_hdf5,
)
from sample_filters import get_patch_supervision_thresholds

log = logging.getLogger(__name__)

# compat alias used by main.py
_find_day_folders = find_day_folders


QC_DIAGNOSTIC_FIELDS = [
    "scene_id", "agri_file", "myd06_file", "myd03_file",
    "raw_clp_valid_px", "raw_cer_valid_px", "raw_cot_valid_px", "raw_cth_valid_px",
    "time_ok_px", "geo_ok_px",
    "reg_time_ok_px", "reg_geo_ok_px",
    "cumulative_base_px", "cumulative_after_time_px",
    "cumulative_after_geo_px",
    "cumulative_after_reg_time_px",
    "cumulative_after_reg_geo_px",
    "final_clp_px", "final_cer_px", "final_cot_px", "final_cth_px",
    "time_delta_min_p50", "time_delta_min_p90", "time_delta_min_max",
]


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _reset_qc_diagnostics(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["qc_gate_stats.csv", "qc_gate_stats.jsonl"]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def _write_qc_diagnostics(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qc_gate_stats.csv"
    jsonl_path = out_dir / "qc_gate_stats.jsonl"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QC_DIAGNOSTIC_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k)) for k in QC_DIAGNOSTIC_FIELDS})
    with jsonl_path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = {k: _json_safe(row.get(k)) for k in QC_DIAGNOSTIC_FIELDS}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log.info("QC diagnostics saved -> %s and %s", csv_path, jsonl_path)


def _unpack_scene_result(result):
    if len(result) == 3:
        ok, op, msg = result
        return ok, op, msg, None
    ok, op, msg, diag = result
    return ok, op, msg, diag


def _fuse_one_scene(agri_file, modis_files, out_path, mode, qc_diagnostics_enabled=False):
    """子进程任务：融合单个 AGRI 场景，返回 (ok, out_path, msg)。"""
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    agri_path = Path(agri_file)
    out = Path(out_path)
    diag_row = None
    try:
        agri_dt = parse_agri_datetime(agri_path.name)
        if agri_dt is None:
            return False, out_path, "Cannot parse AGRI datetime", diag_row

        agri = read_agri_scene(agri_path)
        if agri is None:
            return False, out_path, "read_agri_scene None", diag_row

        # ── 保存完整圆盘经纬度（可视化用）──
        full_lat = agri["lat"].copy()
        full_lon = agri["lon"].copy()

        # ── 收紧 AGRI 圆盘边界，向内缩 margin_deg 度 ──
        margin = float(getattr(fc, "AGRI_DISK_MARGIN_DEG", 5.0))
        tight_mask = np.ones(agri["lat"].shape, dtype=bool)
        if margin > 0:
            sub_lon = float(getattr(fc, "AGRI_SUB_LON", 105.0))
            tight_mask = compute_tight_disk_mask(agri["lat"], agri["lon"], margin, sub_lon=sub_lon)
            agri["lat"] = np.where(tight_mask, agri["lat"], np.nan)
            agri["lon"] = np.where(tight_mask, agri["lon"], np.nan)
            agri["VZA"] = np.where(tight_mask, agri["VZA"], np.nan)
            agri["SZA"] = np.where(tight_mask, agri["SZA"], np.nan)
            bt = agri["BT"]
            mask_3d = np.broadcast_to(tight_mask[..., np.newaxis], bt.shape)
            agri["BT"] = np.where(mask_3d, bt, np.nan)
            n_before = int(np.isfinite(full_lat).sum())
            n_after = int(tight_mask.sum())
            log.debug("AGRI disk margin %.1f°: %d → %d valid pixels (%.1f%%)",
                      margin, n_before, n_after, 100.0 * n_after / max(n_before, 1))

        modis_list = []
        myd06_names = []
        myd03_names = []
        n_skipped_geo = 0
        for item in modis_files:
            if isinstance(item, (list, tuple)):
                mf = Path(item[0])
                myd03_file = Path(item[1]) if len(item) > 1 and item[1] else None
            else:
                mf = Path(item)
                myd03_file = None

            # ── 第1层：轻量地理预检（只读 lat/lon，不过则跳过全部科学数据）──
            geo = read_modis_geo_quick(mf, myd03_file=myd03_file)
            if geo is None:
                continue
            modis_lat = geo.get("lat_1km") if geo.get("lat_1km") is not None else geo.get("lat_5km")
            modis_lon = geo.get("lon_1km") if geo.get("lon_1km") is not None else geo.get("lon_5km")
            if modis_lat is None or modis_lon is None:
                continue
            if not check_modis_in_agri_disk(modis_lat, modis_lon, agri["lat"], agri["lon"]):
                n_skipped_geo += 1
                continue

            # ── 第2层：完整读取（传入 geo_cache 避免重复读经纬度）──
            m = read_myd06(mf, agri_dt=agri_dt, myd03_file=myd03_file, geo_cache=geo)
            if m is None:
                continue

            myd06_names.append(mf.name)
            if myd03_file is not None:
                myd03_names.append(myd03_file.name)

            mdt = parse_modis_datetime(mf.name)
            if mdt is None:
                continue
            m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
            m["_file"] = mf.name
            modis_list.append(m)

        if n_skipped_geo > 0:
            log.debug("Geo pre-check skipped %d/%d MODIS granules for %s",
                      n_skipped_geo, n_skipped_geo + len(modis_list), agri_path.name)

        if not modis_list:
            return False, out_path, "No MYD06 after reading", diag_row

        # 收集 MODIS 条带轮廓和边界（用于地理可视化验证）
        modis_bounds = []
        for m in modis_list:
            mlat = m.get("lat_1km") if m.get("lat_1km") is not None else m.get("lat_5km")
            mlon = m.get("lon_1km") if m.get("lon_1km") is not None else m.get("lon_5km")
            if mlat is not None and mlon is not None:
                valid = np.isfinite(mlat) & np.isfinite(mlon)
                if valid.any():
                    outline = _compute_swath_outline(mlat, mlon, valid)
                    border_ok = _check_modis_border_in_agri(
                        mlat, mlon, valid, agri["lat"], agri["lon"])
                    modis_bounds.append({
                        "lat_min": float(mlat[valid].min()),
                        "lat_max": float(mlat[valid].max()),
                        "lon_min": float(mlon[valid].min()),
                        "lon_max": float(mlon[valid].max()),
                        "outline_lat": outline["lat"],
                        "outline_lon": outline["lon"],
                        "border_in_agri": border_ok,
                        "file": m.get("_file", ""),
                    })

        labels = aggregate_modis_to_agri(agri["lat"], agri["lon"], modis_list)
        if labels is None:
            return False, out_path, "aggregate returned None", diag_row

        diagnostics = None
        if qc_diagnostics_enabled:
            diagnostics = {
                "scene_id": agri_dt.strftime("%Y%m%d_%H%M%S"),
                "agri_file": agri_path.name,
                "myd06_file": ";".join(myd06_names) if myd06_names else None,
                "myd03_file": ";".join(myd03_names) if myd03_names else None,
            }
        labels = apply_quality_filter(agri, labels, diagnostics=diagnostics)
        diag_row = diagnostics.get("row") if diagnostics is not None else None

        thresh = get_patch_supervision_thresholds(mode, tuple(cfg.PATCH_SIZE))
        n_clp = int(np.isfinite(labels["CLP"]).sum())
        n_cld = int((
            np.isfinite(labels["CLP"]) & (labels["CLP"] > 0) &
            np.isfinite(labels["CER"]) & np.isfinite(labels["COT"]) &
            np.isfinite(labels["CTH"])
        ).sum())
        if (n_clp < thresh["min_valid_label_pixels"] or
                n_cld < thresh["min_valid_cloudy_pixels"]):
            return False, out_path, (
                f"Too few: clp={n_clp}/{thresh['min_valid_label_pixels']} "
                f"cld={n_cld}/{thresh['min_valid_cloudy_pixels']}"
            ), diag_row

        out.parent.mkdir(parents=True, exist_ok=True)
        if cfg.FUSION_OUTPUT_MODE == "samples_only":
            n_s = write_fused_samples(out, agri, labels, agri_dt, mode)
            _make_geo_figure(agri, labels, agri_dt, modis_bounds,
                             out.with_name(out.stem + "_geo.png"),
                             full_lat=full_lat, full_lon=full_lon, tight_mask=tight_mask)
            return True, out_path, f"OK samples={n_s}", diag_row
        else:
            write_full_disk_hdf5(out, agri, labels, agri_dt)
            _make_geo_figure(agri, labels, agri_dt, modis_bounds,
                             out.with_name(out.stem + "_geo.png"),
                             full_lat=full_lat, full_lon=full_lon, tight_mask=tight_mask)
            return True, out_path, "OK full_disk", diag_row

    except Exception:
        return False, out_path, f"Exception:\n{traceback.format_exc()}", diag_row


def _compute_swath_outline(lat, lon, valid):
    """计算 MODIS 条带有效区域的轮廓（极角分箱近似凸包）。

    内部在 [0,360] 计算避免日期变更线撕裂，输出时为 [-180,180]
    供 cartopy PlateCarree data transform 使用。
    """
    y, x_raw = lat[valid], lon[valid]
    if len(y) < 3:
        return {"lat": np.array([]), "lon": np.array([])}

    x_360 = np.where(x_raw < 0, x_raw + 360.0, x_raw)
    center_lat = np.median(y)
    center_lon_360 = np.median(x_360)

    dlon = x_360 - center_lon_360
    angles = np.arctan2(y - center_lat, dlon)
    angles_2pi = np.where(angles < 0, angles + 2.0 * np.pi, angles)

    n_bins = 72
    bins = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    hull_lat, hull_lon_360 = [], []
    for i in range(n_bins):
        mask = (angles_2pi >= bins[i]) & (angles_2pi < bins[i + 1])
        if not mask.any():
            continue
        dist = np.sqrt((y[mask] - center_lat) ** 2 + (x_360[mask] - center_lon_360) ** 2)
        idx = np.argmax(dist)
        hull_lat.append(float(y[mask][idx]))
        hull_lon_360.append(float(x_360[mask][idx]))

    if len(hull_lat) < 3:
        return {"lat": np.array([]), "lon": np.array([])}

    hull_lat = np.array(hull_lat)
    hull_lon_360 = np.array(hull_lon_360)

    hull_dlon = hull_lon_360 - center_lon_360
    hull_angles = np.arctan2(hull_lat - center_lat, hull_dlon)
    hull_angles_2pi = np.where(hull_angles < 0, hull_angles + 2.0 * np.pi, hull_angles)
    order = np.argsort(hull_angles_2pi)

    hull_lon_plot = np.where(hull_lon_360 > 180.0, hull_lon_360 - 360.0, hull_lon_360)
    return {"lat": hull_lat[order], "lon": hull_lon_plot[order]}


def _check_modis_border_in_agri(modis_lat, modis_lon, modis_valid, agri_lat, agri_lon):
    """检测 MODIS 条带边缘是否完整落入 AGRI 圆盘内。
    采样 MODIS 四边像元，检查每个点到最近有效 AGRI 像元的距离。"""
    agri_valid = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    if not agri_valid.any():
        return False
    from scipy.spatial import cKDTree  # local import for subprocess
    agri_xyz = latlon_to_xyz(agri_lat[agri_valid], agri_lon[agri_valid])
    tree = cKDTree(agri_xyz)
    h, w = modis_lat.shape
    step = max(1, min(h, w) // 30)
    top_c = np.arange(0, w, step, dtype=int)
    bot_c = np.arange(0, w, step, dtype=int)
    left_r = np.arange(0, h, step, dtype=int)
    right_r = np.arange(0, h, step, dtype=int)
    edge_rows = np.concatenate([
        np.zeros(len(top_c), dtype=int),
        np.full(len(bot_c), h - 1, dtype=int),
        left_r,
        right_r,
    ])
    edge_cols = np.concatenate([
        top_c, bot_c,
        np.zeros(len(left_r), dtype=int),
        np.full(len(right_r), w - 1, dtype=int),
    ])
    valid_e = modis_valid[edge_rows, edge_cols]
    if not valid_e.any():
        return False
    slat = modis_lat[edge_rows, edge_cols][valid_e]
    slon = modis_lon[edge_rows, edge_cols][valid_e]
    xyz = latlon_to_xyz(slat, slon)
    dist, _ = tree.query(xyz, k=1)
    dist_km = 2.0 * 6371.0 * np.arcsin(np.clip(dist * 0.5, 0.0, 1.0))
    return bool(np.all(dist_km <= 10.0))


def _make_geo_figure(agri, labels, agri_dt, modis_bounds, save_path,
                     full_lat=None, full_lon=None, tight_mask=None):
    """地理定位验证图。

    使用 cartopy PlateCarree(central_longitude=104.7) 投影，
    AGRI 圆盘为中心，MODIS 条带叠加。cartopy 自动处理坐标变换。
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        sub_lon = float(getattr(fc, "AGRI_SUB_LON", 104.7))
        data_crs = ccrs.PlateCarree()
        map_crs = ccrs.PlateCarree(central_longitude=sub_lon)

        fig = plt.figure(figsize=(12, 11))
        ax = fig.add_subplot(1, 1, 1, projection=map_crs)

        # ── 海岸线 ──
        ax.add_feature(cfeature.COASTLINE, lw=0.5, alpha=0.5, zorder=4)

        # ── 背景：完整 AGRI 全圆盘（浅灰散点）──
        if full_lat is not None and full_lon is not None:
            valid_full = np.isfinite(full_lat) & np.isfinite(full_lon)
            if valid_full.any():
                y_f, x_f = full_lat[valid_full], full_lon[valid_full]
                step = max(1, len(y_f) // 4000)
                ax.scatter(x_f[::step], y_f[::step], s=0.25, alpha=0.35,
                           color="lightgrey", rasterized=True, zorder=1,
                           transform=data_crs, label="AGRI full disk")
                # 全圆盘轮廓
                _draw_disk_outline(ax, full_lat, full_lon,
                                   color="lightgrey", lw=1.0, linestyle="--", alpha=0.5,
                                   label=None, transform=data_crs)

        # ── 收紧后保留区域轮廓（蓝色）──
        lat, lon = agri["lat"], agri["lon"]
        valid_agri = np.isfinite(lat) & np.isfinite(lon)
        if valid_agri.any():
            _draw_disk_outline(ax, lat, lon, color="royalblue", lw=2.2,
                               label="Retained region", transform=data_crs)

        # ── MODIS 条带（按文件着色）──
        swath_colors = plt.cm.tab10(np.linspace(0, 1, max(len(modis_bounds), 1)))
        for i, mb in enumerate(modis_bounds):
            olat = mb.get("outline_lat", np.array([]))
            olon = mb.get("outline_lon", np.array([]))
            inside = mb.get("border_in_agri", None)
            status = "IN" if inside else ("OUT" if inside is False else "?")
            color = swath_colors[i]
            short_name = mb.get("file", "MODIS")[:35]

            if len(olat) > 2:
                ax.fill(olon, olat, alpha=0.10, color=color, zorder=2,
                        transform=data_crs)
                ax.plot(np.append(olon, olon[0]), np.append(olat, olat[0]),
                        color=color, lw=1.6, linestyle="--", alpha=0.85, zorder=3,
                        transform=data_crs, label=f"[{status}] {short_name}")

        # ── 视图范围：以星下点为中心 ±85° ──
        ax.set_extent([-85, 85, -85, 85], crs=map_crs)

        # ── 标注 ──
        gl = ax.gridlines(draw_labels=True, alpha=0.35, linestyle="--", linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # ── 标题 ──
        all_in = all(mb.get("border_in_agri", False) for mb in modis_bounds) if modis_bounds else False
        any_out = any(mb.get("border_in_agri") is False for mb in modis_bounds) if modis_bounds else False
        verdict = "ALL MODIS INSIDE" if all_in else ("SOME OUTSIDE" if any_out else "UNKNOWN")
        n_clp = int(np.isfinite(labels["CLP"]).sum())
        n_total = int(valid_agri.sum()) if valid_agri.any() else 1
        n_cer = int(np.isfinite(labels["CER"]).sum())
        n_cot = int(np.isfinite(labels["COT"]).sum())
        n_cth = int(np.isfinite(labels["CTH"]).sum())
        ax.set_title(
            f"MODIS -> AGRI  Geo Verification\n"
            f"{agri_dt:%Y-%m-%d %H:%M} UTC  |  "
            f"{len(modis_bounds)} MODIS granule(s)  |  {verdict}  |  "
            f"CLP: {n_clp}/{n_total} ({100.*n_clp/max(n_total,1):.1f}%)  |  "
            f"CER: {n_cer}  COT: {n_cot}  CTH: {n_cth}",
            fontsize=12, fontweight="bold")

        ax.legend(loc="upper left", fontsize=6.5, markerscale=2, ncol=1,
                  framealpha=0.85)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Geo figure saved -> %s", save_path)
    except Exception as exc:
        log.warning("Geo figure failed for %s: %s",
                    agri_dt.strftime("%Y%m%d_%H%M%S") if agri_dt else "unknown", exc)


def _draw_disk_outline(ax, lat, lon, **kwargs):
    """画出有效像元外轮廓（极角分箱近似凸包）。

    AGRI 全圆盘经度在 [0,360] 下为连续区间 ~[24.1°, 185.3°]，
    但 _derive_latlon 输出经 _wrap_lon 转成了 [-180,180]，
    导致圆盘在 ±180° 日期变更线处被撕裂。
    因此内部计算统一转到 [0,360]（连续），绘图前再转回 [-180,180]
    交给 cartopy PlateCarree transform 处理。
    """
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 3:
        return
    y = lat[valid]
    x_raw = lon[valid]

    # ── 转到 [0, 360]：AGRI 圆盘在此范围连续 ──
    x_360 = np.where(x_raw < 0, x_raw + 360.0, x_raw)

    center_lat = np.median(y)
    center_lon_360 = np.median(x_360)

    # dlon 在 [0,360] 下连续（约 -80.6° ~ +80.6°），无需 wrap
    dlon = x_360 - center_lon_360
    angles = np.arctan2(y - center_lat, dlon)
    # 转到 [0, 2π)：atan2 分支切割在 ±π（西边缘赤道），
    # +2π 后该处变为 angle=π，整个圆盘轮廓角度连续
    angles_2pi = np.where(angles < 0, angles + 2.0 * np.pi, angles)

    n_bins = 72
    bins = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    hull_lat, hull_lon = [], []
    for i in range(n_bins):
        mask = (angles_2pi >= bins[i]) & (angles_2pi < bins[i + 1])
        if not mask.any():
            continue
        dist = np.sqrt((y[mask] - center_lat) ** 2 + (x_360[mask] - center_lon_360) ** 2)
        idx = np.argmax(dist)
        hull_lat.append(y[mask][idx])
        hull_lon.append(x_360[mask][idx])

    if len(hull_lat) < 3:
        return

    hull_lat = np.array(hull_lat)
    hull_lon_360 = np.array(hull_lon)

    # 按角度排序闭合路径（仍在 [0,2π) 下，连续无跳变）
    hull_dlon = hull_lon_360 - center_lon_360
    hull_angles = np.arctan2(hull_lat - center_lat, hull_dlon)
    hull_angles_2pi = np.where(hull_angles < 0, hull_angles + 2.0 * np.pi, hull_angles)
    order = np.argsort(hull_angles_2pi)

    # 转回 [-180, 180] 供 cartopy PlateCarree data transform
    hull_lon_plot = np.where(hull_lon_360 > 180.0, hull_lon_360 - 360.0, hull_lon_360)

    plot_lon = np.append(hull_lon_plot[order], hull_lon_plot[order[0]])
    plot_lat = np.append(hull_lat[order], hull_lat[order[0]])
    ax.plot(plot_lon, plot_lat, **kwargs)


def fuse_day(
    agri_day_dir: Path,
    modis_day_dir: Path,
    out_dir: Path,
    myd03_day_dir: Path = None,
    mode: str = "train",
    overwrite: bool = False,
    n_workers: int = fc.N_FUSION_WORKERS,
    enable_qc_diagnostics: bool = fc.ENABLE_QC_DIAGNOSTICS,
    qc_diagnostics_dir: Path = Path(fc.QC_DIAGNOSTICS_DIR),
) -> int:
    agri_files = sorted([
        p for p in list(agri_day_dir.glob("*.HDF")) + list(agri_day_dir.glob("*.hdf"))
        if "_FDI-_" in p.name
    ])
    modis_files = sorted(
        list(modis_day_dir.glob("MYD06*.hdf")) +
        list(modis_day_dir.glob("MYD06*.HDF"))
    )
    myd03_files = []
    if myd03_day_dir is not None and myd03_day_dir.is_dir():
        myd03_files = sorted(
            list(myd03_day_dir.glob("MYD03*.hdf")) +
            list(myd03_day_dir.glob("MYD03*.HDF"))
        )
    log.info("Day %s | AGRI=%d MYD06=%d MYD03=%d | workers=%d",
             agri_day_dir.name, len(agri_files), len(modis_files), len(myd03_files), n_workers)

    if not agri_files:
        return 0

    tasks = []
    for agri_file in agri_files:
        agri_dt = parse_agri_datetime(agri_file.name)
        if agri_dt is None:
            continue
        matched = find_matching_modis(agri_dt, modis_files)
        if not matched:
            continue
        matched = [(str(f), str(find_matching_myd03(f, myd03_files) or "")) for f in matched]
        out_name = f"AGRI_MYD06_{agri_dt:%Y%m%d_%H%M%S}.h5"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            continue
        tasks.append((str(agri_file), matched, str(out_path), mode, bool(enable_qc_diagnostics)))

    if not tasks:
        log.info("Day %s - no tasks", agri_day_dir.name)
        return 0

    log.info("Day %s - submitting %d tasks", agri_day_dir.name, len(tasks))
    success = 0
    diagnostic_rows = []

    if n_workers <= 1:
        for args in tasks:
            ok, op, msg, diag = _unpack_scene_result(_fuse_one_scene(*args))
            if diag is not None:
                diagnostic_rows.append(diag)
            if ok:
                success += 1
            else:
                log.debug("Skip %s: %s", Path(args[2]).name, msg[:200])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fuse_one_scene, *t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    ok, op, msg, diag = _unpack_scene_result(fut.result())
                except Exception as exc:
                    ok, op, msg, diag = False, task[2], str(exc), None
                if diag is not None:
                    diagnostic_rows.append(diag)
                if ok:
                    success += 1
                else:
                    log.debug("Skip %s: %s", Path(task[2]).name, msg[:200])

    log.info("Day %s - %d/%d ok", agri_day_dir.name, success, len(tasks))
    if enable_qc_diagnostics:
        _write_qc_diagnostics(diagnostic_rows, Path(qc_diagnostics_dir))
    return success


# compat wrapper called by main.py's stage_fuse
def fuse_day_compat(agri_day, modis_day, out_sub, overwrite=False, max_qc=3):  # noqa: ARG001
    parts = {p.lower() for p in out_sub.parts}
    mode = "val" if ("val" in parts or "valid" in parts) else ("test" if "test" in parts else "train")
    myd03_day = cfg.MYD03_ROOT / agri_day.name
    return fuse_day(agri_day, modis_day, out_sub, mode=mode,
                    myd03_day_dir=myd03_day,
                    overwrite=overwrite)


def _setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )


def main():
    _setup_logging(cfg.LOG_LEVEL)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    choices=["train","val","test"], default="train")
    parser.add_argument("--day",      default=None)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--max_qc",  type=int, default=3)
    parser.add_argument("--workers", type=int, default=fc.N_FUSION_WORKERS)
    parser.add_argument("--enable-qc-diagnostics", action="store_true", default=None)
    parser.add_argument("--qc-diagnostics-dir", default=fc.QC_DIAGNOSTICS_DIR)
    args = parser.parse_args()

    split_out  = {"train":cfg.PAIRED_TRAIN_DIR,"val":cfg.PAIRED_VAL_DIR,"test":cfg.PAIRED_TEST_DIR}[args.split]
    dates      = {"train":cfg.TRAIN_DATES,"val":cfg.VAL_DATES,"test":cfg.TEST_DATES}[args.split]
    if args.day:
        dates = [args.day]

    agri_days  = find_day_folders(cfg.AGRI_ROOT, dates)
    modis_days = {d.name: d for d in find_day_folders(cfg.MODIS_ROOT, dates)}
    myd03_days = {d.name: d for d in find_day_folders(cfg.MYD03_ROOT, dates)}

    total = 0
    qc_diag_enabled = (
        fc.ENABLE_QC_DIAGNOSTICS
        if args.enable_qc_diagnostics is None
        else args.enable_qc_diagnostics
    )
    qc_diag_dir = Path(args.qc_diagnostics_dir)
    if qc_diag_enabled:
        _reset_qc_diagnostics(qc_diag_dir)
    for agri_day in agri_days:
        modis_day = modis_days.get(agri_day.name)
        if modis_day is None:
            log.warning("No MODIS for %s", agri_day.name)
            continue
        myd03_day = myd03_days.get(agri_day.name)
        if myd03_day is None:
            log.warning("No MYD03 for %s; fallback to MYD06 5km geo", agri_day.name)
        total += fuse_day(agri_day, modis_day, split_out / agri_day.name,
                          myd03_day_dir=myd03_day,
                          mode=args.split, overwrite=args.overwrite,
                          n_workers=args.workers,
                          enable_qc_diagnostics=qc_diag_enabled,
                          qc_diagnostics_dir=qc_diag_dir)

    log.info("Fusion done - %d files total", total)


if __name__ == "__main__":
    main()
