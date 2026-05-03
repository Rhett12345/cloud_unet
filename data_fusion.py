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
from fusion_core import aggregate_modis_to_agri, check_modis_in_agri_disk
from fusion_io import (
    apply_quality_filter, find_day_folders, find_matching_modis, find_matching_myd03,
    parse_agri_datetime, parse_modis_datetime,
    read_agri_scene, read_myd06,
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

        modis_list = []
        myd06_names = []
        myd03_names = []
        for item in modis_files:
            if isinstance(item, (list, tuple)):
                mf = Path(item[0])
                myd03_file = Path(item[1]) if len(item) > 1 and item[1] else None
            else:
                mf = Path(item)
                myd03_file = None
            myd06_names.append(mf.name)
            if myd03_file is not None:
                myd03_names.append(myd03_file.name)
            m = read_myd06(mf, agri_dt=agri_dt, myd03_file=myd03_file)
            if m is None:
                continue

            # 检查 MODIS 条带是否完整落入 AGRI 圆盘
            modis_lat = m.get("lat_1km") if m.get("lat_1km") is not None else m.get("lat_5km")
            modis_lon = m.get("lon_1km") if m.get("lon_1km") is not None else m.get("lon_5km")
            if modis_lat is None or modis_lon is None:
                continue
            if not check_modis_in_agri_disk(modis_lat, modis_lon, agri["lat"], agri["lon"]):
                continue

            mdt = parse_modis_datetime(mf.name)
            if mdt is None:
                continue
            m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
            m["_file"] = mf.name
            modis_list.append(m)

        if not modis_list:
            return False, out_path, "No MYD06 after reading", diag_row

        # 收集 MODIS 条带边界框（用于地理可视化验证）
        modis_bounds = []
        for m in modis_list:
            mlat = m.get("lat_1km") if m.get("lat_1km") is not None else m.get("lat_5km")
            mlon = m.get("lon_1km") if m.get("lon_1km") is not None else m.get("lon_5km")
            if mlat is not None and mlon is not None:
                valid = np.isfinite(mlat) & np.isfinite(mlon)
                if valid.any():
                    modis_bounds.append({
                        "lat_min": float(mlat[valid].min()),
                        "lat_max": float(mlat[valid].max()),
                        "lon_min": float(mlon[valid].min()),
                        "lon_max": float(mlon[valid].max()),
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
                             out.with_name(out.stem + "_geo.png"))
            return True, out_path, f"OK samples={n_s}", diag_row
        else:
            write_full_disk_hdf5(out, agri, labels, agri_dt)
            _make_geo_figure(agri, labels, agri_dt, modis_bounds,
                             out.with_name(out.stem + "_geo.png"))
            return True, out_path, "OK full_disk", diag_row

    except Exception:
        return False, out_path, f"Exception:\n{traceback.format_exc()}", diag_row


def _make_geo_figure(agri, labels, agri_dt, modis_bounds, save_path):
    """生成地理定位验证图：AGRI 圆盘边界 + MODIS CLP 覆盖 + 经纬度网格。

    用于在融合完成后快速排查 MODIS 数据是否正确完整落入 AGRI 圆盘。
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        lat = agri["lat"]
        lon = agri["lon"]
        clp = labels["CLP"]

        clp_names = getattr(cfg, "CLP_CLASS_NAMES", ["Clear", "Water", "Ice"])
        clp_cmap = ListedColormap(["#eeeeee", "#4c78a8", "#f58518"][:len(clp_names)])
        clp_norm = BoundaryNorm(np.arange(len(clp_names) + 1) - 0.5, len(clp_names))
        clp_cmap.set_bad("white")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax_map, ax_stats = axes

        # ── 左图：经纬度空间覆盖 ──
        # AGRI 有效像元散点（下采样，灰色背景参考）
        valid_agri = np.isfinite(lat) & np.isfinite(lon)
        if valid_agri.any():
            y_agri, x_agri = lat[valid_agri], lon[valid_agri]
            step = max(1, len(y_agri) // 6000)
            ax_map.scatter(x_agri[::step], y_agri[::step], s=0.2, alpha=0.25,
                           color="lightgrey", rasterized=True, zorder=1)

        # AGRI 圆盘边界线
        _draw_disk_outline(ax_map, lat, lon, color="royalblue", lw=1.8, label="AGRI disk boundary")

        # MODIS CLP 覆盖（按相态着色）
        clp_valid = np.isfinite(clp) & (clp >= 0) & (clp < len(clp_names))
        if clp_valid.any():
            y_c, x_c = lat[clp_valid], lon[clp_valid]
            step_c = max(1, len(y_c) // 8000)
            ax_map.scatter(x_c[::step_c], y_c[::step_c], c=clp[clp_valid][::step_c],
                           cmap=clp_cmap, norm=clp_norm, s=1.2, alpha=0.8,
                           rasterized=True, zorder=3)

        # MODIS 条带边界框（来自原始 MODIS 经纬度范围）
        colors_swath = plt.cm.tab10(np.linspace(0, 1, max(len(modis_bounds), 1)))
        for i, mb in enumerate(modis_bounds):
            lat_min, lat_max = mb["lat_min"], mb["lat_max"]
            lon_min, lon_max = mb["lon_min"], mb["lon_max"]
            rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                 fill=False, edgecolor=colors_swath[i], lw=1.2,
                                 linestyle="--", alpha=0.7, zorder=2,
                                 label=f"MODIS swath {i+1}" if i == 0 else None)
            ax_map.add_patch(rect)

        ax_map.set_xlabel("Longitude (°)", fontsize=10)
        ax_map.set_ylabel("Latitude (°)", fontsize=10)
        ax_map.set_title(f"MODIS→AGRI Geo Coverage | {agri_dt:%Y-%m-%d %H:%M:%S} UTC", fontsize=11)
        ax_map.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)
        ax_map.legend(loc="lower right", fontsize=7, markerscale=4)
        ax_map.set_aspect("equal")

        # ── 右图：覆盖统计 ──
        n_total = int(np.isfinite(lat).sum()) if valid_agri.any() else 0
        n_clp = int(clp_valid.sum())
        coverage = 100.0 * n_clp / max(n_total, 1)

        # 相态分布
        if clp_valid.any():
            clp_int = clp[clp_valid].astype(int)
            counts = np.bincount(clp_int, minlength=len(clp_names))[:len(clp_names)]
        else:
            counts = np.zeros(len(clp_names), dtype=int)

        x_pos = np.arange(len(clp_names))
        bars = ax_stats.bar(x_pos, counts, color=clp_cmap.colors, edgecolor="grey", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            ax_stats.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.max(), 1) * 0.03,
                          str(cnt), ha="center", fontsize=9, fontweight="bold")
        ax_stats.set_xticks(x_pos)
        ax_stats.set_xticklabels(clp_names, fontsize=10)
        ax_stats.set_ylabel("Pixel Count", fontsize=10)
        ax_stats.set_title(f"CLP Phase Distribution\nAGRI valid px={n_total:,}  |  "
                           f"MODIS coverage={coverage:.1f}%  |  cloudy={counts[1:].sum():,}", fontsize=10)

        # 附加文字信息
        info_lines = [
            f"AGRI datetime: {agri_dt:%Y-%m-%d %H:%M:%S}",
            f"AGRI valid pixels: {n_total:,}",
            f"MODIS CLP valid: {n_clp:,} ({coverage:.1f}%)",
            f"Clear px: {counts[0]:,}",
            f"Water px: {counts[1]:,}",
            f"Ice px: {counts[2]:,}",
        ]
        if modis_bounds:
            info_lines.append(f"MODIS granules: {len(modis_bounds)}")
            for i, mb in enumerate(modis_bounds):
                info_lines.append(f"  swath {i+1}: lon=[{mb['lon_min']:.2f}, {mb['lon_max']:.2f}] "
                                  f"lat=[{mb['lat_min']:.2f}, {mb['lat_max']:.2f}]")
        ax_stats.text(1.05, 0.5, "\n".join(info_lines), transform=ax_stats.transAxes,
                      fontsize=7.5, fontfamily="monospace", verticalalignment="center",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8))

        fig.suptitle(f"Fusion Geo Verification — {agri_dt:%Y%m%d_%H%M%S}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        log.info("Geo figure saved -> %s", save_path)
    except Exception as exc:
        log.warning("Geo figure failed for %s: %s",
                    agri_dt.strftime("%Y%m%d_%H%M%S") if agri_dt else "unknown", exc)


def _draw_disk_outline(ax, lat, lon, **kwargs):
    """在 ax 上画出 AGRI 有效像元的外轮廓（极角分箱近似凸包）。"""
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 3:
        return
    y, x = lat[valid], lon[valid]
    center_lat, center_lon = np.median(y), np.median(x)
    angles = np.arctan2(y - center_lat, x - center_lon)
    n_bins = 72
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hull_lat, hull_lon = [], []
    for i in range(n_bins):
        mask = (angles >= bins[i]) & (angles < bins[i + 1])
        if not mask.any():
            continue
        dist = np.sqrt((y[mask] - center_lat) ** 2 + (x[mask] - center_lon) ** 2)
        idx = np.argmax(dist)
        hull_lat.append(y[mask][idx])
        hull_lon.append(x[mask][idx])
    if len(hull_lat) < 3:
        return
    hull_lat = np.array(hull_lat)
    hull_lon = np.array(hull_lon)
    order = np.argsort(np.arctan2(hull_lat - center_lat, hull_lon - center_lon))
    ax.plot(np.append(hull_lon[order], hull_lon[order[0]]),
            np.append(hull_lat[order], hull_lat[order[0]]), **kwargs)


def _make_qc_figure(out_h5: Path, qc_path: Path):
    """从已写出的 HDF5 生成 QC 诊断图。"""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        import h5py

        with h5py.File(out_h5, "r") as f:
            agri_dt_str = f.attrs.get("agri_datetime", "")
            if "Samples" in f and "max_time_diff_min" in f.get("Samples", {}):
                dt_arr = f["Samples/max_time_diff_min"][()]
                wt_arr = f["Samples/mean_sample_weight"][()] if "mean_sample_weight" in f["Samples"] else None
                n = int(f.attrs.get("num_samples", 0))

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].hist(dt_arr[np.isfinite(dt_arr)], bins=20, color="steelblue", edgecolor="white")
                axes[0].set_xlabel("Max time diff per patch (min)")
                axes[0].set_ylabel("Count")
                axes[0].set_title(f"Time Diff Distribution\n{agri_dt_str}")

                if wt_arr is not None and len(wt_arr) > 0:
                    axes[1].hist(wt_arr[np.isfinite(wt_arr)], bins=20, color="tomato", edgecolor="white")
                    axes[1].set_xlabel("Mean sample weight per patch")
                    axes[1].set_ylabel("Count")
                    axes[1].set_title(f"Weight Distribution | n_patches={n}")

                fig.suptitle(f"Fusion QC - {agri_dt_str}", fontsize=12, fontweight="bold")
                fig.tight_layout()

            elif "Labels" in f:
                CLP = f["Labels/CLP"][()]
                CTH = f["Labels/CTH"][()]
                BT_keys = sorted(f["AGRI/BT"].keys())
                bt0 = f[f"AGRI/BT/{BT_keys[0]}"][()]

                clp_names = list(getattr(cfg, "CLP_CLASS_NAMES", ["Clear", "Water", "Ice"]))
                clp_cmap = ListedColormap(["white", "deepskyblue", "red"][:len(clp_names)])
                clp_ticks = list(range(len(clp_names)))
                clp_norm = BoundaryNorm(np.arange(len(clp_names) + 1) - 0.5, clp_cmap.N)
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                fbt = bt0[np.isfinite(bt0)]
                im = axes[0,0].imshow(bt0, cmap="RdYlBu_r",
                    vmin=np.percentile(fbt,2) if fbt.size else 200,
                    vmax=np.percentile(fbt,98) if fbt.size else 310,
                    aspect="auto", interpolation="none")
                plt.colorbar(im, ax=axes[0,0], label="BT(K)")
                axes[0,0].set_title(f"AGRI BT ch{cfg.AGRI_BT_CHANNEL_INDICES[0]+1}")

                cmap_c = plt.cm.viridis_r.copy(); cmap_c.set_bad("lightgrey")
                fcth = CTH[np.isfinite(CTH)]
                im2 = axes[0,1].imshow(np.where(np.isfinite(CTH),CTH,np.nan),
                    cmap=cmap_c, vmin=0, vmax=np.percentile(fcth,98) if fcth.size else 15000,
                    aspect="auto", interpolation="none")
                plt.colorbar(im2, ax=axes[0,1], label="CTH(m)")
                axes[0,1].set_title(f"MODIS CTH | cov={100*np.isfinite(CTH).mean():.1f}%")

                cmap_p = clp_cmap.copy(); cmap_p.set_bad("lightgrey")
                im3 = axes[1,0].imshow(np.where(np.isfinite(CLP),CLP,np.nan),
                    cmap=cmap_p, norm=clp_norm, aspect="auto", interpolation="none")
                cb = plt.colorbar(im3, ax=axes[1,0], ticks=clp_ticks)
                cb.ax.set_yticklabels(clp_names, fontsize=8)
                axes[1,0].set_title("MODIS Phase (多数表决)")

                cloudy = np.isfinite(CLP) & (CLP>0) & np.isfinite(CTH)
                if cloudy.any():
                    sc = np.random.choice(np.where(cloudy.ravel())[0],
                                          min(5000, cloudy.sum()), replace=False)
                    axes[1,1].scatter(bt0.ravel()[sc], CTH.ravel()[sc]/1000,
                        s=2, alpha=0.3, c=CTH.ravel()[sc], cmap="viridis_r", rasterized=True)
                    r = np.corrcoef(bt0.ravel()[sc], CTH.ravel()[sc])[0,1]
                    axes[1,1].set_title(f"BT vs CTH (r={r:.3f})")
                    axes[1,1].set_xlabel("BT(K)"); axes[1,1].set_ylabel("CTH(km)")
                else:
                    axes[1,1].text(0.5,0.5,"No cloudy",ha="center",va="center",
                                   transform=axes[1,1].transAxes)

                fig.suptitle(f"Fusion QC - {agri_dt_str}", fontsize=12, fontweight="bold")
                fig.tight_layout()
            else:
                return

        qc_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(qc_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        log.info("QC saved -> %s", qc_path)
    except Exception as exc:
        log.warning("QC figure failed for %s: %s", out_h5.name, exc)


def fuse_day(
    agri_day_dir: Path,
    modis_day_dir: Path,
    out_dir: Path,
    myd03_day_dir: Path = None,
    mode: str = "train",
    overwrite: bool = False,
    max_qc: int = 3,
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
    success, qc_count = 0, 0
    diagnostic_rows = []

    if n_workers <= 1:
        for args in tasks:
            ok, op, msg, diag = _unpack_scene_result(_fuse_one_scene(*args))
            if diag is not None:
                diagnostic_rows.append(diag)
            if ok:
                success += 1
                if qc_count < max_qc:
                    _make_qc_figure(Path(op), Path(op).with_name(Path(op).stem + "_qc.png"))
                    qc_count += 1
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
                    if qc_count < max_qc:
                        _make_qc_figure(Path(op), Path(op).with_name(Path(op).stem + "_qc.png"))
                        qc_count += 1
                else:
                    log.debug("Skip %s: %s", Path(task[2]).name, msg[:200])

    log.info("Day %s - %d/%d ok | %d QC figs", agri_day_dir.name, success, len(tasks), qc_count)
    if enable_qc_diagnostics:
        _write_qc_diagnostics(diagnostic_rows, Path(qc_diagnostics_dir))
    return success


# compat wrapper called by main.py's stage_fuse
def fuse_day_compat(agri_day, modis_day, out_sub, overwrite=False, max_qc=3):
    parts = {p.lower() for p in out_sub.parts}
    mode = "val" if ("val" in parts or "valid" in parts) else ("test" if "test" in parts else "train")
    myd03_day = cfg.MYD03_ROOT / agri_day.name
    return fuse_day(agri_day, modis_day, out_sub, mode=mode,
                    myd03_day_dir=myd03_day,
                    overwrite=overwrite, max_qc=max_qc)


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
                          max_qc=args.max_qc, n_workers=args.workers,
                          enable_qc_diagnostics=qc_diag_enabled,
                          qc_diagnostics_dir=qc_diag_dir)

    log.info("Fusion done - %d files total", total)


if __name__ == "__main__":
    main()
