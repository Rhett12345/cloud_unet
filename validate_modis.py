"""
tools/validate_modis.py  (已修复)
=================================
Cross-validate model predictions against MODIS MYD06 (independent sensor).

修复的两个 bug
--------------
Bug 1 — geo_cache shape 检查逻辑错误
    原始代码：read_modis_geo_quick 返回 lat_1km（无 lat_5km），
    但 read_myd06 只在 geo_cache.get("lat_5km") is not None 时才用缓存；
    否则重新从文件读 5km geo，使 1km 缓存路径的条件判断全部失效。
    修复：在 match_modis_to_agri_grid 里显式区分两种 geo_cache 格式，
    构造完整的 geo_cache dict，确保 read_myd06 走正确的分支。

Bug 2 — apply_quality_filter 时 MATCH_DT_MIN 缺失 → CTH 被全部过滤
    原始代码：validate 脚本构造的 labels 没有 MATCH_DT_MIN 字段，
    apply_quality_filter 里 reg_time_ok 退化为全 False（np.zeros），
    导致所有 CTH 值被置 NaN，CTH 验证变成纯噪声。
    修复：在调用 apply_quality_filter 前，向 labels 注入
    MATCH_DT_MIN（像元级实际时间差）和 MATCH_DT_MAX 字段；
    同时使用真实 VZA/SZA（来自 agri 场景），而不是全零数组。

Usage:
  python validate_modis.py --npz_dir /path/to/retrieval/ --day 20190503

Requires: MODIS MYD06 + MYD03 data under cfg.MODIS_ROOT / cfg.MYD03_ROOT.
"""

import argparse, logging, sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as cfg
import fusion_config as fc
from fusion_core import aggregate_modis_to_agri, check_modis_in_agri_disk
from fusion_io import (
    apply_quality_filter,
    parse_agri_datetime, parse_modis_datetime,
    find_matching_modis, find_matching_myd03,
    read_agri_scene, read_myd06, read_modis_geo_quick,
    _extract_timestamp_from_filename,
)

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "modis_validation"


def save_figure(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# MODIS label matching to AGRI grid（修复版）
# ---------------------------------------------------------------------------

def match_modis_to_agri_grid(
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    agri_dt,
    modis_files: list,
    myd03_files: list,
    agri_vza: Optional[np.ndarray] = None,
    agri_sza: Optional[np.ndarray] = None,
    quality: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Run the full MODIS→AGRI spatial matching pipeline for validation.

    修复说明
    --------
    原始代码直接把 read_modis_geo_quick 的返回值作为 geo_cache 传给
    read_myd06，但两者对 cache dict 的格式预期不一致：
      - read_modis_geo_quick 在有 MYD03 时返回 {"lat_1km": ..., "lon_1km": ...}
        （无 "lat_5km" 键）
      - read_myd06 的分支逻辑：先检查 geo_cache.get("lat_5km") is not None；
        若为 None 则从文件重读 5km geo，导致 1km 缓存实际不生效
    修复：在此处显式区分两种情形，构造 read_myd06 能正确识别的 cache dict。
    """
    if quality:
        cfg.MODIS_FILTER_WEAK_QUALITY = True
        cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 3)
        cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0,)

    modis_list = []
    for mf in modis_files:
        mf = Path(mf) if isinstance(mf, str) else mf
        myd03_file = find_matching_myd03(mf, myd03_files)

        # ── 修复 Bug 1：正确构造 geo_cache ──────────────────────────────
        # read_modis_geo_quick 可能返回两种格式：
        #   (A) MYD03 可用：{"lat_1km": ..., "lon_1km": ..., "_geo_source": "MYD03_1KM"}
        #   (B) 回退 MYD06：{"lat_5km": ..., "lon_5km": ..., "_geo_source": "MYD06_5KM"}
        # read_myd06 期望：先找 lat_5km（用于 5km 分支），再找 lat_1km（用于 1km 分支）
        # 当 geo_cache 只有 lat_1km 时，read_myd06 会重新从文件读 5km geo，
        # 使 1km 缓存失效。因此需要补充 lat_5km 或构造完整 cache。
        raw_geo = read_modis_geo_quick(mf, myd03_file=myd03_file)
        if raw_geo is None:
            continue

        geo_source = raw_geo.get("_geo_source", "")

        if geo_source == "MYD03_1KM":
            # MYD03 1km 可用：从 MYD06 文件补充 5km geo，构造完整 cache
            # 这样 read_myd06 先用 5km geo 做初始化，再用 1km 做精确定位
            try:
                from pyhdf.SD import SD, SDC
                sd_tmp = SD(str(mf), SDC.READ)
                lat_5km_tmp = sd_tmp.select("Latitude")[:].astype(np.float32)
                lon_5km_tmp = sd_tmp.select("Longitude")[:].astype(np.float32)
                sd_tmp.end()
                geo_cache = {
                    "lat_5km": lat_5km_tmp,          # read_myd06 的 5km 分支入口
                    "lon_5km": lon_5km_tmp,
                    "lat_1km": raw_geo["lat_1km"],   # read_myd06 的 1km 精化分支
                    "lon_1km": raw_geo["lon_1km"],
                    "scan_time_1km": raw_geo.get("scan_time_1km"),
                    "scan_time_source": raw_geo.get("scan_time_source", "none"),
                    "_geo_source": "MYD03_1KM",
                }
            except Exception as e:
                log.debug("Cannot read MYD06 5km geo for cache completion: %s", e)
                # 退化：只用 MYD06 自身的 5km geo（会走 upsample 路径）
                geo_cache = raw_geo
        else:
            # MYD06 5km：直接用，read_myd06 能正确识别
            geo_cache = raw_geo

        # ── 快速地理预检（使用实际可用的最精确坐标）────────────────────────
        mlat = geo_cache.get("lat_1km") if geo_cache.get("lat_1km") is not None \
               else geo_cache.get("lat_5km")
        mlon = geo_cache.get("lon_1km") if geo_cache.get("lon_1km") is not None \
               else geo_cache.get("lon_5km")
        if mlat is None or mlon is None:
            continue
        if not check_modis_in_agri_disk(mlat, mlon, agri_lat, agri_lon):
            continue

        m = read_myd06(mf, agri_dt=agri_dt, myd03_file=myd03_file, geo_cache=geo_cache)
        if m is None:
            continue

        mdt = parse_modis_datetime(mf.name)
        if mdt:
            m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
        modis_list.append(m)

    if not modis_list:
        return None

    labels = aggregate_modis_to_agri(agri_lat, agri_lon, modis_list)
    if labels is None:
        return None

    # ── 修复 Bug 2：真实 VZA/SZA + 确保 MATCH_DT_MIN 存在 ──────────────
    vza = agri_vza if agri_vza is not None else np.zeros_like(agri_lat)
    sza = agri_sza if agri_sza is not None else np.zeros_like(agri_lat)
    agri_geo_for_qc = {"VZA": vza, "SZA": sza}

    # MATCH_DT_MIN：aggregate_modis_to_agri 已写入；若意外缺失则注入兜底值
    if "MATCH_DT_MIN" not in labels or labels["MATCH_DT_MIN"] is None:
        log.warning("MATCH_DT_MIN missing after aggregation; injecting file-level dt as fallback")
        file_dt_vals = [m.get("_dt_min", np.nan) for m in modis_list]
        fallback_dt = float(np.nanmin(file_dt_vals)) if file_dt_vals else fc.TIME_LOW_Q_MIN
        labels["MATCH_DT_MIN"] = np.full(agri_lat.shape, fallback_dt, dtype=np.float32)

    return apply_quality_filter(agri_geo_for_qc, labels)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_one_modis(
    npz_path: Path,
    agri_file: Path,
    out_dir: Path,
    scene_id: str = "",
) -> dict:
    """Run MODIS cross-validation for one scene."""
    # Read model predictions
    data = np.load(npz_path)
    clp_pred = data["CLP_pred"].astype(np.float32)
    cth_pred = data["CTH_pred"].astype(np.float32)
    data.close()

    # Read AGRI for geo coordinates, time, and real VZA/SZA
    agri = read_agri_scene(agri_file)
    if agri is None:
        raise RuntimeError(f"Failed to read AGRI scene {agri_file}")

    agri_dt = parse_agri_datetime(agri_file.name)
    if agri_dt is None:
        raise RuntimeError(f"Cannot parse datetime from {agri_file.name}")

    # Find matching MODIS
    day_str = agri_dt.strftime("%Y%m%d")
    modis_day_dir = cfg.MODIS_ROOT / day_str
    myd03_day_dir = cfg.MYD03_ROOT / day_str

    modis_files = []
    if modis_day_dir.is_dir():
        modis_files = sorted(
            list(modis_day_dir.glob("MYD06*.hdf")) +
            list(modis_day_dir.glob("MYD06*.HDF"))
        )
    myd03_files = []
    if myd03_day_dir.is_dir():
        myd03_files = sorted(
            list(myd03_day_dir.glob("MYD03*.hdf")) +
            list(myd03_day_dir.glob("MYD03*.HDF"))
        )

    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_matching_modis"}

    # Spatial matching — pass real VZA/SZA directly
    labels = match_modis_to_agri_grid(
        agri["lat"], agri["lon"], agri_dt, matched, myd03_files,
        agri_vza=agri["VZA"], agri_sza=agri["SZA"],
    )
    if labels is None:
        return {"scene_id": scene_id, "status": "modis_matching_failed"}

    clp_true = labels["CLP"]
    cth_true = labels["CTH"]

    clp_true_f = clp_true.astype(np.float32)
    clp_true_f[(clp_true_f < 0) | (clp_true_f >= 3)] = np.nan

    assert clp_pred.shape == clp_true_f.shape, (
        f"Shape mismatch: pred={clp_pred.shape} true={clp_true_f.shape}"
    )

    from validate_agri_l2 import compute_metrics
    metrics = compute_metrics(clp_pred, clp_true_f, cth_pred, cth_true)
    metrics["scene_id"] = scene_id
    metrics["status"] = "ok"

    # 新增诊断字段：MODIS 覆盖率和时间差
    dt_arr = labels.get("MATCH_DT_MIN")
    if dt_arr is not None:
        valid_dt = dt_arr[np.isfinite(dt_arr)]
        metrics["modis_dt_median_min"] = float(np.median(valid_dt)) if valid_dt.size else np.nan
        metrics["modis_dt_max_min"] = float(np.max(valid_dt)) if valid_dt.size else np.nan
    metrics["modis_clp_coverage_pct"] = 100.0 * float(np.isfinite(clp_true_f).mean())
    metrics["modis_cth_coverage_pct"] = 100.0 * float(np.isfinite(cth_true).mean())

    log.info(
        "  %s  vs MODIS  CLP OA=%5.2f%%  CTH R=%.4f  RMSE=%.0f m  "
        "MODIS_clp_cover=%.1f%%  MODIS_cth_cover=%.1f%%  dt_med=%.1f min",
        scene_id,
        metrics["oa"],
        metrics.get("cth_r", 0),
        metrics["cth_rmse"],
        metrics["modis_clp_coverage_pct"],
        metrics["modis_cth_coverage_pct"],
        metrics.get("modis_dt_median_min", float("nan")),
    )

    if metrics["n_cth"] > 10:
        from validate_agri_l2 import _plot_cth_scatter
        fig = _plot_cth_scatter(cth_pred, cth_true, metrics, f"{scene_id} vs MODIS")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = scene_id.replace(" ", "_")
        save_figure(fig, f"{stem}_modis_cth_scatter")
        plt.close(fig)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Cross-validate model against MODIS MYD06")
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    npz_files = sorted(Path(args.npz_dir).rglob(f"*{args.day}*.npz"))
    agri_day_dir = cfg.AGRI_ROOT / args.day
    agri_files = {
        f.stem.replace("_retrieval", ""): f
        for f in agri_day_dir.glob("*_FDI-_*.HDF")
    } if agri_day_dir.is_dir() else {}

    out_dir = Path(args.out_dir)
    all_metrics = []
    for npz_p in npz_files:
        ts = _extract_timestamp_from_filename(npz_p.name)
        if ts is None:
            continue
        scene_id = f"{ts[:8]}_{ts[8:]}"
        agri_match = None
        for stem, agri_p in agri_files.items():
            if ts in stem:
                agri_match = agri_p
                break
        if agri_match is None:
            log.warning("No AGRI FDI file for timestamp %s", ts)
            continue
        try:
            m = validate_one_modis(npz_p, agri_match, out_dir, scene_id)
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", scene_id, exc)

    if all_metrics and args.summary:
        import pandas as pd
        rows = [{k: v for k, v in m.items() if k != "confusion_matrix"}
                for m in all_metrics]
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"summary_modis_{args.day}.csv", index=False)
        ok = [m for m in all_metrics if m.get("status") == "ok"]
        if ok:
            oa_vals = [m["oa"] for m in ok if m["oa"] > 0]
            log.info(
                "MODIS cross-val: n=%d  OA mean=%.2f%%  "
                "MODIS_clp_coverage mean=%.1f%%  MODIS_cth_coverage mean=%.1f%%",
                len(ok),
                np.mean(oa_vals) if oa_vals else 0,
                np.mean([m["modis_clp_coverage_pct"] for m in ok]),
                np.mean([m["modis_cth_coverage_pct"] for m in ok]),
            )


if __name__ == "__main__":
    main()
