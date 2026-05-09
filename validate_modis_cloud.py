"""
validate_modis_cloud.py  (已修复)
==================================
Simplified MODIS cross-validation: binary cloud detection + stratified CTH.

修复的两个 bug（与 validate_modis.py 相同根因）
------------------------------------------------
Bug 1 — geo_cache 格式不兼容导致 1km 坐标实际不生效
    read_modis_geo_quick 返回的 dict 在有 MYD03 时只含 lat_1km/lon_1km，
    缺少 lat_5km。read_myd06 优先检查 lat_5km 是否存在：不存在则重读文件，
    导致 1km 缓存的精确坐标实际上从未被用到，全部退化为 5km upsample。
    修复：在此脚本的 match_modis_to_agri_grid 里，当 MYD03 可用时，
    额外从 MYD06 文件读取 5km geo 并补充到 cache dict，
    使 read_myd06 能正确走 "5km初始化 + 1km精化" 的路径。

Bug 2 — labels 缺失 MATCH_DT_MIN → reg_time_ok 全 False → CTH 全部被滤掉
    apply_quality_filter 里 CTH 过滤依赖 MATCH_DT_MIN <= REG_TIME_MAX_MIN，
    若该字段不在 labels 里，reg_time_ok = np.zeros(shape, bool)，
    所有 CTH 像元被置 NaN，分层 CTH 验证退化为纯噪声。
    修复：
      1. aggregate_modis_to_agri 已写入 MATCH_DT_MIN，正常路径无需额外处理；
      2. 兜底：若字段缺失，注入文件级时间差作为保守上限；
      3. 将全零 VZA/SZA 替换为真实 AGRI 场景角度。

Usage:
  # Model vs MODIS
  python validate_modis_cloud.py --npz_dir /path/to/retrieval/ --day 20190505

  # AGRI L2 vs MODIS (baseline)
  python validate_modis_cloud.py --day 20190505 --reference l2
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
    _extract_timestamp_from_filename, _find_matching_l2_file,
)

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "modis_cloud_validation"

CTH_LAYERS = {"Low": (0, 3000), "Mid": (3000, 8000), "High": (8000, 20000)}


def save_figure(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# AGRI L2 readers
# ---------------------------------------------------------------------------

def read_l2_clp_raw(nc_path: Path) -> Optional[np.ndarray]:
    """Read L2 CLP as raw int16 (no remap)."""
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CLP"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.int16)
    ds.close()
    return raw


def read_l2_cth(nc_path: Path) -> Optional[np.ndarray]:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CTH"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.float32)
    ds.close()
    vmin, vmax = cfg.AGRI_L2_CTH_VALID_RANGE
    raw[(raw <= 0) | (raw >= 65500) | ~np.isfinite(raw)] = np.nan
    raw[raw < vmin] = np.nan
    raw[raw > vmax] = np.nan
    return raw


# ---------------------------------------------------------------------------
# Binary cloud mask from CLP
# ---------------------------------------------------------------------------

def to_cloud_mask(clp: np.ndarray, source: str = "agri_l2") -> np.ndarray:
    """Convert CLP array to binary cloud mask: 0=clear, 1=cloudy, NaN=invalid."""
    mask = np.full(clp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(clp)
    if source == "agri_l2_raw":
        mask[valid & (clp == 0)] = 0
        mask[valid & (clp >= 1) & (clp <= 4)] = 1
    else:
        mask[valid & (clp == 0)] = 0
        mask[valid & (clp > 0)] = 1
    return mask


# ---------------------------------------------------------------------------
# MODIS matching（修复版）
# ---------------------------------------------------------------------------

def match_modis_to_agri_grid(
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    agri_dt,
    modis_files: list,
    myd03_files: list,
    agri_vza: Optional[np.ndarray] = None,
    agri_sza: Optional[np.ndarray] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Match MODIS to AGRI grid with strict quality filtering.

    修复 Bug 1：正确构造 geo_cache，使 MYD03 1km 坐标实际生效。
    修复 Bug 2：
      - 使用真实 VZA/SZA（而非全零）供 apply_quality_filter 使用；
      - 确保 MATCH_DT_MIN 存在于 labels 中，避免 CTH 被全部过滤。
    """
    cfg.MODIS_FILTER_WEAK_QUALITY = True
    cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 3)
    cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0,)

    modis_list = []
    for mf in modis_files:
        mf = Path(mf) if isinstance(mf, str) else mf
        m03 = find_matching_myd03(mf, myd03_files)

        # ── 修复 Bug 1：构造完整 geo_cache ─────────────────────────────
        raw_geo = read_modis_geo_quick(mf, myd03_file=m03)
        if raw_geo is None:
            continue

        geo_source = raw_geo.get("_geo_source", "")

        if geo_source == "MYD03_1KM":
            # 有 MYD03 1km：额外补充 MYD06 5km geo，构造完整 cache
            # 原因：read_myd06 先检查 lat_5km 是否存在；若缺失则忽略整个 cache
            try:
                from pyhdf.SD import SD, SDC
                sd_tmp = SD(str(mf), SDC.READ)
                lat_5km_tmp = sd_tmp.select("Latitude")[:].astype(np.float32)
                lon_5km_tmp = sd_tmp.select("Longitude")[:].astype(np.float32)
                sd_tmp.end()
                geo_cache = {
                    "lat_5km": lat_5km_tmp,
                    "lon_5km": lon_5km_tmp,
                    "lat_1km": raw_geo["lat_1km"],
                    "lon_1km": raw_geo["lon_1km"],
                    "scan_time_1km": raw_geo.get("scan_time_1km"),
                    "scan_time_source": raw_geo.get("scan_time_source", "none"),
                    "_geo_source": "MYD03_1KM",
                }
            except Exception as e:
                log.debug("Cannot supplement 5km geo into cache: %s", e)
                geo_cache = raw_geo  # 退化为原始 cache（走 upsample 路径）
        else:
            geo_cache = raw_geo

        # 地理预检
        mlat = geo_cache.get("lat_1km") if geo_cache.get("lat_1km") is not None \
               else geo_cache.get("lat_5km")
        mlon = geo_cache.get("lon_1km") if geo_cache.get("lon_1km") is not None \
               else geo_cache.get("lon_5km")
        if mlat is None or mlon is None:
            continue
        if not check_modis_in_agri_disk(mlat, mlon, agri_lat, agri_lon):
            continue

        m = read_myd06(mf, agri_dt=agri_dt, myd03_file=m03, geo_cache=geo_cache)
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
    # VZA/SZA：使用调用方传入的真实角度；若未提供则用全零（允许所有像元通过）
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
# Metrics
# ---------------------------------------------------------------------------

def cloud_detection_metrics(cmask_ref: np.ndarray, cmask_test: np.ndarray) -> dict:
    """Binary cloud detection metrics."""
    valid = np.isfinite(cmask_ref) & np.isfinite(cmask_test)
    ref = cmask_ref[valid]
    tst = cmask_test[valid]
    n = len(ref)
    if n == 0:
        return {"n_cloud": 0}

    tp = int(((ref == 1) & (tst == 1)).sum())
    tn = int(((ref == 0) & (tst == 0)).sum())
    fp = int(((ref == 0) & (tst == 1)).sum())
    fn = int(((ref == 1) & (tst == 0)).sum())

    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    oa  = (tp + tn) / max(n, 1) * 100
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / max(denom, 1)

    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "pod": pod, "far": far, "oa_cloud": oa, "hss": hss}


def cth_stratified_metrics(
    cth_ref: np.ndarray,
    cth_test: np.ndarray,
    cmask_ref: np.ndarray,
    cmask_test: np.ndarray,
) -> dict:
    """CTH metrics stratified by reference cloud height."""
    cloudy = (
        np.isfinite(cmask_ref) & np.isfinite(cmask_test) &
        (cmask_ref == 1) & (cmask_test == 1) &
        np.isfinite(cth_ref) & np.isfinite(cth_test)
    )
    p, t = cth_ref[cloudy], cth_test[cloudy]

    result = {"n_cth_total": int(cloudy.sum())}
    if result["n_cth_total"] < 10:
        return result

    result["cth_r_all"] = _pearson_r(p, t)
    result["cth_rmse_all"] = float(np.sqrt(np.mean((p - t) ** 2)))
    result["cth_bias_all"] = float(np.mean(t - p))

    for name, (lo, hi) in CTH_LAYERS.items():
        mask = (p >= lo) & (p < hi)
        n_layer = int(mask.sum())
        result[f"n_{name}"] = n_layer
        if n_layer > 10:
            pl, tl = p[mask], t[mask]
            result[f"cth_r_{name}"]        = _pearson_r(pl, tl)
            result[f"cth_rmse_{name}"]     = float(np.sqrt(np.mean((pl - tl) ** 2)))
            result[f"cth_bias_{name}"]     = float(np.mean(tl - pl))
            result[f"cth_mean_ref_{name}"] = float(np.mean(pl))
            result[f"cth_mean_modis_{name}"] = float(np.mean(tl))
        else:
            for k in ["cth_r", "cth_rmse", "cth_bias", "cth_mean_ref", "cth_mean_modis"]:
                result[f"{k}_{name}"] = np.nan

    return result


def _pearson_r(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / max(denom, 1e-12))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_stratified_cth(cth_ref, cth_test, cmask_ref, cmask_test, metrics, scene_id):
    cloudy = (
        np.isfinite(cmask_ref) & np.isfinite(cmask_test) &
        (cmask_ref == 1) & (cmask_test == 1) &
        np.isfinite(cth_ref) & np.isfinite(cth_test)
    )
    p_all, t_all = cth_ref[cloudy], cth_test[cloudy]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    colors = {"Low": "#2E86AB", "Mid": "#A23B72", "High": "#F18F01"}

    for ax, (name, (lo, hi)) in zip(axes[:3], CTH_LAYERS.items()):
        mask = (p_all >= lo) & (p_all < hi)
        pts = p_all[mask], t_all[mask]
        if mask.sum() > 10:
            idx = np.random.choice(mask.sum(), min(5000, mask.sum()), replace=False)
            pts = pts[0][idx], pts[1][idx]
        ax.scatter(pts[1], pts[0], s=0.4, alpha=0.35, color=colors[name], rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{name} ({lo/1000:.0f}–{hi/1000:.0f} km)",
                     fontsize=6.5, fontweight="bold", color=colors[name])
        ax.set_xlabel("MODIS CTH (m)", fontsize=6)
        if name == "Low":
            ax.set_ylabel("Ref CTH (m)", fontsize=6)

    ax = axes[3]; ax.axis("off")
    lines = [f"  {scene_id}", "", "Layer    R      RMSE    Bias"]
    for name in ["Low", "Mid", "High"]:
        r    = metrics.get(f"cth_r_{name}", np.nan)
        rmse = metrics.get(f"cth_rmse_{name}", np.nan)
        bias = metrics.get(f"cth_bias_{name}", np.nan)
        n    = metrics.get(f"n_{name}", 0)
        lines.append(f"  {name:5s}  {r:+.3f}  {rmse:4.0f}m  {bias:+.0f}m  n={n}")
    lines += [
        "",
        f"  ALL   {metrics.get('cth_r_all', 0):+.3f}  "
        f"{metrics.get('cth_rmse_all', 0):4.0f}m  "
        f"{metrics.get('cth_bias_all', 0):+.0f}m",
    ]
    ax.text(0, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=5.5, va="top", fontfamily="monospace")

    fig.suptitle(f"CTH stratified validation — {scene_id}",
                 fontsize=7, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------------

def validate_one(
    ref_clp: np.ndarray,
    ref_cth: np.ndarray,
    agri_lat: np.ndarray,
    agri_lon: np.ndarray,
    agri_vza: Optional[np.ndarray],
    agri_sza: Optional[np.ndarray],
    agri_dt,
    modis_files: list,
    myd03_files: list,
    scene_id: str,
    ref_type: str = "model",
) -> dict:
    """
    Run cloud-mask + stratified-CTH validation for one scene.

    Parameters
    ----------
    agri_vza, agri_sza : 真实 AGRI 卫星/太阳天顶角（来自 GEO 文件）。
        传入真实值可使 apply_quality_filter 的几何门控正常工作；
        若传 None 则退化为全零（允许所有像元通过几何门控，CTH 仍正常过滤）。
    """
    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return {"scene_id": scene_id, "status": "no_modis", "n_granules": 0}

    labels = match_modis_to_agri_grid(
        agri_lat, agri_lon, agri_dt, matched, myd03_files,
        agri_vza=agri_vza, agri_sza=agri_sza,
    )
    if labels is None:
        return {"scene_id": scene_id, "status": "match_failed", "n_granules": len(matched)}

    modis_clp = labels["CLP"].astype(np.float32)
    modis_clp[(modis_clp < 0) | (modis_clp >= 3)] = np.nan
    modis_cmask = to_cloud_mask(modis_clp, "modis")
    modis_cth = labels["CTH"]

    if ref_type == "agri_l2_raw":
        ref_cmask = to_cloud_mask(ref_clp.astype(np.float32), "agri_l2_raw")
    else:
        ref_cmask = to_cloud_mask(ref_clp.astype(np.float32), "model")

    assert ref_clp.shape == modis_clp.shape, (
        f"Shape mismatch: ref={ref_clp.shape} modis={modis_clp.shape}"
    )

    cloud_m = cloud_detection_metrics(ref_cmask, modis_cmask)
    cth_m = cth_stratified_metrics(ref_cth, modis_cth, ref_cmask, modis_cmask)
    metrics = {
        **cloud_m, **cth_m,
        "scene_id": scene_id,
        "status": "ok",
        "n_granules": len(matched),
    }

    # 诊断字段
    dt_arr = labels.get("MATCH_DT_MIN")
    if dt_arr is not None:
        valid_dt = dt_arr[np.isfinite(dt_arr)]
        metrics["modis_dt_median_min"] = float(np.median(valid_dt)) if valid_dt.size else np.nan
        metrics["modis_dt_max_min"] = float(np.max(valid_dt)) if valid_dt.size else np.nan
    metrics["modis_clp_coverage_pct"] = 100.0 * float(np.isfinite(modis_clp).mean())
    metrics["modis_cth_coverage_pct"] = 100.0 * float(np.isfinite(modis_cth).mean())

    layers_str = " ".join(
        f"{n}={metrics.get(f'cth_r_{n}', np.nan):+.2f}" for n in ["Low", "Mid", "High"]
    )
    log.info(
        "  %s  POD=%.3f FAR=%.3f HSS=%.3f  CTH R: %s  "
        "clp_cover=%.1f%%  cth_cover=%.1f%%  dt_med=%.1f min",
        scene_id,
        cloud_m.get("pod", 0), cloud_m.get("far", 0), cloud_m.get("hss", 0),
        layers_str,
        metrics["modis_clp_coverage_pct"],
        metrics["modis_cth_coverage_pct"],
        metrics.get("modis_dt_median_min", float("nan")),
    )

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
    parser = argparse.ArgumentParser(
        description="Cloud-mask + stratified-CTH MODIS validation"
    )
    parser.add_argument("--day", required=True)
    parser.add_argument("--npz_dir", default=None)
    parser.add_argument("--reference", choices=["model", "l2"], default="model")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    day = args.day
    agri_day_dir = cfg.AGRI_ROOT / day
    if not agri_day_dir.is_dir():
        log.error("AGRI L1B dir not found: %s", agri_day_dir)
        sys.exit(1)

    agri_files = sorted([
        f for f in agri_day_dir.glob("*_FDI-_*.HDF")
        if not f.name.endswith(".db")
    ])

    modis_dir  = cfg.MODIS_ROOT / day
    myd03_dir  = cfg.MYD03_ROOT / day
    all_modis  = sorted(
        list(modis_dir.glob("MYD06*.hdf")) + list(modis_dir.glob("MYD06*.HDF"))
    ) if modis_dir.is_dir() else []
    all_myd03  = sorted(
        list(myd03_dir.glob("MYD03*.hdf")) + list(myd03_dir.glob("MYD03*.HDF"))
    ) if myd03_dir.is_dir() else []

    npz_index = {}
    if args.reference == "model" and args.npz_dir:
        for npz_p in Path(args.npz_dir).rglob(f"*{day}*.npz"):
            ts = _extract_timestamp_from_filename(npz_p.name)
            if ts:
                npz_index[ts] = npz_p

    log.info(
        "Day %s: %d AGRI scenes, %d MODIS, %d MYD03, ref=%s",
        day, len(agri_files), len(all_modis), len(all_myd03), args.reference,
    )

    OUTPUT_DIR = OUT_DIR
    all_metrics = []
    for agri_f in agri_files:
        ts = _extract_timestamp_from_filename(agri_f.name)
        if ts is None:
            continue
        scene_id = f"{ts[:8]}_{ts[8:]}"

        # ── 读取 AGRI geo（含真实 VZA/SZA）──────────────────────────────
        agri = read_agri_scene(agri_f)
        if agri is None:
            continue
        agri_dt = parse_agri_datetime(agri_f.name)
        if agri_dt is None:
            continue

        # ── 获取参考数据 ──────────────────────────────────────────────────
        if args.reference == "model":
            if ts not in npz_index:
                continue
            data = np.load(npz_index[ts])
            ref_clp = data["CLP_pred"].astype(np.float32)
            ref_cth = data["CTH_pred"].astype(np.float32)
            data.close()
            ref_type = "model"
        else:
            dummy = Path(
                f"FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_{ts}_x_4000M_V0001.HDF"
            )
            clp_nc = _find_matching_l2_file(dummy, "CLP")
            cth_nc = _find_matching_l2_file(dummy, "CTH")
            if clp_nc is None or cth_nc is None:
                continue
            ref_clp_raw = read_l2_clp_raw(clp_nc)
            ref_cth = read_l2_cth(cth_nc)
            if ref_clp_raw is None or ref_cth is None:
                continue
            ref_clp = ref_clp_raw
            ref_type = "agri_l2_raw"

        try:
            m = validate_one(
                ref_clp, ref_cth,
                agri["lat"], agri["lon"],
                agri["VZA"], agri["SZA"],   # ← 传入真实角度（修复 Bug 2）
                agri_dt,
                all_modis, all_myd03,
                scene_id, ref_type,
            )
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", scene_id, exc)

    if all_metrics and args.summary:
        import pandas as pd
        rows = [
            {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
            for m in all_metrics
        ]
        df = pd.DataFrame(rows)
        tag = "model" if args.reference == "model" else "l2"
        df.to_csv(OUTPUT_DIR / f"cloud_validation_{tag}_{day}.csv", index=False)

        ok = [m for m in all_metrics if m.get("status") == "ok"]
        if ok:
            pod_vals = [m["pod"] for m in ok if m.get("pod", 0) > 0]
            far_vals = [m["far"] for m in ok]
            hss_vals = [m["hss"] for m in ok]
            log.info(
                "=== Cloud detection: %s vs MODIS  %s (n=%d) ===",
                tag, day, len(ok),
            )
            log.info("POD:   %.3f ± %.3f", np.mean(pod_vals), np.std(pod_vals))
            log.info("FAR:   %.3f ± %.3f", np.mean(far_vals), np.std(far_vals))
            log.info("HSS:   %.3f ± %.3f", np.mean(hss_vals), np.std(hss_vals))
            log.info(
                "MODIS CLP coverage: %.1f%% ± %.1f%%",
                np.mean([m["modis_clp_coverage_pct"] for m in ok]),
                np.std([m["modis_clp_coverage_pct"] for m in ok]),
            )
            log.info(
                "MODIS CTH coverage: %.1f%% ± %.1f%%",
                np.mean([m["modis_cth_coverage_pct"] for m in ok]),
                np.std([m["modis_cth_coverage_pct"] for m in ok]),
            )
            for layer in ["Low", "Mid", "High"]:
                r_vals = [
                    m[f"cth_r_{layer}"] for m in ok
                    if m.get(f"n_{layer}", 0) > 10
                    and np.isfinite(m.get(f"cth_r_{layer}", np.nan))
                ]
                rmse_vals = [
                    m[f"cth_rmse_{layer}"] for m in ok
                    if m.get(f"n_{layer}", 0) > 10
                    and np.isfinite(m.get(f"cth_rmse_{layer}", np.nan))
                ]
                if r_vals:
                    log.info(
                        "CTH %s: R=%+.3f±%.3f  RMSE=%.0f±%.0f m  (n_scenes=%d)",
                        layer,
                        np.mean(r_vals), np.std(r_vals),
                        np.mean(rmse_vals), np.std(rmse_vals),
                        len(r_vals),
                    )


if __name__ == "__main__":
    main()
