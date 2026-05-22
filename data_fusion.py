"""
data_fusion.py
==============
GPM + AGRI 数据配对流水线顶层调度器。

架构
----
  data_fusion.py     <- 本文件：调度、多进程
  fusion_core.py     <- 纯数值工具（AGRI→GPM 空间匹配）
  fusion_io.py       <- 文件读写（AGRI FDI/GEO / GPM HDF5 / HDF5 输出）
  fusion_config.py   <- 质量控制阈值（可被环境变量覆盖）

时间匹配策略
-----------
对于每个 GPM 文件：
  - 收集当天所有 AGRI 景
  - 寻找时间差 ≤ TIME_MAX_MIN 的最近 AGRI
  - argmin(abs(Δt))
  - 若无满足条件则跳过该 GPM 文件

用法
----
  python data_fusion.py --split train --day 20190101
  python data_fusion.py --split train --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import config as cfg
import fusion_config as fc
from fusion_core import compute_tight_disk_mask, extract_tiles_with_gpm
from fusion_io import (
    find_day_folders,
    parse_agri_datetime,
    parse_gpm_datetime,
    read_agri_scene,
    read_gpm_file,
    write_gpm_fused_tiles,
)

log = logging.getLogger(__name__)

# compat alias for main.py
_find_day_folders = find_day_folders


# ---------------------------------------------------------------------------
# 时间匹配
# ---------------------------------------------------------------------------

def _find_closest_agri(
    gpm_dt: datetime,
    agri_files: List[Path],
    max_dt_min: float,
) -> Optional[Tuple[Path, datetime, float]]:
    """
    返回时间差最小的 AGRI 文件，若最小时间差超过 max_dt_min 则返回 None。

    Returns
    -------
    (agri_file, agri_dt, abs_dt_min) or None
    """
    best = None
    best_dt = float("inf")
    for af in agri_files:
        adt = parse_agri_datetime(af.name)
        if adt is None:
            continue
        dt = abs((adt - gpm_dt).total_seconds()) / 60.0
        if dt < best_dt:
            best_dt = dt
            best = (af, adt)
    if best is None or best_dt > max_dt_min:
        return None
    return (best[0], best[1], best_dt)


# ---------------------------------------------------------------------------
# 单场景配对（子进程入口）
# ---------------------------------------------------------------------------

def _pair_one_scene(
    agri_file: str,
    gpm_file: str,
    out_path: str,
    mode: str,
) -> Tuple[bool, str, str]:
    """
    子进程任务：将单个 AGRI 景匹配到单个 GPM 文件，提取 tile。
    返回 (ok, out_path, msg)。
    """
    try:
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

        agri_path = Path(agri_file)
        gpm_path = Path(gpm_file)
        out = Path(out_path)

        # ── 读取 AGRI ──
        agri = read_agri_scene(agri_path)
        if agri is None:
            return False, out_path, "read_agri_scene returned None"

        # ── 读取 GPM ──
        gpm = read_gpm_file(gpm_path)
        if gpm is None:
            return False, out_path, "read_gpm_file returned None"

        # ── 计算时间差 ──
        agri_dt = parse_agri_datetime(agri_path.name)
        gpm_dt = parse_gpm_datetime(gpm_path.name)
        if agri_dt is None or gpm_dt is None:
            return False, out_path, "Cannot parse datetime"

        dt_min = abs((agri_dt - gpm_dt).total_seconds()) / 60.0

        # ── GPM 完整覆盖检查：区域内 NaN 占比 ──
        max_nan_frac = float(getattr(fc, "GPM_COVERAGE_MAX_NAN_FRAC", 0.05))
        lat_min = float(getattr(fc, "REGION_LAT_MIN", -90))
        lat_max = float(getattr(fc, "REGION_LAT_MAX", 90))
        lon_min = float(getattr(fc, "REGION_LON_MIN", -180))
        lon_max = float(getattr(fc, "REGION_LON_MAX", 180))
        if lat_min > -89 or lat_max < 89 or lon_min > -179 or lon_max < 179:
            lat_idx = np.searchsorted(gpm["lat"], [lat_min, lat_max])
            lon_idx = np.searchsorted(gpm["lon"], [lon_min, lon_max])
            lat_slice = slice(max(0, lat_idx[0]), min(len(gpm["lat"]), lat_idx[1]))
            lon_slice = slice(max(0, lon_idx[0]), min(len(gpm["lon"]), lon_idx[1]))
            region_precip = gpm["precip"][lat_slice, lon_slice]
            nan_frac = np.isnan(region_precip).mean()
            if nan_frac > max_nan_frac:
                return False, out_path, f"GPM incomplete coverage: nan_frac={nan_frac:.3f} > {max_nan_frac}"

        # ── 收紧 AGRI 圆盘边界 ──
        margin = float(getattr(fc, "AGRI_DISK_MARGIN_DEG", 5.0))
        if margin > 0:
            sub_lon = float(getattr(fc, "AGRI_SUB_LON", 104.7))
            tight_mask = compute_tight_disk_mask(agri["lat"], agri["lon"], margin, sub_lon=sub_lon)
            agri["lat"] = np.where(tight_mask, agri["lat"], np.nan)
            agri["lon"] = np.where(tight_mask, agri["lon"], np.nan)
            agri["VZA"] = np.where(tight_mask, agri["VZA"], np.nan)
            agri["SZA"] = np.where(tight_mask, agri["SZA"], np.nan)
            bt = agri["BT"]
            mask_3d = np.broadcast_to(tight_mask[..., np.newaxis], bt.shape)
            agri["BT"] = np.where(mask_3d, bt, np.nan)

        # ── 提取 tile ──
        samples = extract_tiles_with_gpm(
            agri=agri,
            gpm_precip=gpm["precip"],
            gpm_lat=gpm["lat"],
            gpm_lon=gpm["lon"],
            tile_size=cfg.TILE_SIZE,
            stride=cfg.TILE_STRIDE[0],
            region_lat_min=float(getattr(fc, "REGION_LAT_MIN", -10)),
            region_lat_max=float(getattr(fc, "REGION_LAT_MAX", 20)),
            region_lon_min=float(getattr(fc, "REGION_LON_MIN", 100)),
            region_lon_max=float(getattr(fc, "REGION_LON_MAX", 130)),
        )

        if not samples:
            return False, out_path, f"No valid GPM tiles (dt={dt_min:.1f}min)"

        # ── 每景随机上限：超过则随机子采样 ──
        max_s = int(getattr(fc, "MAX_SAMPLES_PER_SCENE", 0))
        if max_s > 0 and len(samples) > max_s:
            rng = np.random.RandomState(hash(gpm_file) % (2**31 - 1))
            indices = rng.choice(len(samples), size=max_s, replace=False)
            samples = [samples[i] for i in sorted(indices)]

        # ── 写出 ──
        n = write_gpm_fused_tiles(out, samples, agri_dt, gpm_dt, mode)
        return True, out_path, f"OK tiles={n} dt={dt_min:.1f}min"

    except Exception:
        return False, out_path, f"Exception:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------

def _collect_agri_files(agri_day_dir: Path) -> List[Path]:
    """收集某天所有 AGRI FDI 文件。"""
    return sorted([
        p for p in list(agri_day_dir.glob("*.HDF")) + list(agri_day_dir.glob("*.hdf"))
        if "_FDI-_" in p.name and not p.name.endswith(".db")
    ])


def _collect_gpm_files(gpm_day_dir: Path) -> List[Path]:
    """收集某天所有 GPM IMERG HDF5 文件。"""
    return sorted([
        p for p in list(gpm_day_dir.glob("*.HDF5")) + list(gpm_day_dir.glob("*.hdf5"))
    ])


def fuse_day(
    agri_day_dir: Path,
    out_dir: Path,
    mode: str = "train",
    overwrite: bool = False,
    n_workers: int = 1,
    max_dt_min: float = None,
) -> int:
    """
    单日 GPM+AGRI 配对调度。

    Parameters
    ----------
    agri_day_dir : AGRI 日文件夹
    out_dir : 输出目录
    mode : "train" / "val" / "test"
    overwrite : 是否覆盖已有输出
    n_workers : 并行进程数
    max_dt_min : 最大时间差 (默认从 config 读取)
    """
    if max_dt_min is None:
        max_dt_min = float(getattr(fc, "TIME_MAX_MIN", 15.0))

    date_str = agri_day_dir.name
    gpm_day_dir = cfg.GPM_ROOT / date_str

    agri_files = _collect_agri_files(agri_day_dir)
    if not agri_files:
        log.info("Day %s: no AGRI files", date_str)
        return 0

    if not gpm_day_dir.is_dir():
        log.info("Day %s: no GPM directory %s", date_str, gpm_day_dir)
        return 0

    gpm_files = _collect_gpm_files(gpm_day_dir)
    if not gpm_files:
        log.info("Day %s: no GPM files", date_str)
        return 0

    # ── 预先解析所有 AGRI 时间（在主进程完成） ──
    agri_times = []
    for af in agri_files:
        adt = parse_agri_datetime(af.name)
        if adt is not None:
            agri_times.append((af, adt))

    log.info("Day %s | AGRI=%d GPM=%d | workers=%d dt_max=%.0fmin",
             date_str, len(agri_times), len(gpm_files), n_workers, max_dt_min)

    # ── 构建任务列表 ──
    tasks = []
    skipped_no_match = 0
    for gf in gpm_files:
        gdt = parse_gpm_datetime(gf.name)
        if gdt is None:
            continue

        match = _find_closest_agri(gdt, [t[0] for t in agri_times], max_dt_min)
        if match is None:
            skipped_no_match += 1
            continue

        af, adt, dt_min = match
        out_name = f"GPM_AGRI_{date_str}_{adt:%H%M%S}_{gdt:%H%M%S}.h5"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            continue
        tasks.append((str(af), str(gf), str(out_path), mode))

    log.info("Day %s | tasks=%d skipped(no_agri_match)=%d",
             date_str, len(tasks), skipped_no_match)

    if not tasks:
        return 0

    # ── 执行 ──
    success = 0
    if n_workers <= 1:
        for args in tasks:
            ok, op, msg = _pair_one_scene(*args)
            if ok:
                success += 1
                log.debug("OK %s: %s", Path(args[2]).name, msg)
            else:
                log.debug("Skip %s: %s", Path(args[2]).name, msg[:200])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_pair_one_scene, *t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    ok, op, msg = fut.result()
                except Exception as exc:
                    ok, op, msg = False, task[2], str(exc)
                if ok:
                    success += 1
                else:
                    log.debug("Skip %s: %s", Path(task[2]).name, msg[:200])

    log.info("Day %s - %d/%d ok", date_str, success, len(tasks))
    return success


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="GPM+AGRI precipitation classification data fusion")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--day", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=fc.N_FUSION_WORKERS)
    parser.add_argument("--max-dt-min", type=float, default=None)
    args = parser.parse_args()

    split_out = {
        "train": cfg.PAIRED_TRAIN_DIR,
        "val": cfg.PAIRED_VAL_DIR,
        "test": cfg.PAIRED_TEST_DIR,
    }[args.split]

    dates = {
        "train": cfg.TRAIN_DATES,
        "val": cfg.VAL_DATES,
        "test": cfg.TEST_DATES,
    }[args.split]

    if args.day:
        dates = [args.day]

    agri_days = find_day_folders(cfg.AGRI_ROOT, dates)

    total = 0
    for agri_day in agri_days:
        out_sub = split_out / agri_day.name
        total += fuse_day(
            agri_day, out_sub,
            mode=args.split,
            overwrite=args.overwrite,
            n_workers=args.workers,
            max_dt_min=args.max_dt_min,
        )

    log.info("Fusion done - %d files total", total)


if __name__ == "__main__":
    main()
