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
import argparse, logging, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np

import config as cfg
import fusion_config as fc
from fusion_core import aggregate_modis_to_agri
from fusion_io import (
    apply_quality_filter, find_day_folders, find_matching_modis,
    parse_agri_datetime, parse_modis_datetime,
    read_agri_scene, read_myd06,
    write_fused_samples, write_full_disk_hdf5,
)
from sample_filters import get_patch_supervision_thresholds

log = logging.getLogger(__name__)

# compat alias used by main.py
_find_day_folders = find_day_folders


def _fuse_one_scene(agri_file, modis_files, out_path, mode):
    """子进程任务：融合单个 AGRI 场景，返回 (ok, out_path, msg)。"""
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    agri_path = Path(agri_file)
    out = Path(out_path)
    try:
        agri_dt = parse_agri_datetime(agri_path.name)
        if agri_dt is None:
            return False, out_path, "Cannot parse AGRI datetime"

        agri = read_agri_scene(agri_path)
        if agri is None:
            return False, out_path, "read_agri_scene None"

        modis_list = []
        for mf in [Path(p) for p in modis_files]:
            m = read_myd06(mf, agri_dt=agri_dt)
            if m is None:
                continue
            mdt = parse_modis_datetime(mf.name)
            if mdt is None:
                continue
            m["_dt_min"] = abs((mdt - agri_dt).total_seconds()) / 60.0
            m["_file"] = mf.name
            modis_list.append(m)

        if not modis_list:
            return False, out_path, "No MYD06 after reading"
        labels = aggregate_modis_to_agri(agri["lat"], agri["lon"], modis_list)
        if labels is None:
            return False, out_path, "aggregate returned None"

        labels = apply_quality_filter(agri, labels)

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
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        if cfg.FUSION_OUTPUT_MODE == "samples_only":
            n_s = write_fused_samples(out, agri, labels, agri_dt, mode)
            return True, out_path, f"OK samples={n_s}"
        else:
            write_full_disk_hdf5(out, agri, labels, agri_dt)
            return True, out_path, "OK full_disk"

    except Exception:
        return False, out_path, f"Exception:\n{traceback.format_exc()}"


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

                clp_cmap = ListedColormap(["white","deepskyblue","cyan","orange","red"])
                clp_norm = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], clp_cmap.N)
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
                cb = plt.colorbar(im3, ax=axes[1,0], ticks=[0,1,2,3,4])
                cb.ax.set_yticklabels(["Clear","Water","Supercool","Mixed","Ice"], fontsize=8)
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
    mode: str = "train",
    overwrite: bool = False,
    max_qc: int = 3,
    n_workers: int = fc.N_FUSION_WORKERS,
) -> int:
    agri_files = sorted([
        p for p in list(agri_day_dir.glob("*.HDF")) + list(agri_day_dir.glob("*.hdf"))
        if "_FDI-_" in p.name
    ])
    modis_files = sorted(
        list(modis_day_dir.glob("MYD06*.hdf")) +
        list(modis_day_dir.glob("MYD06*.HDF"))
    )
    log.info("Day %s | AGRI=%d MYD06=%d | workers=%d",
             agri_day_dir.name, len(agri_files), len(modis_files), n_workers)

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
        out_name = f"AGRI_MYD06_{agri_dt:%Y%m%d_%H%M%S}.h5"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            continue
        tasks.append((str(agri_file), [str(f) for f in matched], str(out_path), mode))

    if not tasks:
        log.info("Day %s - no tasks", agri_day_dir.name)
        return 0

    log.info("Day %s - submitting %d tasks", agri_day_dir.name, len(tasks))
    success, qc_count = 0, 0

    if n_workers <= 1:
        for args in tasks:
            ok, op, msg = _fuse_one_scene(*args)
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
                    ok, op, msg = fut.result()
                except Exception as exc:
                    ok, op, msg = False, task[2], str(exc)
                if ok:
                    success += 1
                    if qc_count < max_qc:
                        _make_qc_figure(Path(op), Path(op).with_name(Path(op).stem + "_qc.png"))
                        qc_count += 1
                else:
                    log.debug("Skip %s: %s", Path(task[2]).name, msg[:200])

    log.info("Day %s - %d/%d ok | %d QC figs", agri_day_dir.name, success, len(tasks), qc_count)
    return success


# compat wrapper called by main.py's stage_fuse
def fuse_day_compat(agri_day, modis_day, out_sub, overwrite=False, max_qc=3):
    parts = {p.lower() for p in out_sub.parts}
    mode = "val" if ("val" in parts or "valid" in parts) else ("test" if "test" in parts else "train")
    return fuse_day(agri_day, modis_day, out_sub, mode=mode,
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
    args = parser.parse_args()

    split_out  = {"train":cfg.PAIRED_TRAIN_DIR,"val":cfg.PAIRED_VAL_DIR,"test":cfg.PAIRED_TEST_DIR}[args.split]
    dates      = {"train":cfg.TRAIN_DATES,"val":cfg.VAL_DATES,"test":cfg.TEST_DATES}[args.split]
    if args.day:
        dates = [args.day]

    agri_days  = find_day_folders(cfg.AGRI_ROOT, dates)
    modis_days = {d.name: d for d in find_day_folders(cfg.MODIS_ROOT, dates)}

    total = 0
    for agri_day in agri_days:
        modis_day = modis_days.get(agri_day.name)
        if modis_day is None:
            log.warning("No MODIS for %s", agri_day.name)
            continue
        total += fuse_day(agri_day, modis_day, split_out / agri_day.name,
                          mode=args.split, overwrite=args.overwrite,
                          max_qc=args.max_qc, n_workers=args.workers)

    log.info("Fusion done - %d files total", total)


if __name__ == "__main__":
    main()
