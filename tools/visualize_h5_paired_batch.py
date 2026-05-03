#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_h5_modis_region.py

极简版：每个融合 H5 文件只输出一张图。

功能：
  1) 读取 samples_v2 风格 HDF5/H5 文件
  2) 使用 /Samples/row, /Samples/col 把每个 patch 的 labels 贴回 AGRI 整景位置
  3) 每个 H5 输出一张 2x2 PNG：
       - 云相态 CLP: labels[:, 0]
       - 云光学厚度 COT: labels[:, 1]
       - 云粒子有效半径 CER: labels[:, 2]
       - 云顶高度 CTH: labels[:, 3]

适配目录：
  paired/
    train/YYYYMMDD/*.h5
    val/YYYYMMDD/*.h5
    test/YYYYMMDD/*.h5

安装依赖：
  pip install h5py numpy matplotlib

常用命令：
  # 跑 paired 下全部 H5：每个 H5 只出一张图
  python visualize_h5_modis_region.py /data/Data_yuq/unet_workdir/paired \
      --out /home/yuq/cloudmask/unet/tools/h5_modis_region

  # 只跑 val
  python visualize_h5_modis_region.py /data/Data_yuq/unet_workdir/paired \
      --out /home/yuq/cloudmask/unet/tools/h5_modis_region \
      --splits val

  # 只跑某几个日期
  python visualize_h5_modis_region.py /data/Data_yuq/unet_workdir/paired \
      --out /home/yuq/cloudmask/unet/tools/h5_modis_region \
      --dates 20190115,20190315

  # 只试跑前 3 个文件
  python visualize_h5_modis_region.py /data/Data_yuq/unet_workdir/paired \
      --out /home/yuq/cloudmask/unet/tools/h5_modis_region \
      --max-files 3

  # 如果觉得整景太大、patch 区域看不清，可裁剪到 MODIS 落入区域附近
  python visualize_h5_modis_region.py /data/Data_yuq/unet_workdir/paired \
      --out /home/yuq/cloudmask/unet/tools/h5_modis_region_crop \
      --crop-to-data --margin 64
"""

from __future__ import print_function

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


H5_SUFFIXES = {".h5", ".hdf5", ".hdf"}
DEFAULT_SPLITS = {"train", "val", "test"}
LABEL_NAMES = ["CLP", "COT", "CER", "CTH"]


# =============================================================================
# 基础工具
# =============================================================================


def decode_attr(value: Any) -> Any:
    """把 HDF5 attr 中的 bytes / numpy 类型转成普通 Python 对象。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O"}:
            return [decode_attr(x) for x in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_name(text: str) -> str:
    """生成适合文件名/目录名的字符串。"""
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", text)
    return text.strip("_") or "unnamed"


def is_h5_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in H5_SUFFIXES


def get_samples_group(h5: h5py.File, group: str = "Samples") -> h5py.Group:
    """优先返回 /Samples；如果数据直接在根目录，则返回根目录。"""
    if group in h5 and isinstance(h5[group], h5py.Group):
        return h5[group]
    if "labels" in h5 and "row" in h5 and "col" in h5:
        return h5  # type: ignore[return-value]
    raise KeyError("找不到 /{} 组，也没有在根目录发现 labels/row/col。".format(group))


def get_dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    obj = group.get(key)
    if not isinstance(obj, h5py.Dataset):
        raise KeyError("缺少数据集: {}".format(key))
    return obj


def read_1d(group: h5py.Group, key: str) -> np.ndarray:
    ds = get_dataset(group, key)
    return np.asarray(ds[...]).reshape(-1)


def robust_vmin_vmax(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> Tuple[float, float]:
    """用分位数设置连续变量显示范围，避免极端值把图拉白/拉黑。"""
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [p_low, p_high])
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or float(vmin) == float(vmax):
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def infer_scene_shape(
    h5: h5py.File,
    samples: h5py.Group,
    rows: np.ndarray,
    cols: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> Tuple[int, int]:
    """优先从 attr scene_shape 读取整景大小，否则根据 row/col + patch_size 推断。"""
    for holder in (h5.attrs, samples.attrs):
        if "scene_shape" in holder:
            raw = decode_attr(holder["scene_shape"])
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 2:
                return int(raw[0]), int(raw[1])

    valid = np.isfinite(rows) & np.isfinite(cols)
    if not np.any(valid):
        raise ValueError("row/col 全部无效，无法推断 scene_shape。")
    scene_h = int(np.nanmax(rows[valid])) + patch_h
    scene_w = int(np.nanmax(cols[valid])) + patch_w
    return scene_h, scene_w


# =============================================================================
# 核心：把 patch labels 贴回整景
# =============================================================================


def reconstruct_label_scenes(
    h5_path: Union[str, Path],
    group: str = "Samples",
    max_patches: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    从单个 H5 中重建 4 个整景标签图。

    Returns
    -------
    scenes : dict
        keys = CLP/COT/CER/CTH, values = 2D array，shape 为 scene_shape。
    meta : dict
        文件名、样本数、scene_shape、patch_size 等信息。

    处理重叠 patch 的规则：
      - CLP：对 0/1/2 做多数投票；平票时取较小类别。
      - COT/CER/CTH：对重叠像元求平均。
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        samples = get_samples_group(h5, group)
        labels_ds = get_dataset(samples, "labels")
        rows = read_1d(samples, "row").astype(np.float64)
        cols = read_1d(samples, "col").astype(np.float64)

        if labels_ds.ndim != 4:
            raise ValueError("labels 应为 (N, 4, H, W)，当前 shape={}".format(labels_ds.shape))
        if labels_ds.shape[1] < 4:
            raise ValueError("labels 第 2 维通道数应至少为 4：CLP/COT/CER/CTH，当前 shape={}".format(labels_ds.shape))

        n_total = int(labels_ds.shape[0])
        patch_h = int(labels_ds.shape[2])
        patch_w = int(labels_ds.shape[3])
        scene_h, scene_w = infer_scene_shape(h5, samples, rows, cols, patch_h, patch_w)

        if max_patches is None or max_patches <= 0 or max_patches > n_total:
            n_use = n_total
        else:
            n_use = int(max_patches)

        # CLP 多数投票计数：0 Clear, 1 Water, 2 Ice
        clp_counts = np.zeros((3, scene_h, scene_w), dtype=np.uint16)

        # 连续量重叠区域求平均
        sums = {
            "COT": np.zeros((scene_h, scene_w), dtype=np.float64),
            "CER": np.zeros((scene_h, scene_w), dtype=np.float64),
            "CTH": np.zeros((scene_h, scene_w), dtype=np.float64),
        }
        counts = {
            "COT": np.zeros((scene_h, scene_w), dtype=np.uint16),
            "CER": np.zeros((scene_h, scene_w), dtype=np.uint16),
            "CTH": np.zeros((scene_h, scene_w), dtype=np.uint16),
        }

        used_patch_count = 0
        valid_pixel_count = 0

        for i in range(n_use):
            if not (np.isfinite(rows[i]) and np.isfinite(cols[i])):
                continue

            rr0_raw = int(round(rows[i]))
            cc0_raw = int(round(cols[i]))
            rr1_raw = rr0_raw + patch_h
            cc1_raw = cc0_raw + patch_w

            rr0 = max(0, rr0_raw)
            cc0 = max(0, cc0_raw)
            rr1 = min(scene_h, rr1_raw)
            cc1 = min(scene_w, cc1_raw)
            if rr1 <= rr0 or cc1 <= cc0:
                continue

            pr0 = rr0 - rr0_raw
            pc0 = cc0 - cc0_raw
            pr1 = pr0 + (rr1 - rr0)
            pc1 = pc0 + (cc1 - cc0)

            patch = np.asarray(labels_ds[i, :4, pr0:pr1, pc0:pc1], dtype=np.float32)
            used_patch_count += 1

            # 1) CLP: 0 Clear / 1 Water / 2 Ice
            clp = patch[0]
            clp_finite = np.isfinite(clp)
            if np.any(clp_finite):
                clp_int = np.rint(clp).astype(np.int16)
                for cls in (0, 1, 2):
                    m = clp_finite & (clp_int == cls)
                    if np.any(m):
                        view = clp_counts[cls, rr0:rr1, cc0:cc1]
                        view[m] = view[m] + 1

            # 2) COT/CER/CTH: overlapping pixels -> mean
            for channel_idx, name in ((1, "COT"), (2, "CER"), (3, "CTH")):
                arr = patch[channel_idx]
                m = np.isfinite(arr)
                if np.any(m):
                    s_view = sums[name][rr0:rr1, cc0:cc1]
                    c_view = counts[name][rr0:rr1, cc0:cc1]
                    s_view[m] = s_view[m] + arr[m]
                    c_view[m] = c_view[m] + 1
                    valid_pixel_count += int(np.count_nonzero(m))

        scenes: Dict[str, np.ndarray] = {}

        clp_total = clp_counts.sum(axis=0)
        clp_scene = np.full((scene_h, scene_w), np.nan, dtype=np.float32)
        m_clp = clp_total > 0
        if np.any(m_clp):
            clp_scene[m_clp] = np.argmax(clp_counts[:, m_clp], axis=0).astype(np.float32)
        scenes["CLP"] = clp_scene

        for name in ("COT", "CER", "CTH"):
            out = np.full((scene_h, scene_w), np.nan, dtype=np.float32)
            m = counts[name] > 0
            if np.any(m):
                out[m] = (sums[name][m] / counts[name][m]).astype(np.float32)
            scenes[name] = out

        meta = {
            "h5_path": str(h5_path),
            "file_name": h5_path.name,
            "n_total": n_total,
            "n_used": used_patch_count,
            "scene_shape": (scene_h, scene_w),
            "patch_size": (patch_h, patch_w),
            "valid_label_pixel_updates": valid_pixel_count,
        }
        return scenes, meta


# =============================================================================
# 绘图：每个 H5 一张图
# =============================================================================


def crop_scenes_to_data(
    scenes: Dict[str, np.ndarray],
    margin: int = 32,
) -> Tuple[Dict[str, np.ndarray], Tuple[int, int, int, int]]:
    """裁剪到有 MODIS 标签落入的区域附近，返回裁剪后的图和 bbox=(r0,r1,c0,c1)。"""
    mask = np.zeros_like(next(iter(scenes.values())), dtype=bool)
    for arr in scenes.values():
        mask |= np.isfinite(arr)

    if not np.any(mask):
        h, w = mask.shape
        return scenes, (0, h, 0, w)

    rr, cc = np.where(mask)
    h, w = mask.shape
    r0 = max(0, int(rr.min()) - int(margin))
    r1 = min(h, int(rr.max()) + int(margin) + 1)
    c0 = max(0, int(cc.min()) - int(margin))
    c1 = min(w, int(cc.max()) + int(margin) + 1)

    cropped = {name: arr[r0:r1, c0:c1] for name, arr in scenes.items()}
    return cropped, (r0, r1, c0, c1)


def plot_modis_region_figure(
    scenes: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    out_png: Union[str, Path],
    crop_to_data: bool = False,
    margin: int = 32,
    dpi: int = 180,
) -> Path:
    """输出单张 2x2 图：CLP/COT/CER/CTH。"""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_scenes = scenes
    crop_bbox = None
    if crop_to_data:
        plot_scenes, crop_bbox = crop_scenes_to_data(scenes, margin=margin)

    # extent 让坐标轴显示原始 scene row/col，而不是裁剪后坐标
    if crop_bbox is None:
        full_h, full_w = scenes["CLP"].shape
        extent = [0, full_w, full_h, 0]
        coord_note = "full scene"
    else:
        r0, r1, c0, c1 = crop_bbox
        extent = [c0, c1, r1, r0]
        coord_note = "crop row {}:{} col {}:{}".format(r0, r1, c0, c1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.reshape(-1)

    # CLP 离散色图
    clp_cmap = ListedColormap(["#eeeeee", "#4c78a8", "#f58518"])
    clp_cmap.set_bad("white")
    clp_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], clp_cmap.N)
    im0 = axes[0].imshow(plot_scenes["CLP"], cmap=clp_cmap, norm=clp_norm, extent=extent, interpolation="nearest")
    axes[0].set_title("Cloud Phase / CLP\n0=Clear, 1=Water, 2=Ice")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar0.ax.set_yticklabels(["Clear", "Water", "Ice"])

    # 连续变量
    for ax, name, title in zip(
        axes[1:],
        ["COT", "CER", "CTH"],
        ["Cloud Optical Thickness / COT", "Cloud Effective Radius / CER", "Cloud Top Height / CTH"],
    ):
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        vmin, vmax = robust_vmin_vmax(plot_scenes[name])
        im = ax.imshow(plot_scenes[name], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("AGRI col")
        ax.set_ylabel("AGRI row")
        ax.grid(False)

    valid_area = int(np.count_nonzero(np.isfinite(scenes["CLP"])))
    scene_shape = meta.get("scene_shape", None)
    patch_size = meta.get("patch_size", None)
    fig.suptitle(
        "{} | patches used {}/{} | scene={} | patch={} | valid CLP px={} | {}".format(
            meta.get("file_name", "unknown"),
            meta.get("n_used", "?"),
            meta.get("n_total", "?"),
            scene_shape,
            patch_size,
            valid_area,
            coord_note,
        ),
        fontsize=12,
    )

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def visualize_one_h5_modis_region(
    h5_path: Union[str, Path],
    out_png: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    group: str = "Samples",
    crop_to_data: bool = False,
    margin: int = 32,
    dpi: int = 180,
    overwrite: bool = False,
    max_patches: Optional[int] = None,
) -> Path:
    """
    单文件入口：每个 H5 只输出一张图。

    Parameters
    ----------
    h5_path : str | Path
        输入 H5 文件。
    out_png : str | Path | None
        指定输出 PNG。若为 None，则根据 out_dir 自动生成。
    out_dir : str | Path | None
        输出目录。out_png 为 None 时使用。
    crop_to_data : bool
        False：显示完整 AGRI scene；True：裁剪到 MODIS 落入区域附近。
    max_patches : int | None
        只用前 N 个 patch，主要用于快速调试。默认 None 表示使用全部 patch。
    """
    h5_path = Path(h5_path)
    if out_png is None:
        if out_dir is None:
            out_dir = h5_path.parent
        out_png = Path(out_dir) / (h5_path.stem + "_modis_region.png")
    out_png = Path(out_png)

    if out_png.exists() and not overwrite:
        return out_png

    scenes, meta = reconstruct_label_scenes(h5_path, group=group, max_patches=max_patches)
    return plot_modis_region_figure(
        scenes=scenes,
        meta=meta,
        out_png=out_png,
        crop_to_data=crop_to_data,
        margin=margin,
        dpi=dpi,
    )


# =============================================================================
# paired 目录批处理
# =============================================================================


def parse_csv_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    text = text.strip()
    if not text or text.lower() == "all":
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def infer_split_date(path: Path, root: Path) -> Tuple[str, str]:
    """从路径中推断 split/date。"""
    split = "unknown_split"
    date = "unknown_date"

    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except Exception:
        parts = path.parts

    if len(parts) >= 3 and parts[0] in DEFAULT_SPLITS and re.fullmatch(r"\d{8}", parts[1]):
        return parts[0], parts[1]

    all_parts = path.resolve().parts
    for i, p in enumerate(all_parts):
        if p in DEFAULT_SPLITS:
            split = p
            if i + 1 < len(all_parts) and re.fullmatch(r"\d{8}", all_parts[i + 1]):
                date = all_parts[i + 1]
            break

    if date == "unknown_date":
        for p in all_parts:
            if re.fullmatch(r"\d{8}", p):
                date = p
                break

    return split, date


def discover_h5_files(
    input_path: Union[str, Path],
    splits: Optional[Sequence[str]] = None,
    dates: Optional[Sequence[str]] = None,
    recursive: bool = True,
    max_files: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """扫描单个 H5 或目录，返回文件清单。"""
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("输入路径不存在: {}".format(root))

    split_filter = set(splits or [])
    date_filter = set(dates or [])

    if is_h5_file(root):
        files = [root]
        scan_root = root.parent
    elif root.is_dir():
        iterator = root.rglob("*") if recursive else root.glob("*")
        files = sorted([p for p in iterator if is_h5_file(p)])
        scan_root = root
    else:
        raise ValueError("输入路径既不是 H5 文件也不是目录: {}".format(root))

    records: List[Dict[str, Any]] = []
    for p in files:
        split, date = infer_split_date(p, scan_root)
        if split_filter and split not in split_filter:
            continue
        if date_filter and date not in date_filter:
            continue
        records.append({"path": str(p), "split": split, "date": date, "stem": p.stem})

    if max_files is not None and max_files > 0:
        records = records[: int(max_files)]
    return records


def default_out_png(base_out_dir: Union[str, Path], record: Dict[str, Any]) -> Path:
    """输出路径：out/split/date/file_stem_modis_region.png。"""
    base = Path(base_out_dir).expanduser().resolve()
    return (
        base
        / safe_name(record.get("split", "unknown_split"))
        / safe_name(record.get("date", "unknown_date"))
        / (safe_name(record.get("stem", "file")) + "_modis_region.png")
    )


def write_run_summary(summary_csv: Path, rows: List[Dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_no", "status", "split", "date", "h5_path", "out_png", "error"]
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def batch_visualize_paired_modis_region(
    input_path: Union[str, Path],
    out_dir: Union[str, Path],
    splits: Optional[Sequence[str]] = None,
    dates: Optional[Sequence[str]] = None,
    recursive: bool = True,
    max_files: Optional[int] = None,
    group: str = "Samples",
    crop_to_data: bool = False,
    margin: int = 32,
    dpi: int = 180,
    overwrite: bool = False,
    stop_on_error: bool = False,
    max_patches: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """目录批处理：paired 下每个 H5 只出一张图。"""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_h5_files(
        input_path=input_path,
        splits=splits,
        dates=dates,
        recursive=recursive,
        max_files=max_files,
    )

    summary_rows: List[Dict[str, Any]] = []
    summary_csv = out_dir / "run_summary.csv"

    if verbose:
        print("[INFO] discovered H5 files = {}".format(len(records)))
        print("[INFO] output dir = {}".format(out_dir))

    for file_no, rec in enumerate(records, start=1):
        h5_path = Path(rec["path"])
        out_png = default_out_png(out_dir, rec)
        row = {
            "file_no": file_no,
            "status": "pending",
            "split": rec.get("split", ""),
            "date": rec.get("date", ""),
            "h5_path": str(h5_path),
            "out_png": str(out_png),
            "error": "",
        }

        if out_png.exists() and not overwrite:
            row["status"] = "skipped_existing"
            summary_rows.append(row)
            write_run_summary(summary_csv, summary_rows)
            if verbose:
                print("[SKIP] {}/{} {}".format(file_no, len(records), out_png))
            continue

        try:
            if verbose:
                print("[RUN] {}/{} {}".format(file_no, len(records), h5_path))
            visualize_one_h5_modis_region(
                h5_path=h5_path,
                out_png=out_png,
                group=group,
                crop_to_data=crop_to_data,
                margin=margin,
                dpi=dpi,
                overwrite=True,
                max_patches=max_patches,
            )
            row["status"] = "ok"
            if verbose:
                print("[OK]  -> {}".format(out_png))
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            if verbose:
                print("[ERROR] {}: {}".format(h5_path, repr(exc)))
            if stop_on_error:
                summary_rows.append(row)
                write_run_summary(summary_csv, summary_rows)
                raise

        summary_rows.append(row)
        write_run_summary(summary_csv, summary_rows)

    if verbose:
        ok = sum(1 for r in summary_rows if r["status"] == "ok")
        skipped = sum(1 for r in summary_rows if r["status"] == "skipped_existing")
        err = sum(1 for r in summary_rows if r["status"] == "error")
        print("[DONE] ok={}, skipped={}, error={}".format(ok, skipped, err))
        print("[INFO] summary = {}".format(summary_csv))

    return summary_rows


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="极简可视化：paired 目录下每个融合 H5 只输出一张 MODIS 落区标签图：CLP/COT/CER/CTH。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", type=str, help="输入 H5 文件或 paired 根目录")
    parser.add_argument("--out", type=str, default="h5_modis_region", help="输出目录")
    parser.add_argument("--group", type=str, default="Samples", help="样本所在 HDF5 group，通常是 Samples")
    parser.add_argument("--splits", type=str, default=None, help="只处理哪些 split，逗号分隔，如 val 或 train,val；默认全部")
    parser.add_argument("--dates", type=str, default=None, help="只处理哪些日期，逗号分隔，如 20190115,20190315；默认全部")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理多少个 H5 文件，用于试跑")
    parser.add_argument("--no-recursive", action="store_true", help="目录模式下不递归查找 H5")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 PNG；默认跳过")
    parser.add_argument("--stop-on-error", action="store_true", help="某个文件出错时立即停止；默认继续")
    parser.add_argument("--crop-to-data", action="store_true", help="裁剪到 MODIS 落入区域附近；默认显示完整 scene 位置")
    parser.add_argument("--margin", type=int, default=32, help="--crop-to-data 时裁剪边界额外留多少像素")
    parser.add_argument("--dpi", type=int, default=180, help="输出 PNG dpi")
    parser.add_argument("--max-patches", type=int, default=None, help="只使用每个 H5 前 N 个 patch，主要用于快速调试；默认使用全部")
    parser.add_argument("--quiet", action="store_true", help="减少终端输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()
    splits = parse_csv_list(args.splits)
    dates = parse_csv_list(args.dates)

    if input_path.is_file():
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / (input_path.stem + "_modis_region.png")
        visualize_one_h5_modis_region(
            h5_path=input_path,
            out_png=out_png,
            group=args.group,
            crop_to_data=args.crop_to_data,
            margin=args.margin,
            dpi=args.dpi,
            overwrite=args.overwrite,
            max_patches=args.max_patches,
        )
        if not args.quiet:
            print("[OK] {}".format(out_png))
    else:
        batch_visualize_paired_modis_region(
            input_path=input_path,
            out_dir=args.out,
            splits=splits,
            dates=dates,
            recursive=not args.no_recursive,
            max_files=args.max_files,
            group=args.group,
            crop_to_data=args.crop_to_data,
            margin=args.margin,
            dpi=args.dpi,
            overwrite=args.overwrite,
            stop_on_error=args.stop_on_error,
            max_patches=args.max_patches,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
