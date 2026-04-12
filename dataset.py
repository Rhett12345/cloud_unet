"""
dataset.py
==========
PyTorch Dataset for the AGRI-only (no GIIRS) cloud-property retrieval task.

Key design decisions
--------------------
- **Lazy loading**: `__init__` only scans files and builds a patch index (list of
  (file_path, i_start, j_start) tuples). Each `__getitem__` call opens the HDF5
  file, reads only the required patch window via HDF5 hyperslab slicing, and
  closes the file immediately. This keeps memory footprint minimal regardless of
  dataset size, and makes dataset initialisation essentially instantaneous.

- `NormStats` is pre-computed once and loaded from disk (see `compute_and_save_stats`).

- `compute_and_save_stats` uses `concurrent.futures.ProcessPoolExecutor` to process
  multiple files in parallel, then reduces partial sums on the main process.

Label channel order
-------------------
  0 : CLP  (float, integer class 0-4)
  1 : CER  (µm,   z-score normalised in __getitem__)
  2 : COT  (dimensionless, z-score normalised)
  3 : CTH  (m,   z-score normalised)
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg
from sample_filters import get_patch_supervision_thresholds, patch_passes_supervision

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation statistics I/O
# ─────────────────────────────────────────────────────────────────────────────

class NormStats:
    """Container for per-channel mean/std + output percentiles."""

    def __init__(self,
                 agri_mean: np.ndarray, agri_std: np.ndarray,
                 out_mean:  np.ndarray, out_std:  np.ndarray,
                 out_q5:    np.ndarray, out_q95:  np.ndarray):
        self.agri_mean = agri_mean.astype(np.float32)
        self.agri_std  = agri_std.astype(np.float32)
        self.out_mean  = out_mean.astype(np.float32)
        self.out_std   = out_std.astype(np.float32)
        self.out_q5    = out_q5.astype(np.float32)
        self.out_q95   = out_q95.astype(np.float32)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 agri_mean=self.agri_mean, agri_std=self.agri_std,
                 out_mean=self.out_mean,   out_std=self.out_std,
                 out_q5=self.out_q5,       out_q95=self.out_q95)
        log.info("Saved normalisation stats → %s", path)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = np.load(path)
        return cls(d["agri_mean"], d["agri_std"],
                   d["out_mean"],  d["out_std"],
                   d["out_q5"],    d["out_q95"])


# ─────────────────────────────────────────────────────────────────────────────
# Stats: per-file worker (runs in subprocess)
# ─────────────────────────────────────────────────────────────────────────────

# def _stats_worker(h5_path: str) -> Optional[dict]:
#     """
#     Open one paired HDF5 file and return partial sum/sumsq accumulators
#     plus a small reservoir sample for percentile estimation.
#
#     Runs in a worker process spawned by ProcessPoolExecutor.
#     Kept module-level so it is picklable on all platforms.
#     """
#     import h5py, numpy as np, config as cfg
#
#     MAX_SAMPLE = 4096   # pixels kept per file for reservoir
#
#     try:
#         # with h5py.File(h5_path, "r") as f:
#         #     bt_keys = sorted(f["AGRI/BT"].keys())
#         #     # Read channels one by one to avoid stacking a huge intermediate array
#         #     H, W = f[f"AGRI/BT/{bt_keys[0]}"].shape
#         #     n_agri = len(bt_keys)
#         #
#         #     sum_bt   = np.zeros(n_agri, dtype=np.float64)
#         #     sumsq_bt = np.zeros(n_agri, dtype=np.float64)
#         #
#         #     # Stack BT in one go (full scene is manageable per-channel)
#         #     BT = np.stack(
#         #         [f[f"AGRI/BT/{k}"][()].astype(np.float64) for k in bt_keys],
#         #         axis=-1
#         #     )  # (H, W, C)
#         #
#         #     CLP = f["Labels/CLP"][()].astype(np.float64)
#         #     CER = f["Labels/CER"][()].astype(np.float64)
#         #     COT = f["Labels/COT"][()].astype(np.float64)
#         #     CTH = f["Labels/CTH"][()].astype(np.float64)
#
#         with h5py.File(h5_path, "r") as f:
#             if "Samples" in f and "agri" in f["Samples"] and "labels" in f["Samples"]:
#                 BT = f["Samples/agri"][()].astype(np.float64)  # (N, C, H, W)
#                 lbl = f["Samples/labels"][()].astype(np.float64)  # (N, 4, H, W)
#                 n_agri = BT.shape[1]
#                 flat_bt = BT.transpose(0, 2, 3, 1).reshape(-1, n_agri)
#                 flat_out = lbl.transpose(0, 2, 3, 1).reshape(-1, 4)
#             else:
#                 bt_keys = sorted(f["AGRI/BT"].keys())
#                 H, W = f[f"AGRI/BT/{bt_keys[0]}"].shape
#                 n_agri = len(bt_keys)
#
#                 BT = np.stack(
#                     [f[f"AGRI/BT/{k}"][()].astype(np.float64) for k in bt_keys],
#                     axis=-1
#                 )
#
#                 CLP = f["Labels/CLP"][()].astype(np.float64)
#                 CER = f["Labels/CER"][()].astype(np.float64)
#                 COT = f["Labels/COT"][()].astype(np.float64)
#                 CTH = f["Labels/CTH"][()].astype(np.float64)
#
#                 # out = np.stack([CLP, CER, COT, CTH], axis=-1)
#                 # flat_bt = BT.reshape(-1, n_agri)
#                 # flat_out = out.reshape(-1, 4)
#
#     except Exception as exc:
#         return None   # silently skip; caller will log
#
#     out = np.stack([CLP, CER, COT, CTH], axis=-1)   # (H, W, 4)
#     flat_bt  = BT.reshape(-1, n_agri)
#     flat_out = out.reshape(-1, 4)
#
#     # Keep only pixels where BOTH BT and labels are fully finite
#     valid = np.isfinite(flat_bt).all(axis=1) & np.isfinite(flat_out).all(axis=1)
#     if valid.sum() == 0:
#         return None
#
#     flat_bt  = flat_bt[valid]
#     flat_out = flat_out[valid]
#     n        = flat_bt.shape[0]
#
#     sum_bt   = flat_bt.sum(axis=0)
#     sumsq_bt = (flat_bt ** 2).sum(axis=0)
#     sum_out  = flat_out.sum(axis=0)
#     sumsq_out = (flat_out ** 2).sum(axis=0)
#
#     # Reservoir: subsample regression values (CER/COT/CTH)
#     reg = flat_out[:, 1:]   # (n, 3)
#     if n > MAX_SAMPLE:
#         idx = np.random.choice(n, MAX_SAMPLE, replace=False)
#         reg = reg[idx]
#
#     return dict(
#         n=n,
#         sum_bt=sum_bt, sumsq_bt=sumsq_bt,
#         sum_out=sum_out, sumsq_out=sumsq_out,
#         reg_sample=reg,
#         path=h5_path,
#     )

def _stats_worker(h5_path: str) -> Optional[dict]:
    import h5py, numpy as np

    MAX_SAMPLE = 4096

    try:
        with h5py.File(h5_path, "r") as f:
            if "Samples" in f and "agri" in f["Samples"] and "labels" in f["Samples"]:
                # 新格式: Samples/agri -> (N, C, H, W), Samples/labels -> (N, 4, H, W)
                BT = f["Samples/agri"][()].astype(np.float64)
                lbl = f["Samples/labels"][()].astype(np.float64)

                n_agri = BT.shape[1]
                flat_bt = BT.transpose(0, 2, 3, 1).reshape(-1, n_agri)
                flat_out = lbl.transpose(0, 2, 3, 1).reshape(-1, 4)

            else:
                # 旧格式: AGRI/BT + Labels/*
                bt_keys = sorted(f["AGRI/BT"].keys())
                n_agri = len(bt_keys)

                BT = np.stack(
                    [f[f"AGRI/BT/{k}"][()].astype(np.float64) for k in bt_keys],
                    axis=-1,   # (H, W, C)
                )

                CLP = f["Labels/CLP"][()].astype(np.float64)
                CER = f["Labels/CER"][()].astype(np.float64)
                COT = f["Labels/COT"][()].astype(np.float64)
                CTH = f["Labels/CTH"][()].astype(np.float64)

                out = np.stack([CLP, CER, COT, CTH], axis=-1)  # (H, W, 4)
                flat_bt = BT.reshape(-1, n_agri)
                flat_out = out.reshape(-1, 4)

    except Exception as exc:
        log.warning("Stats worker failed for %s: %s", h5_path, exc)
        return None

    # 只保留输入和标签都有限的像素
    valid = np.isfinite(flat_bt).all(axis=1) & np.isfinite(flat_out).all(axis=1)
    if valid.sum() == 0:
        return None

    flat_bt = flat_bt[valid]
    flat_out = flat_out[valid]
    n = flat_bt.shape[0]

    sum_bt = flat_bt.sum(axis=0)
    sumsq_bt = (flat_bt ** 2).sum(axis=0)
    sum_out = flat_out.sum(axis=0)
    sumsq_out = (flat_out ** 2).sum(axis=0)

    reg = flat_out[:, 1:]   # CER/COT/CTH
    if n > MAX_SAMPLE:
        idx = np.random.choice(n, MAX_SAMPLE, replace=False)
        reg = reg[idx]

    return {
        "n": n,
        "sum_bt": sum_bt,
        "sumsq_bt": sumsq_bt,
        "sum_out": sum_out,
        "sumsq_out": sumsq_out,
        "reg_sample": reg,
        "path": h5_path,
    }


def compute_and_save_stats(
    paired_dir: Path,
    out_path: Path = cfg.STATS_FILE,
    n_workers: int = min(8, os.cpu_count() or 1),
) -> "NormStats":
    """
    Compute normalisation statistics from all paired HDF5 files under `paired_dir`.

    Optimisations vs. original:
    1. Parallel file reading via ProcessPoolExecutor (n_workers subprocesses).
       Each worker returns partial accumulators; the main process reduces them.
       This typically gives a 4-8× speedup on multi-core machines.
    2. Only valid (fully-finite) pixels contribute to mean/std.
    3. Reservoir for percentile estimation is capped at 500 000 pixels total.
    """
    log.info("Computing normalisation statistics from %s  (workers=%d)", paired_dir, n_workers)

    h5_files = sorted(paired_dir.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {paired_dir}")

    n_agri = cfg.AGRI_CHANNELS
    n_out  = 4

    total_n   = 0
    sum_bt    = np.zeros(n_agri, dtype=np.float64)
    sumsq_bt  = np.zeros(n_agri, dtype=np.float64)
    sum_out   = np.zeros(n_out, dtype=np.float64)
    sumsq_out = np.zeros(n_out, dtype=np.float64)

    MAX_RESERVOIR = 500_000
    reservoir: List[np.ndarray] = []
    reservoir_n = 0

    paths = [str(p) for p in h5_files]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_stats_worker, p): p for p in paths}
        done = 0
        for fut in as_completed(futures):
            done += 1
            p = futures[fut]
            result = fut.result()
            if result is None:
                log.warning("Skip %s (worker returned None)", p)
                continue

            n = result["n"]
            total_n   += n
            sum_bt    += result["sum_bt"]
            sumsq_bt  += result["sumsq_bt"]
            sum_out   += result["sum_out"]
            sumsq_out += result["sumsq_out"]

            reg = result["reg_sample"]
            reservoir.append(reg)
            reservoir_n += reg.shape[0]

            # Trim reservoir to avoid unbounded growth
            if reservoir_n > MAX_RESERVOIR:
                tmp = np.concatenate(reservoir, axis=0)
                idx = np.random.choice(tmp.shape[0], MAX_RESERVOIR, replace=False)
                reservoir  = [tmp[idx]]
                reservoir_n = MAX_RESERVOIR

            if done % 10 == 0 or done == len(paths):
                log.info("  %d / %d files processed  (total valid px so far: %d)",
                         done, len(paths), total_n)

    if total_n < 2:
        raise RuntimeError("Not enough valid pixels to compute statistics.")

    agri_mean = (sum_bt  / total_n).astype(np.float32)
    out_mean  = (sum_out / total_n).astype(np.float32)

    agri_var = (sumsq_bt  - (sum_bt  ** 2) / total_n) / (total_n - 1)
    out_var  = (sumsq_out - (sum_out ** 2) / total_n) / (total_n - 1)

    agri_std = np.sqrt(np.maximum(agri_var, 1e-12)).astype(np.float32)
    out_std  = np.sqrt(np.maximum(out_var,  1e-12)).astype(np.float32)

    all_reg = np.concatenate(reservoir, axis=0)   # (N_total, 3)
    q5  = np.concatenate([[0.0], np.percentile(all_reg,  5, axis=0)])
    q95 = np.concatenate([[4.0], np.percentile(all_reg, 95, axis=0)])

    stats = NormStats(
        agri_mean=agri_mean, agri_std=agri_std,
        out_mean=out_mean,   out_std=out_std,
        out_q5=q5.astype(np.float32), out_q95=q95.astype(np.float32),
    )
    stats.save(out_path)
    log.info("Stats computed from %d valid pixels across %d files", total_n, len(h5_files))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Patch index builder (runs once in __init__)
# ─────────────────────────────────────────────────────────────────────────────

def _build_patch_index(
    h5_files: List[Path],
    patch_size: Tuple[int, int],
    mode: str,
) -> List[Tuple[Path, int, int]]:
    """
    Scan all HDF5 files and return a list of (file_path, i_start, j_start)
    tuples that will form the dataset.

    Patch filtering is shared with data_fusion.py via sample_filters.py so that
    train / val / test all follow the same supervision thresholds and no stale
    hard-coded `16 pixels` rule remains in the runtime dataset path.
    """
    ph, pw = patch_size
    index: List[Tuple[Path, int, int]] = []

    if mode == "train":
        sh, sw = max(1, ph // 2), max(1, pw // 2)   # 50% overlap
    else:
        sh, sw = ph, pw                              # non-overlapping

    thresholds = get_patch_supervision_thresholds(mode, patch_size)

    for h5f in h5_files:
        try:
            with h5py.File(h5f, "r") as f:
                if "Samples" in f and "agri" in f["Samples"] and "labels" in f["Samples"]:
                    samples = f["Samples"]
                    n_samples = int(samples["agri"].shape[0])

                    has_cached_counts = (
                        "valid_clp_pixels" in samples and "valid_cloudy_pixels" in samples
                    )
                    if has_cached_counts:
                        valid_label_pixels = samples["valid_clp_pixels"][()]
                        valid_cloudy_pixels = samples["valid_cloudy_pixels"][()]
                        for s in range(n_samples):
                            if (
                                int(valid_label_pixels[s]) >= thresholds["min_valid_label_pixels"]
                                and int(valid_cloudy_pixels[s]) >= thresholds["min_valid_cloudy_pixels"]
                            ):
                                index.append((h5f, s, -1))
                    else:
                        for s in range(n_samples):
                            patch_clp, patch_cer, patch_cot, patch_cth = samples["labels"][s]
                            keep, _counts, _ = patch_passes_supervision(
                                patch_clp, patch_cer, patch_cot, patch_cth, mode, patch_size
                            )
                            if keep:
                                index.append((h5f, s, -1))
                    continue

                CLP = f["Labels/CLP"][()]
                CER = f["Labels/CER"][()]
                COT = f["Labels/COT"][()]
                CTH = f["Labels/CTH"][()]
                H, W = CLP.shape

                h_positions = list(range(0, H - ph + 1, sh))
                if h_positions and h_positions[-1] != H - ph:
                    h_positions.append(H - ph)

                w_positions = list(range(0, W - pw + 1, sw))
                if w_positions and w_positions[-1] != W - pw:
                    w_positions.append(W - pw)

                for i in h_positions:
                    for j in w_positions:
                        patch_clp = CLP[i:i + ph, j:j + pw]
                        patch_cer = CER[i:i + ph, j:j + pw]
                        patch_cot = COT[i:i + ph, j:j + pw]
                        patch_cth = CTH[i:i + ph, j:j + pw]

                        keep, _counts, _ = patch_passes_supervision(
                            patch_clp, patch_cer, patch_cot, patch_cth, mode, patch_size
                        )
                        if keep:
                            index.append((h5f, i, j))

        except Exception as exc:
            log.warning("Skip %s during index build: %s", h5f, exc)
            continue

    return index


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AGRIMyd06Dataset(Dataset):
    """
    PyTorch Dataset for AGRI → (CLP, CER, COT, CTH) retrieval.

    **Lazy-loading design**: only a lightweight patch index (file path + pixel
    offsets) is built at construction time.  The actual HDF5 read happens inside
    `__getitem__` using HDF5 hyperslab slicing, so memory usage is O(1) in the
    number of files rather than O(N_files × scene_size).

    Each item is a tuple:
        agri   : FloatTensor  (n_agri_channels, patch_H, patch_W) – z-score normalised
        geo    : FloatTensor  (3, patch_H, patch_W)  [lat, lon, ELE] – raw
        geo    : FloatTensor  (2, patch_H, patch_W)  [lat, lon] – raw
        labels : FloatTensor  (4, patch_H, patch_W)
                   ch0 = CLP (float, integer class 0-4)
                   ch1 = CER (µm,  z-score normalised, NaN for clear/missing)
                   ch2 = COT (z-score normalised, NaN for clear/missing)
                   ch3 = CTH (m,   z-score normalised, NaN for clear/missing)
    """

    def __init__(self,
                 paired_dir: Path,
                 stats: NormStats,
                 patch_size: Tuple[int, int] = cfg.PATCH_SIZE,
                 mode: str = "train"):
        self.stats      = stats
        self.patch_size = patch_size
        self.mode       = mode
        self.ph, self.pw = patch_size

        h5_files = sorted(paired_dir.rglob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {paired_dir}")

        thresholds = get_patch_supervision_thresholds(mode, patch_size)
        log.info(
            "Building patch index from %d files in %s (mode=%s, min_valid_label=%d, min_valid_cloudy=%d) …",
            len(h5_files),
            paired_dir,
            mode,
            thresholds["min_valid_label_pixels"],
            thresholds["min_valid_cloudy_pixels"],
        )

        # Build lightweight index – does NOT load pixel data into RAM
        self._index = _build_patch_index(h5_files, patch_size, mode)

        log.info("Dataset ready – %d patches (from %d files, mode=%s)",
                 len(self._index), len(h5_files), mode)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        h5f, i, j = self._index[idx]
        ph, pw = self.ph, self.pw

        # ── Read only the required patch via HDF5 hyperslab ───────────────
        try:
            with h5py.File(h5f, "r") as f:
                if j < 0 and "Samples" in f and "agri" in f["Samples"]:
                    agri_patch = f["Samples/agri"][i].astype(np.float32)  # (C, H, W)
                    geo_patch = f["Samples/geo"][i].astype(np.float32)  # (4, H, W)
                    label_patch = f["Samples/labels"][i].astype(np.float32)  # (4, H, W)

                    BT = agri_patch.transpose(1, 2, 0)
                    lat = geo_patch[0]
                    lon = geo_patch[1]
                    CLP, CER, COT, CTH = label_patch
                else:
                    bt_keys = sorted(f["AGRI/BT"].keys())
                    bt_patches = [
                        f[f"AGRI/BT/{k}"][i:i + ph, j:j + pw].astype(np.float32)
                        for k in bt_keys
                    ]
                    BT = np.stack(bt_patches, axis=-1)

                    lat = f["AGRI/Geolocation/lat"][i:i + ph, j:j + pw].astype(np.float32)
                    lon = f["AGRI/Geolocation/lon"][i:i + ph, j:j + pw].astype(np.float32)

                    CLP = f["Labels/CLP"][i:i + ph, j:j + pw].astype(np.float32)
                    CER = f["Labels/CER"][i:i + ph, j:j + pw].astype(np.float32)
                    COT = f["Labels/COT"][i:i + ph, j:j + pw].astype(np.float32)
                    CTH = f["Labels/CTH"][i:i + ph, j:j + pw].astype(np.float32)
            # with h5py.File(h5f, "r") as f:
            #     bt_keys = sorted(f["AGRI/BT"].keys())
            #     # Each channel is a 2-D (H, W) dataset; slice individually.
            #     bt_patches = [
            #         f[f"AGRI/BT/{k}"][i:i + ph, j:j + pw].astype(np.float32)
            #         for k in bt_keys
            #     ]
            #     BT = np.stack(bt_patches, axis=-1)   # (ph, pw, C)
            #
            #     lat = f["AGRI/Geolocation/lat"][i:i + ph, j:j + pw].astype(np.float32)
            #     lon = f["AGRI/Geolocation/lon"][i:i + ph, j:j + pw].astype(np.float32)
            #     # ele = f["AGRI/Aux/ELE"][i:i + ph, j:j + pw].astype(np.float32)
            #
            #     CLP = f["Labels/CLP"][i:i + ph, j:j + pw].astype(np.float32)
            #     CER = f["Labels/CER"][i:i + ph, j:j + pw].astype(np.float32)
            #     COT = f["Labels/COT"][i:i + ph, j:j + pw].astype(np.float32)
            #     CTH = f["Labels/CTH"][i:i + ph, j:j + pw].astype(np.float32)

        except Exception as exc:
            # Return a zero tensor on read failure (rare but resilient)
            log.warning("Read error at %s [%d,%d]: %s", h5f.name, i, j, exc)
            C = len(cfg.AGRI_BT_CHANNEL_INDICES)
            agri_t = torch.zeros(C, ph, pw, dtype=torch.float32)
            # geo_t  = torch.zeros(3, ph, pw, dtype=torch.float32)
            geo_t = torch.zeros(2, ph, pw, dtype=torch.float32)
            lbl_t  = torch.full((4, ph, pw), float("nan"), dtype=torch.float32)
            return agri_t, geo_t, lbl_t

        # ── Label QC (per-channel NaN masking) ──────────────────────────
        bad_clp = (CLP < 0) | (CLP >= cfg.CLP_CLASSES)
        bad_cer = (CER < 0) | (CER > 100)
        bad_cot = (COT < 0) | (COT > 200)
        bad_cth = (CTH < 0) | (CTH > 25000)

        CLP[bad_clp] = np.nan
        CER[bad_cer] = np.nan
        COT[bad_cot] = np.nan
        CTH[bad_cth] = np.nan

        # ── Normalise BT (z-score) ────────────────────────────────────────
        agri_norm = (BT - self.stats.agri_mean) / (self.stats.agri_std + 1e-8)
        agri_norm = np.nan_to_num(agri_norm, nan=0.0)

        # ── Normalise regression labels; keep CLP raw; keep NaN in labels ─
        lbl = np.stack([CLP, CER, COT, CTH], axis=-1)   # (ph, pw, 4)
        lbl[..., 1:] = (lbl[..., 1:] - self.stats.out_mean[1:]) / (self.stats.out_std[1:] + 1e-8)

        # geo = np.stack([lat, lon, ele], axis=-1)
        geo = np.stack([lat, lon], axis=-1)
        geo = np.nan_to_num(geo, nan=0.0)

        # ── Data augmentation (train only) ────────────────────────────────
        if self.mode == "train":
            if np.random.rand() < 0.5:
                agri_norm = np.flip(agri_norm, axis=1).copy()
                geo       = np.flip(geo,       axis=1).copy()
                lbl       = np.flip(lbl,       axis=1).copy()
            if np.random.rand() < 0.5:
                agri_norm = np.flip(agri_norm, axis=0).copy()
                geo       = np.flip(geo,       axis=0).copy()
                lbl       = np.flip(lbl,       axis=0).copy()
            k = np.random.randint(0, 4)
            if k:
                agri_norm = np.rot90(agri_norm, k=k, axes=(0, 1)).copy()
                geo       = np.rot90(geo,       k=k, axes=(0, 1)).copy()
                lbl       = np.rot90(lbl,       k=k, axes=(0, 1)).copy()

        # ── (H, W, C) → (C, H, W) for PyTorch ───────────────────────────
        agri_t = torch.from_numpy(agri_norm.transpose(2, 0, 1))
        geo_t  = torch.from_numpy(geo.transpose(2, 0, 1))
        lbl_t  = torch.from_numpy(lbl.transpose(2, 0, 1))

        return agri_t, geo_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(stats: NormStats):
    """Return train / val / test DataLoaders using config paths."""
    train_ds = AGRIMyd06Dataset(cfg.PAIRED_TRAIN_DIR, stats, mode="train")
    val_ds   = AGRIMyd06Dataset(cfg.PAIRED_VAL_DIR,   stats, mode="val")
    test_ds  = AGRIMyd06Dataset(cfg.PAIRED_TEST_DIR,  stats, mode="test")

    common   = dict(pin_memory=True, num_workers=cfg.NUM_WORKERS)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **common)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **common)
    test_dl  = DataLoader(test_ds,  batch_size=1,              shuffle=False, **common)

    return train_dl, val_dl, test_dl
