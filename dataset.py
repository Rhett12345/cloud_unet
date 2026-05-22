"""
dataset.py
==========
PyTorch Dataset for AGRI → GPM precipitation regression (tile-level).

Key design decisions
--------------------
- **Lazy loading**: __init__ builds a tile index. Each __getitem__ opens the
  HDF5 file, reads the required tile, and closes immediately.
- **NormStats** is pre-computed once and loaded from disk (BT channels only).
- Each sample: X=(9,128,128) BT+geo, Y=(1,128,128) precipitation (log1p).
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

log = logging.getLogger(__name__)


def _split_dates_for_mode(mode: str):
    date_map = {
        "train": getattr(cfg, "TRAIN_DATES", []),
        "val": getattr(cfg, "VAL_DATES", []),
        "test": getattr(cfg, "TEST_DATES", []),
    }
    return set(date_map.get(mode, []) or [])


def _filter_h5_files_by_dates(h5_files: List[Path], mode: str) -> List[Path]:
    dates = _split_dates_for_mode(mode)
    if not dates:
        return h5_files
    filtered = [p for p in h5_files if any(part in dates for part in p.parts)]
    log.info("Using %d/%d %s files after date filter",
             len(filtered), len(h5_files), mode)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation statistics I/O
# ─────────────────────────────────────────────────────────────────────────────

class NormStats:
    """Container for per-channel AGRI BT mean/std (7 channels only)."""

    def __init__(self, agri_mean: np.ndarray, agri_std: np.ndarray):
        self.agri_mean = agri_mean.astype(np.float32)
        self.agri_std  = agri_std.astype(np.float32)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, agri_mean=self.agri_mean, agri_std=self.agri_std)
        log.info("Saved normalisation stats → %s", path)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = np.load(path)
        return cls(d["agri_mean"], d["agri_std"])


# ─────────────────────────────────────────────────────────────────────────────
# Stats: per-file worker
# ─────────────────────────────────────────────────────────────────────────────

def _stats_worker(h5_path: str) -> Optional[dict]:
    try:
        with h5py.File(h5_path, "r") as f:
            if "Tiles" not in f or "agri" not in f["Tiles"]:
                return None
            agri = f["Tiles/agri"][()].astype(np.float64)  # (N, 9, H, W)
            n_agri = cfg.AGRI_CHANNELS  # 7
            bt = agri[:, :n_agri, :, :]  # only BT channels
            flat_bt = bt.transpose(0, 2, 3, 1).reshape(-1, n_agri)
    except Exception:
        return None

    valid = np.isfinite(flat_bt).all(axis=1) & (flat_bt != 0.0).any(axis=1)
    n_bt = int(valid.sum())
    if n_bt == 0:
        return None
    bt_valid = flat_bt[valid]
    return {
        "n": n_bt,
        "sum_bt": bt_valid.sum(axis=0),
        "sumsq_bt": (bt_valid ** 2).sum(axis=0),
    }


def compute_and_save_stats(
    paired_dir: Path,
    out_path: Path = cfg.STATS_FILE,
    n_workers: int = min(8, os.cpu_count() or 1),
) -> "NormStats":
    """Compute normalisation statistics from all paired HDF5 tile files."""
    log.info("Computing normalisation statistics from %s (workers=%d)", paired_dir, n_workers)

    h5_files = _filter_h5_files_by_dates(sorted(paired_dir.rglob("*.h5")), "train")
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {paired_dir}")

    n_agri = cfg.AGRI_CHANNELS
    total_n = 0
    sum_bt   = np.zeros(n_agri, dtype=np.float64)
    sumsq_bt = np.zeros(n_agri, dtype=np.float64)

    paths = [str(p) for p in h5_files]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_stats_worker, p): p for p in paths}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is None:
                continue
            total_n += result["n"]
            sum_bt   += result["sum_bt"]
            sumsq_bt += result["sumsq_bt"]
            if done % 10 == 0 or done == len(paths):
                log.info("  %d / %d files processed", done, len(paths))

    if total_n < 2:
        raise RuntimeError(f"Not enough valid pixels: {total_n}")

    agri_mean = (sum_bt / total_n).astype(np.float32)
    agri_var  = (sumsq_bt - (sum_bt ** 2) / total_n) / (total_n - 1)
    agri_std  = np.sqrt(np.maximum(agri_var, 1e-12)).astype(np.float32)

    stats = NormStats(agri_mean=agri_mean, agri_std=agri_std)
    stats.save(out_path)
    log.info("Stats computed across %d files (valid px=%d)", len(h5_files), total_n)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Tile index builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_tile_index(h5_files: List[Path], mode: str) -> List[Tuple[Path, int, bool]]:
    """Scan all HDF5 files and return list of (file_path, tile_idx, has_rain) tuples."""
    index: List[Tuple[Path, int, bool]] = []
    for h5f in h5_files:
        try:
            with h5py.File(h5f, "r") as f:
                if "Tiles" not in f or "agri" not in f["Tiles"]:
                    continue
                n = int(f["Tiles/agri"].shape[0])
                has_rain_arr = f["Tiles/has_rain"][()] if "Tiles/has_rain" in f else None
                for s in range(n):
                    hr = bool(has_rain_arr[s]) if has_rain_arr is not None else False
                    index.append((h5f, s, hr))
        except Exception:
            continue
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PrecipTileDataset(Dataset):
    """
    PyTorch Dataset for AGRI → GPM precipitation regression (tile-level).

    Each item:
        x : FloatTensor (9, 128, 128)  — z-score BT + geo
        y : FloatTensor (1, 128, 128)  — precip mm/h (NaN→0)
    """

    def __init__(self,
                 paired_dir: Path,
                 stats: NormStats,
                 mode: str = "train"):
        self.stats = stats
        self.mode  = mode

        h5_files = _filter_h5_files_by_dates(sorted(paired_dir.rglob("*.h5")), mode)
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {paired_dir}")

        log.info("Building tile index from %d files (mode=%s)...", len(h5_files), mode)
        self._index = _build_tile_index(h5_files, mode)
        log.info("Dataset ready – %d tiles (from %d files, mode=%s)",
                 len(self._index), len(h5_files), mode)

        self._warned_files = set()
        self._fh_cache: dict = {}

    def __len__(self) -> int:
        return len(self._index)

    def _get_fh(self, h5f: Path) -> h5py.File:
        fh = self._fh_cache.get(h5f)
        if fh is None or not fh.id.valid:
            if len(self._fh_cache) >= 300:
                oldest = next(iter(self._fh_cache))
                try:
                    self._fh_cache[oldest].close()
                except Exception:
                    pass
                del self._fh_cache[oldest]
            fh = h5py.File(h5f, "r")
            self._fh_cache[h5f] = fh
        return fh

    def __getitem__(self, idx: int):
        h5f, s_idx, has_rain = self._index[idx]

        for attempt in range(10):
            try:
                fh = self._get_fh(h5f)
                tiles = fh["Tiles"]
                agri_tile = tiles["agri"][s_idx].astype(np.float32)  # (9, 128, 128)
                gpm_tile  = tiles["gpm"][s_idx].astype(np.float32)   # (1, 128, 128)
                break
            except Exception:
                try:
                    del self._fh_cache[h5f]
                except Exception:
                    pass
                if attempt == 0 and h5f.name not in self._warned_files:
                    log.warning("Read error at %s [%d]", h5f.name, s_idx)
                    self._warned_files.add(h5f)
                if attempt == 9:
                    log.error("All read retries exhausted, returning zero tile")
                    return (
                        torch.zeros(cfg.IN_CHANNELS, *cfg.TILE_SIZE),
                        torch.zeros(1, *cfg.TILE_SIZE),
                    )
                h5f, s_idx, has_rain = self._index[np.random.randint(0, len(self._index))]

        n_agri = cfg.AGRI_CHANNELS  # 7

        # ── BT normalisation (first 7 channels) ──
        agri_norm = agri_tile.copy()
        agri_norm[:n_agri] = (agri_norm[:n_agri] - self.stats.agri_mean[:, None, None]) / \
                              (self.stats.agri_std[:, None, None] + 1e-8)
        agri_norm = np.nan_to_num(agri_norm, nan=0.0)

        # ── GPM: fill NaN → 0, keep linear mm/h ──
        gpm = np.where(np.isfinite(gpm_tile), gpm_tile, 0.0).astype(np.float32)

        # ── Train augmentations ──
        if self.mode == "train":
            # Gaussian noise on BT channels
            noise = np.random.randn(n_agri, *cfg.TILE_SIZE).astype(np.float32) * 0.05
            agri_norm[:n_agri] = agri_norm[:n_agri] + noise

            # Channel dropout (20% prob, zero 1-2 BT channels)
            if np.random.rand() < 0.20:
                n_drop = np.random.randint(1, 3)
                drop_ch = np.random.choice(n_agri, size=n_drop, replace=False)
                agri_norm[drop_ch] = 0.0

            # Random flips (BT + geo + GPM all together)
            if np.random.rand() < 0.5:
                agri_norm = np.flip(agri_norm, axis=2).copy()
                gpm = np.flip(gpm, axis=2).copy()
            if np.random.rand() < 0.5:
                agri_norm = np.flip(agri_norm, axis=1).copy()
                gpm = np.flip(gpm, axis=1).copy()

            # Random 90° rotation
            k = np.random.randint(0, 4)
            if k:
                agri_norm = np.rot90(agri_norm, k=k, axes=(1, 2)).copy()
                gpm = np.rot90(gpm, k=k, axes=(1, 2)).copy()

        x = torch.from_numpy(agri_norm.copy())
        y = torch.from_numpy(gpm.copy())

        return x, y


def build_test_dataloader(stats: NormStats):
    """Return only the test DataLoader."""
    test_ds = PrecipTileDataset(cfg.PAIRED_TEST_DIR, stats, mode="test")
    return DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                      pin_memory=True, num_workers=cfg.NUM_WORKERS,
                      persistent_workers=True, prefetch_factor=4)
