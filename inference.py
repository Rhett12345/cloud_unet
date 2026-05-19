"""
inference.py
============
Full-disk inference using a sliding-window patch strategy with overlap blending.

For each AGRI scene file:
  1. Reads raw AGRI BT + geolocation (lat, lon, VZA, SZA).
  2. Slices into overlapping patches (cfg.PATCH_SIZE, cfg.PATCH_OVERLAP).
  3. Runs CloudPropertyNet in batch mode.
  4. Reassembles predictions via Gaussian-weighted averaging in overlap regions.
  5. Saves outputs as a compressed .npz file.

Output .npz keys:
    latitude, longitude
    CLP_pred   : integer class map (H, W)   0=Clear, 1=Water, 2=Ice
    CTH_pred   : m             (H, W)        NaN where clear/invalid
    CLP_prob   : (H, W, 3)     softmax probabilities

Usage (standalone):
    python inference.py --agri_file /path/to/FY4A_..._FDI_...HDF

Usage (called from main.py): --stages infer --agri_file ...
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from fusion_io import read_agri_scene
from dataset import NormStats
from model import build_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _gaussian_weight_map(ph: int, pw: int) -> np.ndarray:
    """2-D Gaussian window centred on the patch, for overlap blending."""
    sigma_h, sigma_w = ph / 4.0, pw / 4.0
    yy = np.arange(ph) - ph / 2.0
    xx = np.arange(pw) - pw / 2.0
    xx, yy = np.meshgrid(xx, yy)
    w = np.exp(-(xx ** 2 / (2 * sigma_w ** 2) + yy ** 2 / (2 * sigma_h ** 2)))
    return w.astype(np.float32)


def _extract_patches(arr: np.ndarray, ph: int, pw: int, stride_h: int, stride_w: int):
    """Slide over arr (H, W, C) and yield (patch, i_start, j_start)."""
    H, W = arr.shape[:2]
    for i in range(0, H - ph + 1, stride_h):
        for j in range(0, W - pw + 1, stride_w):
            yield arr[i:i + ph, j:j + pw, :], i, j


def _stitch(pred_sum: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    """Normalise accumulated predictions by accumulated weights."""
    wt = weight_sum[..., np.newaxis] if pred_sum.ndim == 3 else weight_sum
    return np.where(wt > 0, pred_sum / np.maximum(wt, 1e-8), np.nan)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(agri_file: Path,
                  stats: NormStats,
                  checkpoint: Optional[Path] = None,
                  out_dir: Optional[Path] = None,
                  batch_size: int = 64) -> Path:
    """Produce full-disk retrieval for one AGRI scene file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    out_dir = out_dir or cfg.RETRIEVAL_DIR

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded model from %s", checkpoint)

    # ── Read AGRI (支持原始 FDI HDF5 和转换后的 .npz) ──────────────────
    if agri_file.suffix.lower() in (".npz",):
        # 转换后的 FY-4B→FY-4A 数据
        log.info("Loading converted .npz: %s", agri_file.name)
        d = np.load(agri_file, allow_pickle=True)
        BT  = d["BT_converted"] if "BT_converted" in d else d["BT"]
        lat = d["latitude"]; lon = d["longitude"]
        vza = d.get("VZA", np.zeros_like(lat))
        sza = d.get("SZA", np.zeros_like(lat))
    else:
        agri = read_agri_scene(agri_file)
        if agri is None:
            raise RuntimeError(f"Failed to read {agri_file}")
        BT  = agri["BT"]
        lat = agri["lat"]
        lon = agri["lon"]
        vza = agri.get("VZA")
        sza = agri.get("SZA")
    H, W = BT.shape[:2]

    if vza is None or sza is None:
        log.warning("VZA/SZA missing from GEO; filling with zeros")
        vza = np.zeros((H, W), dtype=np.float32)
        sza = np.zeros((H, W), dtype=np.float32)

    # ── Normalise BT globally ─────────────────────────────────────────────
    BT_norm = (BT - stats.agri_mean) / (stats.agri_std + 1e-8)

    # ── Build geo stack (4 channels, same normalisation as training) ──────
    # lat/90, lon/180, VZA/90, SZA/90
    geo = np.stack([
        lat / 90.0,
        lon / 180.0,
        vza / 90.0,
        sza / 90.0,
    ], axis=-1).astype(np.float32)
    geo = np.nan_to_num(geo, nan=0.0)

    # ── Patch geometry ────────────────────────────────────────────────────
    ph, pw   = cfg.PATCH_SIZE
    overlap  = cfg.PATCH_OVERLAP
    stride_h = max(1, ph - overlap)
    stride_w = max(1, pw - overlap)
    wmap     = _gaussian_weight_map(ph, pw)   # (ph, pw)

    # ── Accumulation buffers ──────────────────────────────────────────────
    clp_sum     = np.zeros((H, W, cfg.CLP_CLASSES), dtype=np.float32)   # 3 classes
    cth_sum     = np.zeros((H, W), dtype=np.float32)                     # 1 channel
    weight_sum  = np.zeros((H, W), dtype=np.float32)

    # ── Batch-wise inference ──────────────────────────────────────────────
    patches_buf, geo_buf, positions_buf = [], [], []

    def _flush():
        """Process accumulated patch batch."""
        if not patches_buf:
            return
        # AGRI BT: (B, C, ph, pw)
        x = torch.from_numpy(
            np.stack(patches_buf, axis=0).transpose(0, 3, 1, 2)
        ).to(device)
        # Geo: (B, 4, ph, pw)
        g = torch.from_numpy(
            np.stack(geo_buf, axis=0).transpose(0, 3, 1, 2)
        ).to(device)

        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                clp_logits, comp_norm = model(x, geo=g)

        clp_prob = F.softmax(clp_logits, dim=1).cpu().numpy()   # (B, 3, ph, pw)

        # Denormalise CTH: comp_norm is (B, 1, ph, pw)
        cth_std  = stats.out_std[1]   # scalar
        cth_mean = stats.out_mean[1]  # scalar
        cth_dn   = comp_norm.cpu().numpy()[:, 0, :, :] * cth_std + cth_mean  # (B, ph, pw)

        for b, (si, sj) in enumerate(positions_buf):
            clp_sum[si:si+ph, sj:sj+pw, :] += clp_prob[b].transpose(1, 2, 0) * wmap[..., np.newaxis]
            cth_sum[si:si+ph, sj:sj+pw]    += cth_dn[b] * wmap
            weight_sum[si:si+ph, sj:sj+pw] += wmap

        patches_buf.clear()
        geo_buf.clear()
        positions_buf.clear()

    # ── Sliding window + batched forward ──────────────────────────────────
    for patch, si, sj in _extract_patches(BT_norm, ph, pw, stride_h, stride_w):
        nan_ratio = np.isnan(patch).mean()
        if nan_ratio > 0.8:
            continue

        geo_patch = geo[si:si+ph, sj:sj+pw, :]

        # Fill NaN with zeros before inference
        patch_filled = np.where(np.isnan(patch), 0.0, patch)
        patches_buf.append(patch_filled)
        geo_buf.append(geo_patch)
        positions_buf.append((si, sj))

        if len(patches_buf) >= batch_size:
            _flush()

    _flush()  # remaining

    # ── Stitch ────────────────────────────────────────────────────────────
    clp_prob_map = _stitch(clp_sum, weight_sum)   # (H, W, 3)
    CTH_pred     = _stitch(cth_sum[..., np.newaxis], weight_sum)[..., 0]  # (H, W)

    CLP_pred = np.full(clp_prob_map.shape[:2], -1, dtype=np.int16)
    valid_mask = np.isfinite(clp_prob_map).any(axis=-1)
    if valid_mask.any():
        CLP_pred[valid_mask] = np.nanargmax(clp_prob_map[valid_mask], axis=-1).astype(np.int16)

    # ── Physical clipping ─────────────────────────────────────────────────
    CTH_pred = np.clip(CTH_pred, 0, 20000)
    CTH_pred[CLP_pred <= 0] = np.nan  # clear-sky or invalid → no height

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = agri_file.stem
    out_path = out_dir / f"{stem}_retrieval.npz"

    np.savez_compressed(
        out_path,
        latitude=lat,
        longitude=lon,
        CLP_pred=CLP_pred,
        CTH_pred=CTH_pred.astype(np.float32),
        CLP_prob=clp_prob_map.astype(np.float32),
    )
    log.info("Saved retrieval → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Full-disk AGRI cloud retrieval")
    parser.add_argument("--agri_file", default=None, help="Path to a single AGRI FDI HDF5 file")
    parser.add_argument("--agri_dir",  default=None, help="Path to directory of AGRI FDI HDF5 files (batch)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    stats = NormStats.load(cfg.STATS_FILE)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    out_d = Path(args.out_dir) if args.out_dir else None

    if args.agri_dir:
        agri_dir = Path(args.agri_dir)
        agri_files = sorted(
            list(agri_dir.glob("*.HDF")) + list(agri_dir.glob("*.hdf"))
        )
        agri_files = [f for f in agri_files if "_FDI-_" in f.name]
        log.info("Batch inference on %d files in %s", len(agri_files), agri_dir)
        for f in agri_files:
            try:
                run_inference(f, stats, ckpt, out_d)
            except Exception as exc:
                log.error("Failed for %s: %s", f.name, exc)
    elif args.agri_file:
        run_inference(
            agri_file=Path(args.agri_file),
            stats=stats,
            checkpoint=ckpt,
            out_dir=out_d,
            batch_size=args.batch_size,
        )
    else:
        parser.error("Either --agri_file or --agri_dir required")


if __name__ == "__main__":
    main()
