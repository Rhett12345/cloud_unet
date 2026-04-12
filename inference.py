"""
inference.py
============
Full-disk inference using a sliding-window patch strategy with overlap blending.

For each AGRI scene file:
  1. Reads raw AGRI BT + geolocation.
  2. Slices into overlapping patches (cfg.PATCH_SIZE, cfg.PATCH_OVERLAP).
  3. Runs CloudPropertyNet in batch mode.
  4. Reassembles predictions via Gaussian-weighted averaging in overlap regions.
  5. Saves outputs as a compressed .npz file.

Output .npz keys:
    latitude, longitude
    CLP_pred   : integer class map (H, W)
    CER_pred   : µm           (H, W)
    COT_pred   : dimensionless (H, W)
    CTH_pred   : m             (H, W)
    CLP_prob   : (H, W, 5)  softmax probabilities

Usage (called by main.py or standalone):
    python inference.py --agri_file /path/to/FY4B_AGRI_*.HDF [--checkpoint <path>]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from data_fusion import read_agri_scene
from dataset import NormStats
from model import build_model

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_weight_map(ph: int, pw: int) -> np.ndarray:
    """2-D Gaussian window centred on the patch, for overlap blending."""
    sigma_h, sigma_w = ph / 4.0, pw / 4.0
    yy = np.arange(ph) - ph / 2.0
    xx = np.arange(pw) - pw / 2.0
    xx, yy = np.meshgrid(xx, yy)
    w = np.exp(-(xx ** 2 / (2 * sigma_w ** 2) + yy ** 2 / (2 * sigma_h ** 2)))
    return w.astype(np.float32)


def _extract_patches(BT: np.ndarray, ph: int, pw: int, stride_h: int, stride_w: int):
    """
    Slide over BT (H, W, C) and yield (patch, i_start, j_start).
    """
    H, W, _ = BT.shape
    for i in range(0, H - ph + 1, stride_h):
        for j in range(0, W - pw + 1, stride_w):
            patch = BT[i:i + ph, j:j + pw, :]
            yield patch, i, j


def _stitch(pred_sum: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    """Normalise accumulated predictions by accumulated weights."""
    wt = weight_sum[..., np.newaxis] if pred_sum.ndim == 3 else weight_sum
    return np.where(wt > 0, pred_sum / np.maximum(wt, 1e-8), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional
def run_inference(agri_file: Path,
                  stats: NormStats,
                  checkpoint: Optional[Path] = None,
                  out_dir: Optional[Path] = None,
                  batch_size: int = 64) -> Path:
    """
    Produce full-disk retrieval for one AGRI scene file.
    Returns path to the saved .npz file.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    out_dir    = out_dir    or cfg.RETRIEVAL_DIR

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    log.info("Loaded model from %s", checkpoint)

    # ── Read AGRI ─────────────────────────────────────────────────────────
    agri = read_agri_scene(agri_file)
    if agri is None:
        raise RuntimeError(f"Failed to read {agri_file}")

    BT  = agri["BT"]          # (H, W, n_ch)
    lat = agri["lat"]
    lon = agri["lon"]
    H, W, _ = BT.shape

    # ── Patch geometry ────────────────────────────────────────────────────
    ph, pw   = cfg.PATCH_SIZE
    overlap  = cfg.PATCH_OVERLAP
    stride_h = max(1, ph - overlap)
    stride_w = max(1, pw - overlap)
    wmap     = _gaussian_weight_map(ph, pw)   # (ph, pw)

    # ── Normalise BT globally ─────────────────────────────────────────────
    BT_norm = (BT - stats.agri_mean) / (stats.agri_std + 1e-8)

    # ── Accumulation buffers ──────────────────────────────────────────────
    # CLP: accumulate class probabilities (5 channels)
    clp_sum     = np.zeros((H, W, cfg.CLP_CLASSES), dtype=np.float32)
    # Regression: CER, COT, CTH
    comp_sum    = np.zeros((H, W, cfg.COMP_CHANNELS), dtype=np.float32)
    weight_sum  = np.zeros((H, W), dtype=np.float32)

    # ── Batch-wise inference ──────────────────────────────────────────────
    patches_buf, positions_buf = [], []

    def _flush(patches_buf, positions_buf):
        """Process accumulated patch batch."""
        if not patches_buf:
            return
        x = torch.from_numpy(
            np.stack(patches_buf, axis=0).transpose(0, 3, 1, 2)   # (B, C, ph, pw)
        ).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                clp_logits, comp_norm = model(x)

        clp_prob  = F.softmax(clp_logits, dim=1).cpu().numpy()   # (B, 5, ph, pw)
        comp_dn   = (
            comp_norm.cpu().numpy() * stats.out_std[1:].reshape(1, 3, 1, 1) +
            stats.out_mean[1:].reshape(1, 3, 1, 1)
        )  # (B, 3, ph, pw)

        for b, (si, sj) in enumerate(positions_buf):
            w = wmap[np.newaxis, :, :]          # (1, ph, pw)
            clp_sum[si:si+ph, sj:sj+pw, :] += clp_prob[b].transpose(1, 2, 0) * w[0, :, :, np.newaxis]
            comp_sum[si:si+ph, sj:sj+pw, :] += comp_dn[b].transpose(1, 2, 0) * w[0, :, :, np.newaxis]
            weight_sum[si:si+ph, sj:sj+pw]  += wmap

        patches_buf.clear()
        positions_buf.clear()

    for patch, si, sj in _extract_patches(BT_norm, ph, pw, stride_h, stride_w):
        nan_ratio = np.isnan(patch).mean()
        if nan_ratio > 0.8:
            continue   # skip mostly-missing patches

        # Fill NaN with channel means before inference (will be masked in output)
        patch_filled = np.where(np.isnan(patch), 0.0, patch)
        patches_buf.append(patch_filled)
        positions_buf.append((si, sj))

        if len(patches_buf) >= batch_size:
            _flush(patches_buf, positions_buf)

    _flush(patches_buf, positions_buf)   # remaining

    # ── Stitch ────────────────────────────────────────────────────────────
    clp_prob_map = _stitch(clp_sum,  weight_sum)   # (H, W, 5)
    comp_map     = _stitch(comp_sum, weight_sum)   # (H, W, 3)

    CLP_pred = np.nanargmax(clp_prob_map, axis=-1).astype(np.int16)
    CER_pred = comp_map[..., 0]
    COT_pred = comp_map[..., 1]
    CTH_pred = comp_map[..., 2]

    # ── Physical clipping ─────────────────────────────────────────────────
    CER_pred = np.clip(CER_pred, 0, 100)
    COT_pred = np.clip(COT_pred, 0, 200)
    CTH_pred = np.clip(CTH_pred, 0, 25000)

    # Mask regression for clear pixels
    clear_mask = CLP_pred == 0
    for arr in [CER_pred, COT_pred, CTH_pred]:
        arr[clear_mask] = np.nan

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = agri_file.stem
    out_path = out_dir / f"{stem}_retrieval.npz"

    np.savez_compressed(
        out_path,
        latitude=lat,
        longitude=lon,
        CLP_pred=CLP_pred,
        CER_pred=CER_pred.astype(np.float32),
        COT_pred=COT_pred.astype(np.float32),
        CTH_pred=CTH_pred.astype(np.float32),
        CLP_prob=clp_prob_map.astype(np.float32),
    )
    log.info("Saved retrieval → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    parser = argparse.ArgumentParser(description="Full-disk AGRI cloud retrieval")
    parser.add_argument("--agri_file", required=True, help="Path to AGRI HDF file")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    stats = NormStats.load(cfg.STATS_FILE)
    run_inference(
        agri_file=Path(args.agri_file),
        stats=stats,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
