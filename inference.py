"""
inference.py
============
Full-disk inference for precipitation regression (tile-based).

For each AGRI scene file:
  1. Reads raw AGRI BT + geolocation (lat, lon).
  2. Slices into 128×128 tiles with stride 64.
  3. Runs U-Net in batch mode.
  4. Reassembles predictions via Gaussian-weighted blending.
  5. Saves outputs as a compressed .npz file.

Output .npz keys:
    latitude, longitude
    precip_pred  : (H, W)     predicted precipitation (mm/h)
    precip_prob  : (H, W)     rain probability
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config as cfg
from fusion_io import read_agri_scene
from dataset import NormStats
from model import build_model

log = logging.getLogger(__name__)


def _build_region_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """构建区域 mask（True=在训练区域内）。"""
    try:
        import fusion_config as fc
        lat_min = float(getattr(fc, "REGION_LAT_MIN", -90))
        lat_max = float(getattr(fc, "REGION_LAT_MAX", 90))
        lon_min = float(getattr(fc, "REGION_LON_MIN", -180))
        lon_max = float(getattr(fc, "REGION_LON_MAX", 180))
    except ImportError:
        return np.ones(lat.shape, dtype=bool)

    if lat_min <= -89 and lat_max >= 89 and lon_min <= -179 and lon_max >= 179:
        return np.ones(lat.shape, dtype=bool)

    log.info("Inference region: lat=[%.1f, %.1f] lon=[%.1f, %.1f]",
             lat_min, lat_max, lon_min, lon_max)
    mask = (np.isfinite(lat) & np.isfinite(lon)
            & (lat >= lat_min) & (lat <= lat_max)
            & (lon >= lon_min) & (lon <= lon_max))
    return mask


def _gaussian_weight_map(th: int, tw: int) -> np.ndarray:
    """2D Gaussian weight map for tile blending."""
    sigma_h, sigma_w = th / 4.0, tw / 4.0
    yy = np.arange(th) - th / 2.0
    xx = np.arange(tw) - tw / 2.0
    xx, yy = np.meshgrid(xx, yy)
    w = np.exp(-(xx ** 2 / (2 * sigma_w ** 2) + yy ** 2 / (2 * sigma_h ** 2)))
    return w.astype(np.float32)


def _extract_tiles(arr: np.ndarray, th: int, tw: int, stride: int):
    """Generator yielding (tile, row, col) for sliding window."""
    H, W = arr.shape[:2]
    for i in range(0, H - th + 1, stride):
        for j in range(0, W - tw + 1, stride):
            yield arr[i:i + th, j:j + tw, ...], i, j


def run_inference(agri_file: Path,
                  stats: NormStats,
                  checkpoint: Optional[Path] = None,
                  out_dir: Optional[Path] = None,
                  batch_size: int = 32) -> Path:
    """Produce full-disk precipitation map for one AGRI scene."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    out_dir = out_dir or cfg.RETRIEVAL_DIR

    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded model from %s", checkpoint)

    # ── Read AGRI ──
    if agri_file.suffix.lower() in (".npz",):
        d = np.load(agri_file, allow_pickle=True)
        BT = d["BT_converted"] if "BT_converted" in d else d["BT"]
        lat = d["latitude"]; lon = d["longitude"]
    else:
        agri = read_agri_scene(agri_file)
        if agri is None:
            raise RuntimeError(f"Failed to read {agri_file}")
        BT = agri["BT"]
        lat = agri["lat"]
        lon = agri["lon"]

    H, W = BT.shape[:2]
    n_agri = cfg.AGRI_CHANNELS  # 7

    # ── Normalise BT (first 7 channels) ──
    BT_norm = (BT - stats.agri_mean) / (stats.agri_std + 1e-8)
    BT_norm = np.nan_to_num(BT_norm, nan=0.0).astype(np.float32)

    # ── Geo channels: lat/90, lon/180 ──
    geo_full = np.stack([
        np.nan_to_num(lat, nan=0.0) / 90.0,
        np.nan_to_num(lon, nan=0.0) / 180.0,
    ], axis=-1).astype(np.float32)

    # ── Combined feature array (H, W, 9) ──
    features = np.concatenate([BT_norm, geo_full], axis=-1)

    # ── Region mask ──
    region_mask = _build_region_mask(lat, lon)

    # ── Tile geometry ──
    th, tw = cfg.TILE_SIZE
    stride = cfg.INFERENCE_STRIDE
    wmap = _gaussian_weight_map(th, tw)

    # ── Accumulation buffers ──
    precip_sum  = np.zeros((H, W), dtype=np.float32)
    weight_sum  = np.zeros((H, W), dtype=np.float32)

    x_buf, positions_buf = [], []

    def _flush():
        if not x_buf:
            return
        x = torch.from_numpy(np.stack(x_buf, axis=0)).to(device)  # (B, 9, th, tw)

        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits, rain = model(x)

        prob = torch.sigmoid(logits)
        pred = (prob * rain).cpu().numpy()  # (B, 1, th, tw)

        for bi, (si, sj) in enumerate(positions_buf):
            precip_sum[si:si+th, sj:sj+tw] += pred[bi, 0] * wmap
            weight_sum[si:si+th, sj:sj+tw] += wmap

        x_buf.clear()
        positions_buf.clear()

    for tile, si, sj in _extract_tiles(features, th, tw, stride):
        nan_ratio = np.isnan(tile).mean()
        if nan_ratio > 0.8:
            continue
        if not region_mask[si:si+th, sj:sj+tw].any():
            continue

        tile_filled = np.nan_to_num(tile, nan=0.0)
        x_patch = np.ascontiguousarray(tile_filled.transpose(2, 0, 1))
        x_buf.append(x_patch)
        positions_buf.append((si, sj))

        if len(x_buf) >= batch_size:
            _flush()
    _flush()

    # ── Stitch ──
    precip_map = np.where(weight_sum > 0, precip_sum / np.maximum(weight_sum, 1e-8), np.nan)
    precip_map[~np.isfinite(lat)] = np.nan

    # ── Save ──
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = agri_file.stem
    out_path = out_dir / f"{stem}_precip.npz"

    np.savez_compressed(
        out_path,
        latitude=lat,
        longitude=lon,
        precip_pred=precip_map.astype(np.float32),
    )
    log.info("Saved retrieval → %s", out_path)
    log.info("Precip range: [%.2f, %.2f] mm/h, rain fraction: %.2f%%",
             np.nanmin(precip_map), np.nanmax(precip_map),
             np.nanmean(precip_map > cfg.RAIN_THRESHOLD) * 100)
    return out_path


def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Full-disk AGRI precipitation inference")
    parser.add_argument("--agri_file", default=None)
    parser.add_argument("--agri_dir",  default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--batch_size", type=int, default=32)
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
        log.info("Batch inference on %d files", len(agri_files))
        for f in agri_files:
            try:
                run_inference(f, stats, ckpt, out_d)
            except Exception as exc:
                log.error("Failed for %s: %s", f.name, exc)
    elif args.agri_file:
        run_inference(Path(args.agri_file), stats=stats, checkpoint=ckpt,
                      out_dir=out_d, batch_size=args.batch_size)
    else:
        parser.error("Either --agri_file or --agri_dir required")


if __name__ == "__main__":
    main()
