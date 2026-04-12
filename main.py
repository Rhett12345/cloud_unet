"""
main.py
=======
Pipeline orchestrator – single entry point for all stages.

Stages
------
  fuse      → data_fusion.py   : match AGRI + MYD06, write paired HDF5
  stats     → dataset.py       : compute normalisation statistics
  train     → train.py         : train the model
  test      → test.py          : evaluate on held-out test set
  infer     → inference.py     : full-disk retrieval for new AGRI files

Usage examples
--------------
  # Full pipeline from scratch
  python main.py --stages fuse stats train test

  # Only training (data already fused and stats computed)
  python main.py --stages train

  # Inference on a new file
  python main.py --stages infer --agri_file /path/to/FY4B_AGRI_20230615_0600.HDF

  # Fuse only the test split for a specific day
  python main.py --stages fuse --split test --day 20230710

Flags
-----
  --stages       One or more of: fuse stats train test infer  (in order)
  --split        Data split for 'fuse': train | val | test   (default: train)
  --day          Single day YYYYMMDD for 'fuse' (default: all configured days)
  --overwrite    Re-process existing paired files during 'fuse'
  --checkpoint   Custom .pth path for 'test' or 'infer'
  --agri_file    Path to a single AGRI file for 'infer'
  --agri_dir     Directory of AGRI files for batch 'infer'
  --out_dir      Custom output directory for 'infer'
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import config as cfg

log = logging.getLogger(__name__)


def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.LOG_DIR / "pipeline.log"),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────

def stage_fuse(args):
    from data_fusion import _find_day_folders, fuse_day

    splits = [args.split] if hasattr(args, "split") else ["train", "val", "test"]
    split_out = {
        "train": cfg.PAIRED_TRAIN_DIR,
        "val":   cfg.PAIRED_VAL_DIR,
        "test":  cfg.PAIRED_TEST_DIR,
    }
    split_dates = {
        "train": cfg.TRAIN_DATES,
        "val":   cfg.VAL_DATES,
        "test":  cfg.TEST_DATES,
    }

    for split in splits:
        dates = [args.day] if getattr(args, "day", None) else split_dates[split]
        agri_days  = _find_day_folders(cfg.AGRI_ROOT, dates)
        modis_days = {d.name: d for d in _find_day_folders(cfg.MODIS_ROOT, dates)}

        for agri_day in agri_days:
            modis_day = modis_days.get(agri_day.name)
            if modis_day is None:
                log.warning("No MODIS folder for %s – skipping", agri_day.name)
                continue
            out_sub = split_out[split] / agri_day.name
            fuse_day(agri_day, modis_day, out_sub, overwrite=getattr(args, "overwrite", False))


def stage_stats(args):
    from dataset import compute_and_save_stats
    log.info("Computing normalisation statistics from training split…")
    stats = compute_and_save_stats(cfg.PAIRED_TRAIN_DIR, out_path=cfg.STATS_FILE)
    log.info("Stats saved to %s", cfg.STATS_FILE)
    return stats


def stage_train(args):
    from dataset import NormStats
    from train import train

    if not cfg.STATS_FILE.exists():
        log.error("Stats file not found: %s – run --stages stats first", cfg.STATS_FILE)
        sys.exit(1)

    stats = NormStats.load(cfg.STATS_FILE)
    log.info("Starting training…")
    train(stats)


def stage_test(args):
    from dataset import NormStats
    from test import evaluate

    if not cfg.STATS_FILE.exists():
        log.error("Stats file not found: %s", cfg.STATS_FILE)
        sys.exit(1)

    stats  = NormStats.load(cfg.STATS_FILE)
    ckpt   = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    evaluate(stats, ckpt)


def stage_infer(args):
    from dataset import NormStats
    from inference import run_inference

    if not cfg.STATS_FILE.exists():
        log.error("Stats file not found: %s", cfg.STATS_FILE)
        sys.exit(1)

    stats  = NormStats.load(cfg.STATS_FILE)
    ckpt   = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    out_d  = Path(args.out_dir)    if getattr(args, "out_dir", None)    else cfg.RETRIEVAL_DIR

    # Single file
    if getattr(args, "agri_file", None):
        run_inference(Path(args.agri_file), stats, ckpt, out_d)

    # Directory of files
    elif getattr(args, "agri_dir", None):
        agri_dir = Path(args.agri_dir)
        agri_files = sorted(agri_dir.rglob("*.HDF")) + sorted(agri_dir.rglob("*.hdf"))
        log.info("Batch inference on %d files in %s", len(agri_files), agri_dir)
        for f in agri_files:
            try:
                run_inference(f, stats, ckpt, out_d)
            except Exception as exc:
                log.error("Failed for %s: %s", f.name, exc)
    else:
        log.error("Provide --agri_file or --agri_dir for inference stage")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

STAGE_FN = {
    "fuse":  stage_fuse,
    "stats": stage_stats,
    "train": stage_train,
    "test":  stage_test,
    "infer": stage_infer,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="AGRI + MYD06 cloud retrieval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--stages",     nargs="+", choices=list(STAGE_FN.keys()), required=True,
                   help="Pipeline stage(s) to run (in order)")
    p.add_argument("--split",      default="train", choices=["train", "val", "test"],
                   help="Which data split to fuse (default: train)")
    p.add_argument("--day",        default=None,
                   help="Single day YYYYMMDD for fuse stage (default: all days)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite existing paired files during fuse")
    p.add_argument("--checkpoint", default=None,
                   help="Custom model checkpoint for test/infer")
    p.add_argument("--agri_file",  default=None,
                   help="Single AGRI file for inference")
    p.add_argument("--agri_dir",   default=None,
                   help="Directory of AGRI files for batch inference")
    p.add_argument("--out_dir",    default=None,
                   help="Output directory for inference results")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _setup_logging()
    args = parse_args()

    log.info("=" * 60)
    log.info("Pipeline stages: %s", " → ".join(args.stages))
    log.info("=" * 60)

    for stage in args.stages:
        log.info("▶  Stage: %s", stage.upper())
        STAGE_FN[stage](args)
        log.info("✓  Stage %s complete", stage.upper())

    log.info("All stages finished.")


if __name__ == "__main__":
    main()
