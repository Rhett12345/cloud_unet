"""
main.py
========
Pipeline orchestrator for AGRI → GPM precipitation classification.

Stages
------
  fuse      -> data_fusion.py  : GPM+AGRI data pairing
  stats     -> dataset.py      : compute normalisation statistics
  train     -> train.py        : train the model
  test      -> test.py         : evaluate on held-out test set
  infer     -> inference.py    : full-disk inference for new AGRI files

Usage examples
--------------
  python main.py --stages fuse stats train test
  python main.py --stages fuse --split train --day 20190101 --workers 8
  python main.py --stages train
  python main.py --stages infer --agri_file /path/to/FY4A_AGRI_*.HDF
"""

import argparse
import logging
import sys
from pathlib import Path

import config as cfg

log = logging.getLogger(__name__)


def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.LOG_DIR / "pipeline.log"),
        ]
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_fuse(args):
    """GPM+AGRI 数据配对。"""
    from data_fusion import find_day_folders, fuse_day

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

    split = getattr(args, "split", "train")
    dates = [args.day] if getattr(args, "day", None) else split_dates[split]
    n_workers = getattr(args, "workers", 1)
    max_dt_min = getattr(args, "max_dt_min", None)

    agri_days = find_day_folders(cfg.AGRI_ROOT, dates)

    for agri_day in agri_days:
        out_sub = split_out[split] / agri_day.name
        fuse_day(
            agri_day_dir=agri_day,
            out_dir=out_sub,
            mode=split,
            overwrite=getattr(args, "overwrite", False),
            n_workers=n_workers,
            max_dt_min=max_dt_min,
        )


def stage_stats(args):
    from dataset import compute_and_save_stats
    log.info("Computing normalisation statistics from training split...")
    stats = compute_and_save_stats(cfg.PAIRED_TRAIN_DIR, out_path=cfg.STATS_FILE)
    log.info("Stats saved to %s", cfg.STATS_FILE)
    return stats


def stage_train(args):
    from dataset import NormStats
    from train import train
    if not cfg.STATS_FILE.exists():
        log.error("Stats file not found: %s - run --stages stats first", cfg.STATS_FILE)
        sys.exit(1)
    stats = NormStats.load(cfg.STATS_FILE)
    resume = getattr(args, "resume", None)
    if resume:
        log.info("Resuming from checkpoint: %s", resume)
    else:
        log.info("Starting training from scratch...")
    train(stats, resume_checkpoint=resume)


def stage_test(args):
    from dataset import NormStats
    from test import evaluate
    if not cfg.STATS_FILE.exists():
        log.error("Stats file not found: %s", cfg.STATS_FILE)
        sys.exit(1)
    stats = NormStats.load(cfg.STATS_FILE)
    ckpt = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    evaluate(stats, ckpt)


def stage_infer(args):
    from dataset import NormStats
    from inference import run_inference
    if not cfg.STATS_FILE.exists():
        log.error("Stats file not found: %s", cfg.STATS_FILE)
        sys.exit(1)
    stats = NormStats.load(cfg.STATS_FILE)
    ckpt = Path(args.checkpoint) if getattr(args, "checkpoint", None) else None
    out_d = Path(args.out_dir) if getattr(args, "out_dir", None) else cfg.RETRIEVAL_DIR

    if getattr(args, "agri_file", None):
        run_inference(Path(args.agri_file), stats, ckpt, out_d)
    elif getattr(args, "agri_dir", None):
        agri_dir = Path(args.agri_dir)
        agri_files = (sorted(agri_dir.rglob("*.HDF")) + sorted(agri_dir.rglob("*.hdf"))
                      + sorted(agri_dir.rglob("*.npz")))
        log.info("Batch inference on %d files", len(agri_files))
        for f in agri_files:
            try:
                run_inference(f, stats, ckpt, out_d)
            except Exception as exc:
                log.error("Failed for %s: %s", f.name, exc)
    else:
        log.error("Provide --agri_file or --agri_dir for inference stage")
        sys.exit(1)


STAGE_FN = {
    "fuse":  stage_fuse,
    "stats": stage_stats,
    "train": stage_train,
    "test":  stage_test,
    "infer": stage_infer,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="AGRI → GPM precipitation classification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stages",     nargs="+", choices=list(STAGE_FN.keys()), required=True)
    p.add_argument("--split",      default="train", choices=["train", "val", "test"])
    p.add_argument("--day",        default=None)
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--workers",    type=int, default=None)
    p.add_argument("--max-dt-min", type=float, default=None)
    p.add_argument("--resume",     default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--agri_file",  default=None)
    p.add_argument("--agri_dir",   default=None)
    p.add_argument("--out_dir",    default=None)
    return p.parse_args()


def main():
    _setup_logging()
    args = parse_args()

    if args.workers is None:
        try:
            import fusion_config as fc
            args.workers = fc.N_FUSION_WORKERS
        except ImportError:
            args.workers = 1

    log.info("=" * 60)
    log.info("Pipeline stages: %s", " -> ".join(args.stages))
    log.info("=" * 60)

    for stage in args.stages:
        log.info("Stage: %s", stage.upper())
        STAGE_FN[stage](args)
        log.info("Stage %s complete", stage.upper())

    log.info("All stages finished.")


if __name__ == "__main__":
    main()
