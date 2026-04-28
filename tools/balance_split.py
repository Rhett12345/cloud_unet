#!/usr/bin/env python3
"""
balance_split.py
================
Scan fused H5 files, compute per-date CLP class distributions, and report or
suggest stratified train/val/test date splits with balanced class proportions.

Usage:
  # Report current split balance
  python tools/balance_split.py --report

  # Suggest a stratified 24/6/6 split from all available dates
  python tools/balance_split.py --suggest --n-train 24 --n-val 6 --n-test 6

  # Suggest with a specific random seed
  python tools/balance_split.py --suggest --seed 42
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

import config as cfg


def _scan_date_distribution(paired_dirs):
    """Scan all H5 files and return {date: {clear, water, ice, total}}."""
    date_stats = {}
    for paired_dir in paired_dirs:
        for h5 in sorted(Path(paired_dir).rglob("*.h5")):
            date_str = None
            for part in h5.parts:
                if len(part) == 8 and part.isdigit():
                    date_str = part
                    break
            if date_str is None:
                continue
            try:
                with h5py.File(h5, "r") as f:
                    if "Samples" in f and "labels" in f["Samples"]:
                        labels_all = f["Samples/labels"][()]
                        clp = labels_all[:, 0]
                    elif "Labels" in f and "CLP" in f["Labels"]:
                        clp = f["Labels/CLP"][()]
                    else:
                        continue
                valid = np.isfinite(clp)
                if valid.sum() == 0:
                    continue
                counts = {int(c): int((clp[valid] == c).sum()) for c in range(cfg.CLP_CLASSES)}
                if date_str not in date_stats:
                    date_stats[date_str] = {c: 0 for c in range(cfg.CLP_CLASSES)}
                    date_stats[date_str]["total"] = 0
                for c in range(cfg.CLP_CLASSES):
                    date_stats[date_str][c] += counts.get(c, 0)
                    date_stats[date_str]["total"] += counts.get(c, 0)
            except Exception as exc:
                print(f"  Skip {h5.name}: {exc}", file=sys.stderr)
    return date_stats


def _distribution_vector(date_stats, date_str):
    s = date_stats[date_str]
    total = max(s["total"], 1)
    return np.array([s[c] / total for c in range(cfg.CLP_CLASSES)], dtype=np.float64)


def _print_split_report(date_stats, train_dates, val_dates, test_dates, label=""):
    def _summarize(dates):
        total = np.zeros(cfg.CLP_CLASSES, dtype=np.int64)
        for d in dates:
            if d in date_stats:
                for c in range(cfg.CLP_CLASSES):
                    total[c] += date_stats[d].get(c, 0)
        s = total.sum()
        if s == 0:
            return 0, np.zeros(cfg.CLP_CLASSES)
        return s, total / s * 100

    names = getattr(cfg, "CLP_CLASS_NAMES", ["Clear", "Water", "Ice"])
    for split_name, dates in [("train", train_dates), ("val", val_dates), ("test", test_dates)]:
        n, fracs = _summarize(dates)
        parts = " | ".join(f"{names[c]}={fracs[c]:.1f}%" for c in range(cfg.CLP_CLASSES))
        print(f"  {split_name:6s}  n={n:>10,d}  {parts}  [{len(dates)} days]")
    if label:
        print(label)


def cmd_report():
    date_stats = _scan_date_distribution([cfg.PAIRED_TRAIN_DIR, cfg.PAIRED_VAL_DIR, cfg.PAIRED_TEST_DIR])
    if not date_stats:
        print("No H5 files found. Run fusion first.", file=sys.stderr)
        sys.exit(1)
    print(f"\nFound {len(date_stats)} dates with fused data.\n")
    _print_split_report(date_stats, cfg.TRAIN_DATES, cfg.VAL_DATES, cfg.TEST_DATES,
                        label="\nCurrent config.py split above.")


def _greedy_balanced_split(date_stats, n_train, n_val, n_test, seed):
    """Assign dates to splits greedily, balancing per-class proportions."""
    rng = np.random.RandomState(seed)
    all_dates = sorted(date_stats.keys())
    rng.shuffle(all_dates)

    target = np.zeros(cfg.CLP_CLASSES, dtype=np.float64)
    for d in all_dates:
        for c in range(cfg.CLP_CLASSES):
            target[c] += date_stats[d].get(c, 0)
    target = target / target.sum()

    splits = {"train": [], "val": [], "test": []}
    counts = {k: np.zeros(cfg.CLP_CLASSES, dtype=np.float64) for k in splits}
    quotas = {"train": n_train, "val": n_val, "test": n_test}

    for d in all_dates:
        vec = _distribution_vector(date_stats, d)
        candidates = [k for k in splits if len(splits[k]) < quotas[k]]
        if not candidates:
            continue
        best_split = min(candidates, key=lambda k: _divergence(
            (counts[k] + date_stats[d].get(c, 0) for c in range(cfg.CLP_CLASSES)), target))
        splits[best_split].append(d)
        for c in range(cfg.CLP_CLASSES):
            counts[best_split][c] += date_stats[d].get(c, 0)

    return splits["train"], splits["val"], splits["test"]


def _divergence(counts_tuple, target):
    counts = np.array(counts_tuple, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 1e9
    return float(np.sum(np.abs(counts / total - target)))


def cmd_suggest(args):
    date_stats = _scan_date_distribution([cfg.PAIRED_TRAIN_DIR, cfg.PAIRED_VAL_DIR, cfg.PAIRED_TEST_DIR])
    if not date_stats:
        print("No H5 files found. Run fusion first.", file=sys.stderr)
        sys.exit(1)

    n_train = getattr(args, "n_train", 24)
    n_val = getattr(args, "n_val", 6)
    n_test = getattr(args, "n_test", 6)
    seed = getattr(args, "seed", 42)

    train_d, val_d, test_d = _greedy_balanced_split(date_stats, n_train, n_val, n_test, seed)
    print(f"\nSuggested stratified split (seed={seed}):\n")
    print(f"TRAIN_DATES = {json.dumps(sorted(train_d))}")
    print(f"VAL_DATES   = {json.dumps(sorted(val_d))}")
    print(f"TEST_DATES  = {json.dumps(sorted(test_d))}")
    print()
    _print_split_report(date_stats, train_d, val_d, test_d,
                        label="\nCopy the lists above into config.py.")


def main():
    parser = argparse.ArgumentParser(description="Balance date split by CLP class distribution")
    parser.add_argument("--report", action="store_true", help="Report current split balance")
    parser.add_argument("--suggest", action="store_true", help="Suggest a stratified split")
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.report:
        cmd_report()
    elif args.suggest:
        cmd_suggest(args)
    else:
        cmd_report()
        print("\nUse --suggest to generate a stratified split.")


if __name__ == "__main__":
    main()
