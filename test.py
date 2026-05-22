"""
test.py
=======
Evaluation of precipitation regression model on the held-out test set.

Metrics reported
----------------
  Detection: POD, FAR, CSI, HSS
  Regression: MAE, RMSE, CC, Bias
  Scatter plot: pred vs true

Outputs saved to cfg.EVAL_OUTPUT_DIR:
  - metrics_summary.csv
  - scatter_pred_vs_true.{svg,pdf,png}
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

import config as cfg
from dataset import NormStats, build_test_dataloader
from model import build_model

log = logging.getLogger(__name__)

# ── Style ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Skill scores
# ─────────────────────────────────────────────────────────────────────────────

def _compute_hss(tp: int, fp: int, fn: int, tn: int) -> float:
    """Heidke Skill Score for binary detection."""
    N = tp + fp + fn + tn
    if N == 0:
        return 0.0
    correct = tp + tn
    expected = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / N
    denom = N - expected
    if denom == 0:
        return 0.0
    return (correct - expected) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _save_pub(fig, base_path):
    base = str(base_path).replace(".png", "")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    log.info("Saved → %s.{svg,pdf,png}", base)


def _plot_scatter(pred: np.ndarray, true: np.ndarray, out_path: Path):
    """Scatter density plot: predicted vs true precipitation."""
    mask = (true > 0.01) | (pred > 0.01)
    p = pred[mask]
    t = true[mask]

    if len(p) < 10:
        log.warning("Not enough non-zero points for scatter plot")
        return

    # Subsample for plotting if too many points
    if len(p) > 50000:
        idx = np.random.choice(len(p), size=50000, replace=False)
        p = p[idx]
        t = t[idx]

    fig, ax = plt.subplots(figsize=(85 / 25.4, 80 / 25.4))

    # 2D histogram / density
    max_val = max(np.percentile(p, 99), np.percentile(t, 99), 10)
    ax.hist2d(t, p, bins=80, range=[[0, max_val], [0, max_val]],
              cmap="Blues", cmin=1)

    # 1:1 line
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.6, alpha=0.5)

    ax.set_xlabel("GPM Precipitation (mm/h)", fontsize=7)
    ax.set_ylabel("Predicted Precipitation (mm/h)", fontsize=7)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Stats annotation
    mask_rain = true > cfg.RAIN_THRESHOLD
    if mask_rain.sum() >= 2:
        r, _ = scipy_stats.pearsonr(pred[mask_rain], true[mask_rain])
        mae = float(np.mean(np.abs(pred[mask_rain] - true[mask_rain])))
        rmse = float(np.sqrt(np.mean((pred[mask_rain] - true[mask_rain]) ** 2)))
    else:
        r, mae, rmse = 0.0, 0.0, 0.0

    ax.text(0.97, 0.25,
            f"CC = {r:.3f}\nMAE = {mae:.2f} mm/h\nRMSE = {rmse:.2f} mm/h",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    _save_pub(fig, out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_predictions(
    stats: NormStats,
    checkpoint: Path,
    test_dl=None,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    log.info("Loaded checkpoint %s", checkpoint)
    model.eval()

    if test_dl is None:
        test_dl = build_test_dataloader(stats)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)

            logits, rain = model(x)
            pred_mmh = (torch.sigmoid(logits) * rain).cpu().numpy().ravel()
            true_mmh = y.cpu().numpy().ravel()

            valid = np.isfinite(true_mmh)
            if valid.any():
                all_preds.append(pred_mmh[valid])
                all_trues.append(true_mmh[valid])

    return {
        "y_pred": np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float32),
        "y_true": np.concatenate(all_trues) if all_trues else np.array([], dtype=np.float32),
    }


def evaluate(stats: NormStats, checkpoint: Optional[Path] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating on %s", device)

    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    try:
        arrays = collect_test_predictions(stats, checkpoint, device=device)
    except FileNotFoundError:
        log.error("Checkpoint not found: %s", checkpoint)
        return

    pred = arrays["y_pred"]
    true = arrays["y_true"]

    if len(true) == 0:
        log.warning("No valid predictions found")
        return

    # ── Detection metrics ──
    threshold = cfg.RAIN_THRESHOLD
    pred_rain = pred > threshold
    true_rain = true > threshold

    tp = int((pred_rain & true_rain).sum())
    fp = int((pred_rain & ~true_rain).sum())
    fn = int((~pred_rain & true_rain).sum())
    tn = int((~pred_rain & ~true_rain).sum())

    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    csi = tp / max(tp + fp + fn, 1)
    hss = _compute_hss(tp, fp, fn, tn)

    # ── Regression metrics ──
    rain_mask = true > threshold
    if rain_mask.sum() >= 2:
        p_rain = pred[rain_mask]
        t_rain = true[rain_mask]
        mae  = float(np.mean(np.abs(p_rain - t_rain)))
        rmse = float(np.sqrt(np.mean((p_rain - t_rain) ** 2)))
        bias = float(np.mean(p_rain - t_rain))
        cc, _ = scipy_stats.pearsonr(p_rain, t_rain)
        cc = max(-1.0, min(1.0, float(cc))) if np.isfinite(cc) else 0.0
    else:
        mae, rmse, bias, cc = 0.0, 0.0, 0.0, 0.0

    # ── Print summary ──
    log.info("─" * 60)
    log.info("Detection (threshold=%.1f mm/h):", threshold)
    log.info("  POD=%.4f  FAR=%.4f  CSI=%.4f  HSS=%.4f", pod, far, csi, hss)
    log.info("  TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)
    log.info("─" * 60)
    log.info("Regression (rain pixels only, n=%d):", int(rain_mask.sum()))
    log.info("  MAE=%.3f mm/h  RMSE=%.3f mm/h  CC=%.3f  Bias=%.3f mm/h",
             mae, rmse, cc, bias)
    log.info("─" * 60)
    log.info("All pixels (n=%d): mean pred=%.3f  mean true=%.3f",
             len(true), float(np.mean(pred)), float(np.mean(true)))
    log.info("─" * 60)

    # ── Save outputs ──
    cfg.EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {"metric": "POD",  "value": pod,  "unit": "score"},
        {"metric": "FAR",  "value": far,  "unit": "score"},
        {"metric": "CSI",  "value": csi,  "unit": "score"},
        {"metric": "HSS",  "value": hss,  "unit": "score"},
        {"metric": "MAE",  "value": mae,  "unit": "mm/h"},
        {"metric": "RMSE", "value": rmse, "unit": "mm/h"},
        {"metric": "CC",   "value": cc,   "unit": "correlation"},
        {"metric": "Bias", "value": bias, "unit": "mm/h"},
        {"metric": "N_rain", "value": int(rain_mask.sum()), "unit": "pixels"},
        {"metric": "N_total", "value": len(true), "unit": "pixels"},
    ]
    pd.DataFrame(rows).to_csv(cfg.EVAL_OUTPUT_DIR / "metrics_summary.csv", index=False)

    _plot_scatter(pred, true, cfg.EVAL_OUTPUT_DIR / "scatter_pred_vs_true")

    log.info("Evaluation complete – results in %s", cfg.EVAL_OUTPUT_DIR)


def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate Precipitation Regression Model")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    stats = NormStats.load(cfg.STATS_FILE)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    evaluate(stats, ckpt)


if __name__ == "__main__":
    main()
