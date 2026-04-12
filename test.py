"""
test.py
=======
Evaluation of CloudPropertyNet on the held-out test set.

Metrics reported
----------------
  CLP : Overall Accuracy (OA), per-class accuracy, confusion matrix
  CER : RMSE, MAE, Bias, R  (µm, cloudy pixels only)
  COT : same
  CTH : same (m)

Outputs saved to cfg.EVAL_OUTPUT_DIR:
  - metrics_summary.csv
  - confusion_matrix.png
  - scatter_{CER,COT,CTH}.png

Usage (called by main.py or standalone):
    python test.py [--checkpoint <path>]
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

import config as cfg
from dataset import NormStats, build_dataloaders
from model import build_model

log = logging.getLogger(__name__)

PHASE_NAMES = ["Clear", "Water", "Supercool", "Mixed", "Ice"]


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(true: np.ndarray, pred: np.ndarray, valid: np.ndarray):
    t, p = true[valid], pred[valid]
    if t.size == 0:
        return dict(rmse=np.nan, mae=np.nan, bias=np.nan, r=np.nan, n=0)
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae  = float(np.mean(np.abs(t - p)))
    bias = float(np.mean(p - t))
    if t.std() > 0 and p.std() > 0:
        r = float(np.corrcoef(t, p)[0, 1])
    else:
        r = 0.0
    return dict(rmse=rmse, mae=mae, bias=bias, r=r, n=int(t.size))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cfg.CLP_CLASSES))
    ax.set_yticks(range(cfg.CLP_CLASSES))
    ax.set_xticklabels(PHASE_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(PHASE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cloud Phase Confusion Matrix")
    for i in range(cfg.CLP_CLASSES):
        for j in range(cfg.CLP_CLASSES):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max() * 0.5 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", out_path)


def _plot_scatter(true, pred, label, unit, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(true, pred, gridsize=60, cmap="plasma", mincnt=1)
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lim, lim, "w--", lw=1)
    ax.set_xlabel(f"MYD06 {label} ({unit})")
    ax.set_ylabel(f"Predicted {label} ({unit})")
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved scatter → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional
def evaluate(stats: NormStats, checkpoint: Optional[Path] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating on %s", device)

    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    model = build_model().to(device)
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        log.info("Loaded checkpoint %s", checkpoint)
    except FileNotFoundError:
        log.error("Checkpoint not found: %s", checkpoint)
        return

    model.eval()

    _, _, test_dl = build_dataloaders(stats)

    # Accumulators
    all_clp_true, all_clp_pred = [], []
    all_cer_true, all_cer_pred = [], []
    all_cot_true, all_cot_pred = [], []
    all_cth_true, all_cth_pred = [], []

    out_std  = torch.from_numpy(stats.out_std[1:]).to(device).reshape(1, 3, 1, 1)
    out_mean = torch.from_numpy(stats.out_mean[1:]).to(device).reshape(1, 3, 1, 1)

    with torch.no_grad():
        for agri, _geo, labels in test_dl:
            agri   = agri.to(device)
            labels = labels.to(device)

            clp_logits, comp_norm = model(agri)

            # De-normalise
            comp_dn = comp_norm * out_std + out_mean
            lbl_dn  = labels[:, 1:] * out_std + out_mean

            # CLP
            clp_pred_cls = clp_logits.argmax(dim=1).cpu().numpy().ravel()
            clp_true_cls = labels[:, 0].long().cpu().numpy().ravel()

            # Regression – only cloudy pixels (CLP > 0)
            clp_mask = clp_true_cls > 0

            all_clp_true.append(clp_true_cls)
            all_clp_pred.append(clp_pred_cls)
            all_cer_true.append(lbl_dn[:, 0].cpu().numpy().ravel())
            all_cer_pred.append(comp_dn[:, 0].cpu().numpy().ravel())
            all_cot_true.append(lbl_dn[:, 1].cpu().numpy().ravel())
            all_cot_pred.append(comp_dn[:, 1].cpu().numpy().ravel())
            all_cth_true.append(lbl_dn[:, 2].cpu().numpy().ravel())
            all_cth_pred.append(comp_dn[:, 2].cpu().numpy().ravel())

    clp_true = np.concatenate(all_clp_true)
    clp_pred = np.concatenate(all_clp_pred)
    cer_true = np.concatenate(all_cer_true)
    cer_pred = np.concatenate(all_cer_pred)
    cot_true = np.concatenate(all_cot_true)
    cot_pred = np.concatenate(all_cot_pred)
    cth_true = np.concatenate(all_cth_true)
    cth_pred = np.concatenate(all_cth_pred)

    # Valid masks (cloudy + physically reasonable)
    cloudy = clp_true > 0
    v_cer  = cloudy & (cer_true >= 0) & (cer_true <= 100) & np.isfinite(cer_true)
    v_cot  = cloudy & (cot_true >= 0) & (cot_true <= 200) & np.isfinite(cot_true)
    v_cth  = cloudy & (cth_true >= 0) & (cth_true <= 25000) & np.isfinite(cth_true)

    # ── CLP metrics ───────────────────────────────────────────────────────
    oa = float((clp_true == clp_pred).mean() * 100)
    cm = confusion_matrix(clp_true, clp_pred, labels=list(range(cfg.CLP_CLASSES)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1) * 100

    # ── Regression metrics ────────────────────────────────────────────────
    cer_m = _stats(cer_true, cer_pred, v_cer)
    cot_m = _stats(cot_true, cot_pred, v_cot)
    cth_m = _stats(cth_true, cth_pred, v_cth)

    # ── Print summary ─────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Cloud Phase  – OA = %.2f%%", oa)
    for i, name in enumerate(PHASE_NAMES):
        log.info("  %-12s acc = %.2f%%", name, per_class_acc[i])
    for var, m, u in [("CER", cer_m, "µm"), ("COT", cot_m, ""), ("CTH", cth_m, "m")]:
        log.info(
            "%-4s (n=%7d)  RMSE=%.3f %s  MAE=%.3f  Bias=%.3f  R=%.4f",
            var, m["n"], m["rmse"], u, m["mae"], m["bias"], m["r"]
        )
    log.info("─" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────
    cfg.EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [{"variable": "CLP_OA", "value": oa, "unit": "%"}]
    for i, name in enumerate(PHASE_NAMES):
        rows.append({"variable": f"CLP_{name}_acc", "value": per_class_acc[i], "unit": "%"})
    for var, m, u in [("CER", cer_m, "um"), ("COT", cot_m, ""), ("CTH", cth_m, "m")]:
        for k, v in m.items():
            rows.append({"variable": f"{var}_{k}", "value": v, "unit": u})
    pd.DataFrame(rows).to_csv(cfg.EVAL_OUTPUT_DIR / "metrics_summary.csv", index=False)

    _plot_confusion_matrix(cm, cfg.EVAL_OUTPUT_DIR / "confusion_matrix.png")

    for true, pred, valid, label, unit in [
        (cer_true, cer_pred, v_cer, "CER", "µm"),
        (cot_true, cot_pred, v_cot, "COT", ""),
        (cth_true, cth_pred, v_cth, "CTH", "m"),
    ]:
        if valid.sum() > 100:
            _plot_scatter(true[valid], pred[valid], label, unit,
                          cfg.EVAL_OUTPUT_DIR / f"scatter_{label}.png")

    log.info("Evaluation complete – results in %s", cfg.EVAL_OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate CloudPropertyNet")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    stats = NormStats.load(cfg.STATS_FILE)
    ckpt  = Path(args.checkpoint) if args.checkpoint else None
    evaluate(stats, ckpt)


if __name__ == "__main__":
    main()
