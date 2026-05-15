"""
test.py
=======
Evaluation of CloudPropertyNet on the held-out test set.

Metrics reported
----------------
  CLP : Overall Accuracy (OA), per-class accuracy, confusion matrix
  CTH : RMSE, MAE, Bias, R  (m, cloudy pixels only)

Outputs saved to cfg.EVAL_OUTPUT_DIR:
  - metrics_summary.csv
  - confusion_matrix.{svg,pdf,png}
  - scatter_CTH.{svg,pdf,png}

Usage (called by main.py or standalone):
    python test.py [--checkpoint <path>]
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

import config as cfg
from dataset import NormStats, build_test_dataloader
from model import build_model

log = logging.getLogger(__name__)

PHASE_NAMES = list(getattr(cfg, "CLP_CLASS_NAMES", ["Clear", "Water", "Ice"]))

# ── Nature 风格全局设置 ──
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
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# 色板
C_BLUE   = "#0F4D92"
C_GREEN  = "#2E9E44"
C_RED    = "#E53935"
C_TEAL   = "#42949E"
C_ORANGE = "#E8871D"
C_NEUTRAL = "#767676"
C_LIGHT  = "#CFCECE"


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

def _save_pub(fig, base_path):
    """导出 SVG + PDF + PNG"""
    base = str(base_path).replace(".png", "")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    log.info("Saved → %s.{svg,pdf,png}", base)


def _label_panel(ax, s):
    ax.set_title(s, fontweight="bold", fontsize=8, loc="left", pad=4)


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    """混淆矩阵热力图 — 三分类 CLP"""
    n_cls = cm.shape[0]
    cm_pct = cm / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(90 / 25.4, 75 / 25.4))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max() * 1.1, aspect="auto")

    # 四象限类别标签 (对角=正确, 非对角=错误)
    for i in range(n_cls):
        for j in range(n_cls):
            pct = cm_pct[i, j]
            cnt = cm[i, j]
            color = "white" if pct > cm_pct.max() * 0.5 else "black"
            txt = f"{cnt:,}" if cnt < 1e6 else f"{cnt/1e6:.1f}M"
            ax.text(j, i - 0.15, txt, ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")
            ax.text(j, i + 0.18, f"({pct:.1f}%)", ha="center", va="center",
                    fontsize=5.5, color=color)
            # 角标: correct / misclassified
            tag, tag_c = ("Correct", C_GREEN) if i == j else ("Error", C_RED)
            lbl_c = "white" if pct > cm_pct.max() * 0.5 else tag_c
            ax.text(j, i + 0.42, tag, ha="center", va="center",
                    fontsize=5, color=lbl_c, style="italic")

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(PHASE_NAMES, fontsize=6)
    ax.set_yticklabels(PHASE_NAMES, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=6.5)
    ax.set_ylabel("True", fontsize=6.5)
    ax.tick_params(length=0)

    _label_panel(ax, "a")
    _save_pub(fig, out_path)
    plt.close(fig)


def _plot_scatter(true, pred, label, unit, metrics_dict, out_path: Path):
    """CTH 密度散点图 + 1:1 线 + 指标标注"""
    fig, ax = plt.subplots(figsize=(85 / 25.4, 80 / 25.4))

    hb = ax.hexbin(true, pred, gridsize=70, cmap="YlOrBr", mincnt=1, bins="log")
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    padding = (lim[1] - lim[0]) * 0.05
    lim[0] -= padding
    lim[1] += padding
    ax.plot(lim, lim, "--", color="black", lw=0.6, alpha=0.4, zorder=2)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"MYD06 {label} (m)", fontsize=6.5)
    ax.set_ylabel(f"Predicted {label} (m)", fontsize=6.5)
    ax.set_aspect("equal")

    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)

    # 标注指标
    text = (
        f"RMSE = {metrics_dict['rmse']:.1f} m\n"
        f"MAE  = {metrics_dict['mae']:.1f} m\n"
        f"Bias = {metrics_dict['bias']:+.1f} m\n"
        f"R    = {metrics_dict['r']:.3f}\n"
        f"N    = {metrics_dict['n']:,}"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes,
            fontsize=5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=C_LIGHT, lw=0.5, alpha=0.9))

    _label_panel(ax, "b")
    _save_pub(fig, out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional
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

    # Accumulators
    all_clp_true, all_clp_pred = [], []
    all_clp_for_reg = []
    all_cth_true, all_cth_pred = [], []
    valid_clp_pixels = 0
    total_pixels = 0

    out_std  = torch.from_numpy(stats.out_std[1:]).to(device).reshape(1, 1, 1, 1)
    out_mean = torch.from_numpy(stats.out_mean[1:]).to(device).reshape(1, 1, 1, 1)

    with torch.no_grad():
        for batch_idx, (agri, geo, labels) in enumerate(test_dl):
            agri   = agri.to(device)
            geo    = geo.to(device)
            labels = labels.to(device)

            clp_logits, comp_norm = model(agri, geo=geo)

            # De-normalise
            comp_dn = comp_norm * out_std + out_mean
            lbl_dn  = labels[:, 1:] * out_std + out_mean

            # CLP - evaluate only finite in-range labels, matching train/val masks.
            clp_true_raw = labels[:, 0]
            valid_clp = (
                torch.isfinite(clp_true_raw)
                & (clp_true_raw >= 0)
                & (clp_true_raw < cfg.CLP_CLASSES)
            )
            batch_valid_clp = int(valid_clp.sum().item())
            batch_total = int(clp_true_raw.numel())
            valid_clp_pixels += batch_valid_clp
            total_pixels += batch_total
            if batch_valid_clp == 0:
                log.warning("Batch %d has no valid CLP pixels; skipping CLP metrics for this batch", batch_idx)

            clp_pred_map = clp_logits.argmax(dim=1)
            clp_true_cls = clp_true_raw[valid_clp].long().cpu().numpy().ravel()
            clp_pred_cls = clp_pred_map[valid_clp].cpu().numpy().ravel()

            all_clp_true.append(clp_true_cls)
            all_clp_pred.append(clp_pred_cls)
            all_clp_for_reg.append(clp_true_raw.cpu().numpy().ravel())
            all_cth_true.append(lbl_dn[:, 0].cpu().numpy().ravel())
            all_cth_pred.append(comp_dn[:, 0].cpu().numpy().ravel())

    return {
        "clp_true": np.concatenate(all_clp_true) if all_clp_true else np.array([], dtype=np.int64),
        "clp_pred": np.concatenate(all_clp_pred) if all_clp_pred else np.array([], dtype=np.int64),
        "clp_for_reg": np.concatenate(all_clp_for_reg) if all_clp_for_reg else np.array([], dtype=np.float32),
        "cth_true": np.concatenate(all_cth_true),
        "cth_pred": np.concatenate(all_cth_pred),
        "valid_clp_pixels": valid_clp_pixels,
        "total_pixels": total_pixels,
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

    clp_true = arrays["clp_true"]
    clp_pred = arrays["clp_pred"]
    clp_for_reg = arrays["clp_for_reg"]
    cth_true = arrays["cth_true"]
    cth_pred = arrays["cth_pred"]
    valid_clp_pixels = arrays["valid_clp_pixels"]
    total_pixels = arrays["total_pixels"]

    # Valid masks (cloudy + physically reasonable)
    cloudy = clp_for_reg > 0
    max_cth = getattr(cfg, "MAX_CTH_M", 18000)
    v_cth  = cloudy & (cth_true >= 0) & (cth_true <= max_cth) & np.isfinite(cth_true)

    # ── CLP metrics ───────────────────────────────────────────────────────
    valid_ratio = float(valid_clp_pixels / total_pixels) if total_pixels > 0 else np.nan
    if valid_clp_pixels == 0:
        log.warning("No valid CLP pixels found in the test set; CLP metrics are undefined")
        oa = np.nan
    else:
        oa = float((clp_true == clp_pred).mean() * 100)
    cm = confusion_matrix(clp_true, clp_pred, labels=list(range(cfg.CLP_CLASSES)))
    class_support = cm.sum(axis=1)
    per_class_acc = np.full(cfg.CLP_CLASSES, np.nan, dtype=float)
    valid_classes = class_support > 0
    per_class_acc[valid_classes] = cm.diagonal()[valid_classes] / class_support[valid_classes] * 100
    macro_acc = float(per_class_acc[valid_classes].mean()) if valid_classes.any() else np.nan

    # ── CTH regression metrics ────────────────────────────────────────────
    cth_m = _stats(cth_true, cth_pred, v_cth)

    # ── Print summary ─────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info(
        "CLP valid pixels: valid_clp_pixels=%d total_pixels=%d valid_ratio=%.6f",
        valid_clp_pixels, total_pixels, valid_ratio,
    )
    log.info("Cloud Phase  – OA = %.2f%%", oa)
    log.info("Cloud Phase  – macro acc/recall = %.2f%%", macro_acc)
    for i, name in enumerate(PHASE_NAMES):
        log.info("  %-12s acc/recall = %.2f%%", name, per_class_acc[i])
    log.info(
        "CTH (n=%7d)  RMSE=%.3f m  MAE=%.3f  Bias=%.3f  R=%.4f",
        cth_m["n"], cth_m["rmse"], cth_m["mae"], cth_m["bias"], cth_m["r"]
    )
    log.info("─" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────
    cfg.EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {"variable": "valid_clp_pixels", "value": valid_clp_pixels, "unit": "pixels"},
        {"variable": "total_pixels", "value": total_pixels, "unit": "pixels"},
        {"variable": "valid_ratio", "value": valid_ratio, "unit": "ratio"},
        {"variable": "CLP_OA", "value": oa, "unit": "%"},
        {"variable": "CLP_macro_acc", "value": macro_acc, "unit": "%"},
        {"variable": "CLP_macro_recall", "value": macro_acc, "unit": "%"},
    ]
    for i, name in enumerate(PHASE_NAMES):
        rows.append({"variable": f"CLP_{name}_acc", "value": per_class_acc[i], "unit": "%"})
        rows.append({"variable": f"CLP_{name}_recall", "value": per_class_acc[i], "unit": "%"})
    for k, v in cth_m.items():
        rows.append({"variable": f"CTH_{k}", "value": v, "unit": "m"})
    pd.DataFrame(rows).to_csv(cfg.EVAL_OUTPUT_DIR / "metrics_summary.csv", index=False)

    _plot_confusion_matrix(cm, cfg.EVAL_OUTPUT_DIR / "confusion_matrix")

    if v_cth.sum() > 100:
        _plot_scatter(cth_true[v_cth], cth_pred[v_cth], "CTH", "m", cth_m,
                      cfg.EVAL_OUTPUT_DIR / "scatter_CTH")

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
