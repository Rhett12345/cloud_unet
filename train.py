"""
train.py
========
Training loop for CloudPropertyNet (AGRI-only).

Features
--------
- Mixed-precision (AMP) training via torch.cuda.amp
- Gradient clipping
- ReduceLROnPlateau scheduler
- Saves best checkpoint (val loss) + last checkpoint each epoch
- Writes per-epoch CSV log for easy plotting

Usage (called by main.py or standalone):
    python train.py
"""

import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast

import config as cfg
from dataset import NormStats, build_dataloaders
from model import build_model

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int = cfg.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_losses(clp_logits, comp_out, labels, ce_loss_fn, reg_loss_fn):
    """
    labels: (B, 2, H, W)
    ch0 = CLP
    ch1 = CTH (normalised)
    NaN means invalid supervision
    """
    zero = clp_logits.sum() * 0.0

    # ---- CLP masked CE ----
    clp_raw = labels[:, 0, :, :]
    valid_clp = torch.isfinite(clp_raw) & (clp_raw >= 0) & (clp_raw < cfg.CLP_CLASSES)

    if valid_clp.any():
        clp_target = torch.where(
            valid_clp,
            clp_raw,
            torch.full_like(clp_raw, -100)
        ).long()
        loss_clp = ce_loss_fn(clp_logits, clp_target) * cfg.LOSS_W_CLP
    else:
        loss_clp = zero

    # ---- CTH masked regression loss ----
    def masked_reg_loss(pred, target, weight):
        valid = torch.isfinite(target)
        if valid.any():
            return reg_loss_fn(pred[valid], target[valid]) * weight
        return zero

    loss_cth = masked_reg_loss(comp_out[:, 0], labels[:, 1], cfg.LOSS_W_CTH)

    total = loss_clp + loss_cth
    return total, loss_clp, loss_cth

@torch.no_grad()
def _batch_metrics(clp_logits, comp_out, labels, stats: NormStats, device):
    clp_pred = clp_logits.argmax(dim=1)
    clp_true = labels[:, 0]

    valid_clp = torch.isfinite(clp_true) & (clp_true >= 0) & (clp_true < cfg.CLP_CLASSES)
    if valid_clp.any():
        oa = (clp_pred[valid_clp] == clp_true[valid_clp].long()).float().mean().item() * 100.0
        per_class_acc = []
        for c in range(cfg.CLP_CLASSES):
            c_mask = valid_clp & (clp_true == c)
            if c_mask.any():
                acc = (clp_pred[c_mask] == c).float().mean().item() * 100.0
            else:
                acc = -1.0
            per_class_acc.append(acc)
    else:
        oa = 0.0
        per_class_acc = [-1.0] * cfg.CLP_CLASSES

    out_std  = torch.from_numpy(stats.out_std[1:]).to(device).reshape(1, 1, 1, 1)
    out_mean = torch.from_numpy(stats.out_mean[1:]).to(device).reshape(1, 1, 1, 1)

    pred_dn = comp_out * out_std + out_mean
    true_dn = labels[:, 1:] * out_std + out_mean

    def rmse_masked(a, b):
        valid = torch.isfinite(b)
        if valid.any():
            return torch.sqrt(torch.mean((a[valid] - b[valid]) ** 2)).item()
        return 0.0

    result = {
        "oa": oa,
        "cth_rmse": rmse_masked(pred_dn[:, 0], true_dn[:, 0]),
    }
    for c in range(cfg.CLP_CLASSES):
        result[f"cls{c}_acc"] = per_class_acc[c]
    valid_acc = [acc for acc in per_class_acc if acc >= 0.0]
    result["macro_acc"] = float(np.mean(valid_acc)) if valid_acc else 0.0
    return result


def _macro_acc(metrics):
    values = [
        float(metrics[f"cls{c}_acc"])
        for c in range(cfg.CLP_CLASSES)
        if f"cls{c}_acc" in metrics and float(metrics[f"cls{c}_acc"]) >= 0.0
    ]
    return float(np.mean(values)) if values else 0.0


def _metric_value(metrics, monitor: str):
    monitor = (monitor or "val_loss").lower()
    if monitor in {"val_loss", "loss"}:
        return float(metrics["loss"])
    if monitor in {"val_oa", "oa"}:
        return float(metrics["oa"])
    if monitor in {"val_macro_acc", "macro_acc", "balanced_acc"}:
        return float(metrics.get("macro_acc", _macro_acc(metrics)))
    raise ValueError(f"Unsupported checkpoint monitor: {monitor}")


def _monitor_mode(monitor: str) -> str:
    return "min" if (monitor or "").lower() in {"val_loss", "loss"} else "max"


def _is_better(candidate, current, monitor: str) -> bool:
    if current is None:
        return True
    cand_v = _metric_value(candidate, monitor)
    curr_v = _metric_value(current, monitor)
    if _monitor_mode(monitor) == "min":
        return cand_v < curr_v
    return cand_v > curr_v

# ─────────────────────────────────────────────────────────────────────────────
# Epoch runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, ce_fn, reg_fn, stats, device, optimizer=None, scaler=None):
    """
    Shared forward-pass loop for both train and validation.
    Pass optimizer=None for validation mode.
    """
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "clp": 0.0, "cth": 0.0,
              "oa": 0.0, "macro_acc": 0.0,
              "cth_rmse": 0.0, "n": 0}
    for c in range(cfg.CLP_CLASSES):
        totals[f"cls{c}_acc"] = 0.0

    for agri, geo, labels in loader:
        agri   = agri.to(device)
        geo    = geo.to(device)
        labels = labels.to(device)

        with autocast(enabled=(scaler is not None)):
            clp_logits, comp_out = model(agri, geo=geo)
            total, l_clp, l_cth = _compute_losses(
                clp_logits, comp_out, labels, ce_fn, reg_fn
            )

        if training:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

        m = _batch_metrics(clp_logits, comp_out, labels, stats, device)

        B = agri.shape[0]
        totals["loss"]     += total.item()  * B
        totals["clp"]      += l_clp.item()  * B
        totals["cth"]      += l_cth.item()  * B
        totals["oa"]       += m["oa"]       * B
        totals["macro_acc"] += m["macro_acc"] * B
        totals["cth_rmse"] += m["cth_rmse"] * B
        for c in range(cfg.CLP_CLASSES):
            totals[f"cls{c}_acc"] += max(0.0, m[f"cls{c}_acc"]) * B
        totals["n"]        += B

    N = max(totals["n"], 1)
    result = {k: v / N for k, v in totals.items() if k != "n"}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(stats: NormStats):
    _seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    train_dl, val_dl, _ = build_dataloaders(stats)

    log.info("train patches = %d", len(train_dl.dataset))
    log.info("val patches   = %d", len(val_dl.dataset))
    log.info("train iters/epoch = %d", len(train_dl))
    log.info("val iters/epoch   = %d", len(val_dl))

    model     = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR
    )
    scaler = GradScaler() if torch.cuda.is_available() else None

    log.info("Computing CLP class distribution from training set ...")
    class_counts = torch.zeros(cfg.CLP_CLASSES, dtype=torch.float32)
    for _, _, labels in train_dl:
        clp = labels[:, 0]
        valid = torch.isfinite(clp) & (clp >= 0) & (clp < cfg.CLP_CLASSES)
        for c in range(cfg.CLP_CLASSES):
            class_counts[c] += (valid & (clp == c)).sum().item()
    total = class_counts.sum()
    pcts = [(class_counts[c] / total * 100).item() for c in range(cfg.CLP_CLASSES)]
    log.info("CLP class distribution: %s",
             " | ".join(f"c{c}={pcts[c]:.1f}%" for c in range(cfg.CLP_CLASSES)))

    class_weights = torch.ones(cfg.CLP_CLASSES, dtype=torch.float32)
    valid_classes = class_counts > 0
    if valid_classes.any():
        freq = class_counts[valid_classes] / class_counts[valid_classes].sum().clamp_min(1.0)
        raw = 1.0 / freq.clamp_min(1e-6)
        class_weights[valid_classes] = raw.clamp_max(10.0)
    log.info("CLP loss weights: %s",
             " | ".join(f"c{c}={class_weights[c].item():.2f}" for c in range(cfg.CLP_CLASSES)))

    ce_fn  = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights.to(device))
    reg_fn = nn.SmoothL1Loss()

    monitor = getattr(cfg, "CHECKPOINT_MONITOR", "val_loss")
    best_selected = None
    best_loss = None
    best_oa = None
    best_macro = None
    epochs_no_best = 0
    log_rows      = []

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = _run_epoch(model, train_dl, ce_fn, reg_fn, stats, device,
                             optimizer=optimizer, scaler=scaler)
        val_m   = _run_epoch(model, val_dl,   ce_fn, reg_fn, stats, device)

        scheduler.step(val_m["macro_acc"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        cls_str = " ".join(
            f"c{c}={val_m.get(f'cls{c}_acc', -1):.1f}"
            for c in range(cfg.CLP_CLASSES) if val_m.get(f"cls{c}_acc", -1) >= 0
        )
        log.info(
            "Epoch %3d/%d | TrainLoss=%.4f (CLP=%.4f CTH=%.4f) "
            "| ValLoss=%.4f | OA=%.2f%% | CTH_RMSE=%.1f "
            "| LR=%.2e | %s | %.1fs",
            epoch, cfg.NUM_EPOCHS,
            train_m["loss"], train_m["clp"], train_m["cth"],
            val_m["loss"], val_m["oa"],
            val_m["cth_rmse"],
            lr_now, cls_str, elapsed
        )

        # ── Checkpoints ───────────────────────────────────────────────────
        if _is_better(val_m, best_loss, "val_loss"):
            best_loss = dict(val_m)
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST_LOSS)
            if monitor == "val_loss":
                torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)

        if _is_better(val_m, best_oa, "val_oa"):
            best_oa = dict(val_m)
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST_OA)
            if monitor == "val_oa":
                torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)

        if _is_better(val_m, best_macro, "val_macro_acc"):
            best_macro = dict(val_m)
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST_MACRO)
            if monitor == "val_macro_acc":
                torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)

        is_best = _is_better(val_m, best_selected, monitor)
        if is_best:
            best_selected = dict(val_m)
            epochs_no_best = 0
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)
            log.info("  ✓ New best %s: %.6f (OA=%.2f%% Macro=%.2f%%) → saved %s",
                     monitor, _metric_value(val_m, monitor), val_m["oa"],
                     val_m["macro_acc"], cfg.CHECKPOINT_BEST.name)
        else:
            epochs_no_best += 1

        torch.save(model.state_dict(), cfg.CHECKPOINT_LAST)

        # ── CSV log ───────────────────────────────────────────────────────
        row = dict(epoch=epoch, lr=lr_now, **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(cfg.LOG_DIR / "train_log.csv", index=False)

        # ── Early stopping ────────────────────────────────────────────────
        if epochs_no_best >= cfg.EARLY_STOP_PATIENCE:
            log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                     epoch, epochs_no_best)
            break

    if best_selected is not None:
        log.info("Training complete. Best %s: %.6f, OA: %.2f%%, Macro: %.2f%%",
                 monitor, _metric_value(best_selected, monitor),
                 best_selected["oa"], best_selected["macro_acc"])
