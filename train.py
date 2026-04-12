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
# Data augmentation (geometric only – safe for all channels)
# ─────────────────────────────────────────────────────────────────────────────

def _augment(agri: torch.Tensor, lbl: torch.Tensor):
    if random.random() < 0.5:
        agri = torch.flip(agri, dims=[3])
        lbl  = torch.flip(lbl,  dims=[3])
    if random.random() < 0.5:
        agri = torch.flip(agri, dims=[2])
        lbl  = torch.flip(lbl,  dims=[2])
    k = random.randint(0, 3)
    if k:
        agri = torch.rot90(agri, k=k, dims=[2, 3])
        lbl  = torch.rot90(lbl,  k=k, dims=[2, 3])
    return agri, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

# def _compute_losses(clp_logits, comp_out, labels, ce_loss_fn, reg_loss_fn):
#     """
#     labels  : (B, 4, H, W)   ch0=CLP(int class), ch1-3=normalised regression
#     """
#     # CLP classification loss
#     clp_target = labels[:, 0, :, :].long()
#     loss_clp   = ce_loss_fn(clp_logits, clp_target) * cfg.LOSS_W_CLP
#
#     # Regression losses  (CER=ch0, COT=ch1, CTH=ch2 in comp_out)
#     loss_cer = reg_loss_fn(comp_out[:, 0], labels[:, 1]) * cfg.LOSS_W_CER
#     loss_cot = reg_loss_fn(comp_out[:, 1], labels[:, 2]) * cfg.LOSS_W_COT
#     loss_cth = reg_loss_fn(comp_out[:, 2], labels[:, 3]) * cfg.LOSS_W_CTH
#
#     total = loss_clp + loss_cer + loss_cot + loss_cth
#     return total, loss_clp, loss_cer, loss_cot, loss_cth
#
# def _compute_losses(clp_logits, comp_out, labels, ce_loss_fn, reg_loss_fn):
#     """
#     labels: (B, 4, H, W)
#     ch0=CLP, ch1-3=regression
#     """
#     # ---------- CLP classification ----------
#     clp_raw = labels[:, 0, :, :]
#     valid_clp = torch.isfinite(clp_raw) & (clp_raw >= 0) & (clp_raw < cfg.CLP_CLASSES)
#
#     clp_target = torch.where(
#         valid_clp,
#         clp_raw,
#         torch.full_like(clp_raw, -100)
#     ).long()
#
#     loss_clp = ce_loss_fn(clp_logits, clp_target) * cfg.LOSS_W_CLP
#
#     # ---------- regression masked loss ----------
#     def masked_reg_loss(pred, target, weight):
#         valid = torch.isfinite(target)
#         if valid.any():
#             return reg_loss_fn(pred[valid], target[valid]) * weight
#         return pred.new_tensor(0.0)
#
#     loss_cer = masked_reg_loss(comp_out[:, 0], labels[:, 1], cfg.LOSS_W_CER)
#     loss_cot = masked_reg_loss(comp_out[:, 1], labels[:, 2], cfg.LOSS_W_COT)
#     loss_cth = masked_reg_loss(comp_out[:, 2], labels[:, 3], cfg.LOSS_W_CTH)
#
#     total = loss_clp + loss_cer + loss_cot + loss_cth
#     return total, loss_clp, loss_cer, loss_cot, loss_cth

def _compute_losses(clp_logits, comp_out, labels, ce_loss_fn, reg_loss_fn):
    """
    labels: (B, 4, H, W)
    ch0 = CLP
    ch1-3 = normalised regression labels
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

    # ---- Regression masked loss ----
    def masked_reg_loss(pred, target, weight):
        valid = torch.isfinite(target)
        if valid.any():
            return reg_loss_fn(pred[valid], target[valid]) * weight
        return zero

    loss_cer = masked_reg_loss(comp_out[:, 0], labels[:, 1], cfg.LOSS_W_CER)
    loss_cot = masked_reg_loss(comp_out[:, 1], labels[:, 2], cfg.LOSS_W_COT)
    loss_cth = masked_reg_loss(comp_out[:, 2], labels[:, 3], cfg.LOSS_W_CTH)

    total = loss_clp + loss_cer + loss_cot + loss_cth
    return total, loss_clp, loss_cer, loss_cot, loss_cth

# @torch.no_grad()
# def _batch_metrics(clp_logits, comp_out, labels, stats: NormStats, device):
#     """Return OA (%), CER RMSE, COT RMSE, CTH RMSE (de-normalised units)."""
#     clp_pred = clp_logits.argmax(dim=1)
#     oa = (clp_pred == labels[:, 0].long()).float().mean().item() * 100.0
#
#     out_std  = torch.from_numpy(stats.out_std[1:]).to(device).reshape(1, 3, 1, 1)
#     out_mean = torch.from_numpy(stats.out_mean[1:]).to(device).reshape(1, 3, 1, 1)
#
#     # De-normalise predictions and targets
#     pred_dn = comp_out * out_std + out_mean
#     true_dn = labels[:, 1:] * out_std + out_mean
#
#     def rmse(a, b):
#         return torch.sqrt(torch.mean((a - b) ** 2)).item()
#
#     return oa, rmse(pred_dn[:, 0], true_dn[:, 0]), \
#                rmse(pred_dn[:, 1], true_dn[:, 1]), \
#                rmse(pred_dn[:, 2], true_dn[:, 2])

@torch.no_grad()
def _batch_metrics(clp_logits, comp_out, labels, stats: NormStats, device):
    clp_pred = clp_logits.argmax(dim=1)
    clp_true = labels[:, 0]

    valid_clp = torch.isfinite(clp_true) & (clp_true >= 0) & (clp_true < cfg.CLP_CLASSES)
    if valid_clp.any():
        oa = (clp_pred[valid_clp] == clp_true[valid_clp].long()).float().mean().item() * 100.0
    else:
        oa = 0.0

    out_std  = torch.from_numpy(stats.out_std[1:]).to(device).reshape(1, 3, 1, 1)
    out_mean = torch.from_numpy(stats.out_mean[1:]).to(device).reshape(1, 3, 1, 1)

    pred_dn = comp_out * out_std + out_mean
    true_dn = labels[:, 1:] * out_std + out_mean

    def rmse_masked(a, b):
        valid = torch.isfinite(b)
        if valid.any():
            return torch.sqrt(torch.mean((a[valid] - b[valid]) ** 2)).item()
        return 0.0

    return (
        oa,
        rmse_masked(pred_dn[:, 0], true_dn[:, 0]),
        rmse_masked(pred_dn[:, 1], true_dn[:, 1]),
        rmse_masked(pred_dn[:, 2], true_dn[:, 2]),
    )

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

    totals = dict(loss=0, clp=0, cer=0, cot=0, cth=0,
                  oa=0, cer_rmse=0, cot_rmse=0, cth_rmse=0, n=0)

    for agri, _geo, labels in loader:
        agri   = agri.to(device)
        labels = labels.to(device)

        if training:
            agri, labels = _augment(agri, labels)

        # if training and torch.rand(1).item() < 0.001:
        #     clp = labels[:, 0]
        #     print("CLP finite:", torch.isfinite(clp).sum().item(), "/", clp.numel())
        #     print("CLP min/max(valid):",
        #           clp[torch.isfinite(clp)].min().item() if torch.isfinite(clp).any() else None,
        #           clp[torch.isfinite(clp)].max().item() if torch.isfinite(clp).any() else None)

        with autocast(enabled=(scaler is not None)):
            clp_logits, comp_out = model(agri)
            total, l_clp, l_cer, l_cot, l_cth = _compute_losses(
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

        oa, cer_r, cot_r, cth_r = _batch_metrics(clp_logits, comp_out, labels, stats, device)

        B = agri.shape[0]
        totals["loss"]     += total.item()  * B
        totals["clp"]      += l_clp.item()  * B
        totals["cer"]      += l_cer.item()  * B
        totals["cot"]      += l_cot.item()  * B
        totals["cth"]      += l_cth.item()  * B
        totals["oa"]       += oa            * B
        totals["cer_rmse"] += cer_r         * B
        totals["cot_rmse"] += cot_r         * B
        totals["cth_rmse"] += cth_r         * B
        totals["n"]        += B

    N = max(totals["n"], 1)
    return {k: v / N for k, v in totals.items() if k != "n"}


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
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR
    )
    scaler = GradScaler() if torch.cuda.is_available() else None

    ce_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    reg_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    log_rows      = []

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = _run_epoch(model, train_dl, ce_fn, reg_fn, stats, device,
                             optimizer=optimizer, scaler=scaler)
        val_m   = _run_epoch(model, val_dl,   ce_fn, reg_fn, stats, device)

        scheduler.step(val_m["loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | TrainLoss=%.4f (CLP=%.4f CER=%.4f COT=%.4f CTH=%.4f) "
            "| ValLoss=%.4f | OA=%.2f%% | CER_RMSE=%.2f COT_RMSE=%.2f CTH_RMSE=%.1f "
            "| LR=%.2e | %.1fs",
            epoch, cfg.NUM_EPOCHS,
            train_m["loss"], train_m["clp"], train_m["cer"],
            train_m["cot"],  train_m["cth"],
            val_m["loss"], val_m["oa"],
            val_m["cer_rmse"], val_m["cot_rmse"], val_m["cth_rmse"],
            lr_now, elapsed
        )

        # ── Checkpoints ───────────────────────────────────────────────────
        torch.save(model.state_dict(), cfg.CHECKPOINT_LAST)
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)
            log.info("  ✓ New best val loss: %.6f → saved %s",
                     best_val_loss, cfg.CHECKPOINT_BEST.name)

        # ── CSV log ───────────────────────────────────────────────────────
        row = dict(epoch=epoch, lr=lr_now, **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(cfg.LOG_DIR / "train_log.csv", index=False)

    log.info("Training complete. Best val loss: %.6f", best_val_loss)
