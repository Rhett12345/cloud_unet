"""
train.py
========
Training loop for AGRI → GPM precipitation regression (U-Net).

Features
--------
- Dual-head loss (BCE + weighted MSE)
- WeightedRandomSampler: rain tiles sampled 3× more often
- Mixed-precision (AMP) training
- Regression metrics: MAE, CSI, CC
- Multi-checkpoint saving (best loss, best CSI)
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as cfg
from dataset import NormStats, PrecipTileDataset
from losses import build_loss
from model import build_model

log = logging.getLogger(__name__)


def _seed_everything(seed: int = cfg.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Build dataloaders
# ─────────────────────────────────────────────────────────────────────────────

def _build_train_dl(stats: NormStats, batch_size: int = None):
    """Build train DataLoader with WeightedRandomSampler (rain tiles ×3)."""
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    train_ds = PrecipTileDataset(cfg.PAIRED_TRAIN_DIR, stats, mode="train")
    total = len(train_ds)
    log.info("train tiles (total) = %d", total)

    # Build sample weights from has_rain flag in index
    sample_weights = np.ones(total, dtype=np.float32)
    rain_count = 0
    for i, (_, _, has_rain) in enumerate(train_ds._index):
        if has_rain:
            sample_weights[i] = cfg.RAIN_SAMPLE_WEIGHT
            rain_count += 1
    log.info("Rain tiles: %d (%.1f%%), weight=%.0f×",
             rain_count, rain_count / max(total, 1) * 100, cfg.RAIN_SAMPLE_WEIGHT)

    n_epoch = max(1, int(total * cfg.SUBSAMPLE_FRAC))
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=n_epoch,
        replacement=True,
    )
    log.info("Sampler: %d tiles/epoch (%.0f%%)", n_epoch, cfg.SUBSAMPLE_FRAC * 100)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          sampler=sampler,
                          pin_memory=True, num_workers=cfg.NUM_WORKERS,
                          persistent_workers=True, prefetch_factor=4)
    return train_ds, train_dl, sample_weights, n_epoch


def _build_val_test_dls(stats: NormStats):
    """Build val/test DataLoaders."""
    val_ds  = PrecipTileDataset(cfg.PAIRED_VAL_DIR, stats, mode="val")
    test_ds = PrecipTileDataset(cfg.PAIRED_TEST_DIR, stats, mode="test")
    log.info("val tiles   = %d", len(val_ds))
    log.info("test tiles  = %d", len(test_ds))

    common = dict(pin_memory=True, num_workers=cfg.NUM_WORKERS,
                  persistent_workers=True, prefetch_factor=4)
    val_dl  = DataLoader(val_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, **common)
    return val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (regression + detection)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_detection_metrics(pred_mmh: np.ndarray, true_mmh: np.ndarray,
                                threshold: float = cfg.RAIN_THRESHOLD):
    """CSI, POD, FAR for rain detection at given threshold."""
    pred_rain = (pred_mmh > threshold)
    true_rain = (true_mmh > threshold)

    tp = int((pred_rain & true_rain).sum())
    fp = int((pred_rain & ~true_rain).sum())
    fn = int((~pred_rain & true_rain).sum())

    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    csi = tp / max(tp + fp + fn, 1)

    return {"pod": pod, "far": far, "csi": csi, "tp": tp, "fp": fp, "fn": fn}


def _compute_regression_metrics(pred_mmh: np.ndarray, true_mmh: np.ndarray):
    """MAE, RMSE, CC, Bias on rain pixels only."""
    mask = true_mmh > cfg.RAIN_THRESHOLD
    if mask.sum() < 2:
        return {"mae": 0.0, "rmse": 0.0, "cc": 0.0, "bias": 0.0, "n_rain": int(mask.sum())}

    p = pred_mmh[mask]
    t = true_mmh[mask]

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    bias = float(np.mean(p - t))

    # Pearson CC
    p_std = p.std()
    t_std = t.std()
    if p_std > 1e-9 and t_std > 1e-9:
        cc = float(np.corrcoef(p, t)[0, 1])
        cc = max(-1.0, min(1.0, cc)) if np.isfinite(cc) else 0.0
    else:
        cc = 0.0

    return {"mae": mae, "rmse": rmse, "cc": cc, "bias": bias, "n_rain": int(mask.sum())}


def _metric_value(metrics, monitor: str):
    monitor = (monitor or "val_loss").lower()
    if monitor in {"val_loss", "loss"}:
        return float(metrics.get("loss", 1e9))
    if monitor == "val_csi":
        return float(metrics.get("csi", 0.0))
    if monitor == "val_cc":
        return float(metrics.get("cc", 0.0))
    return 0.0


def _monitor_mode(monitor: str) -> str:
    return "min" if (monitor or "").lower() in {"val_loss", "loss"} else "max"


def _is_better(candidate, current, monitor: str) -> bool:
    if current is None:
        return True
    cv = _metric_value(candidate, monitor)
    cv_cur = _metric_value(current, monitor)
    return cv < cv_cur if _monitor_mode(monitor) == "min" else cv > cv_cur


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "n": 0}
    # Accumulate for per-pixel detection metrics
    all_preds = []
    all_trues = []

    total_batches = len(loader)
    log_interval = max(1, total_batches // 8)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad()

        with autocast(device.type if scaler else "cpu", enabled=(scaler is not None)):
            logits, rain = model(x)
            loss = loss_fn(logits, rain, y)

        if training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

        B = x.shape[0]
        totals["loss"] += loss.item() * B
        totals["n"]   += B

        # Accumulate predictions for metrics (CPU, only at log intervals)
        if (batch_idx + 1) % log_interval == 0 or batch_idx == total_batches - 1:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pred_mmh = (prob * rain).detach().cpu().numpy().ravel()
                true_mmh = y.detach().cpu().numpy().ravel()
                all_preds.append(pred_mmh)
                all_trues.append(true_mmh)

            tag = "train" if training else "val"
            log.info("  %s %d/%d | loss=%.4f",
                     tag, batch_idx + 1, total_batches,
                     totals["loss"] / max(totals["n"], 1))

    N = max(totals["n"], 1)
    loss_avg = totals["loss"] / N

    if all_preds:
        preds_all = np.concatenate(all_preds)
        trues_all = np.concatenate(all_trues)
        det = _compute_detection_metrics(preds_all, trues_all)
        reg = _compute_regression_metrics(preds_all, trues_all)
    else:
        det = {"pod": 0, "far": 0, "csi": 0, "tp": 0, "fp": 0, "fn": 0}
        reg = {"mae": 0, "rmse": 0, "cc": 0, "bias": 0, "n_rain": 0}

    return {
        "loss": loss_avg,
        **det,
        **reg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(stats: NormStats, resume_checkpoint: str = None):
    _seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    multi_gpu = n_gpus > 1
    log.info("Training on %s (%d GPUs)", device, max(n_gpus, 1))

    # Scale batch size with GPU count (keep per-GPU workload constant)
    train_batch = cfg.BATCH_SIZE * max(n_gpus, 1)

    train_ds, train_dl, sample_weights, n_epoch = _build_train_dl(stats, batch_size=train_batch)
    val_dl, test_dl = _build_val_test_dls(stats)

    log.info("train iters/epoch = %d", len(train_dl))
    log.info("val iters/epoch   = %d", len(val_dl))

    model = build_model()
    if multi_gpu:
        model = nn.DataParallel(model)
        log.info("Wrapped model with DataParallel across %d GPUs", n_gpus)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR,
    )
    scaler = GradScaler(device.type) if torch.cuda.is_available() else None

    loss_fn = build_loss()
    log.info("Loss: DualHeadLoss (BCE×0.4 + wMSE×0.6, rain_weight=2)")

    # Helper: get state_dict accounting for DataParallel wrapper
    def _state_dict():
        return model.module.state_dict() if multi_gpu else model.state_dict()

    monitor = getattr(cfg, "CHECKPOINT_MONITOR", "val_csi")
    best_selected = None
    best_loss = None
    best_csi = None
    epochs_no_best = 0
    log_rows = []
    start_epoch = 1

    # ── Resume from checkpoint ──
    if resume_checkpoint:
        ckpt_path = Path(resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if multi_gpu:
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        log.info("Loaded model weights from %s", ckpt_path)

        # Try loading full training state
        opt_path = ckpt_path.with_suffix(".opt.pth")
        if opt_path.exists():
            ckpt = torch.load(opt_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 1) + 1
            best_selected = ckpt.get("best_selected")
            best_loss = ckpt.get("best_loss")
            best_csi = ckpt.get("best_csi")
            epochs_no_best = ckpt.get("epochs_no_best", 0)
            log.info("Resumed optimizer/scheduler/scaler from epoch %d", ckpt.get("epoch", 0))
        else:
            # Model-only resume: keep default optimizer state
            log.info("No .opt.pth found — using fresh optimizer (lr=%.0e)", cfg.LEARNING_RATE)

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        # Rebuild train sampler each epoch (different subset)
        if epoch > 1:
            del train_dl
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights),
                num_samples=n_epoch,
                replacement=True,
            )
            train_dl = DataLoader(train_ds, batch_size=train_batch,
                                  sampler=sampler,
                                  pin_memory=True, num_workers=cfg.NUM_WORKERS,
                                  persistent_workers=True, prefetch_factor=4)

        train_m = _run_epoch(model, train_dl, loss_fn, device,
                             optimizer=optimizer, scaler=scaler)
        val_m   = _run_epoch(model, val_dl,   loss_fn, device)

        scheduler.step(val_m["csi"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | TrainLoss=%.4f | ValLoss=%.4f | "
            "CSI=%.3f POD=%.3f FAR=%.3f | MAE=%.3f RMSE=%.3f CC=%.3f | %.1fs",
            epoch, cfg.NUM_EPOCHS,
            train_m["loss"], val_m["loss"],
            val_m["csi"], val_m["pod"], val_m["far"],
            val_m["mae"], val_m["rmse"], val_m["cc"],
            elapsed,
        )

        # Checkpoints
        if _is_better(val_m, best_loss, "val_loss"):
            best_loss = dict(val_m)
            torch.save(_state_dict(), cfg.CHECKPOINT_BEST_LOSS)
            if monitor == "val_loss":
                torch.save(_state_dict(), cfg.CHECKPOINT_BEST)

        if _is_better(val_m, best_csi, "val_csi"):
            best_csi = dict(val_m)
            torch.save(_state_dict(), cfg.CHECKPOINT_BEST_CSI)
            if monitor == "val_csi":
                torch.save(_state_dict(), cfg.CHECKPOINT_BEST)

        is_best = _is_better(val_m, best_selected, monitor)
        if is_best:
            best_selected = dict(val_m)
            epochs_no_best = 0
            torch.save(_state_dict(), cfg.CHECKPOINT_BEST)
            log.info("  New best %s: %.4f CSI=%.3f CC=%.3f",
                     monitor, _metric_value(val_m, monitor),
                     val_m["csi"], val_m["cc"])
        else:
            epochs_no_best += 1

        torch.save(_state_dict(), cfg.CHECKPOINT_LAST)

        # Save optimizer state for resume
        opt_state = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_selected": best_selected,
            "best_loss": best_loss,
            "best_csi": best_csi,
            "epochs_no_best": epochs_no_best,
        }
        if scaler:
            opt_state["scaler"] = scaler.state_dict()
        torch.save(opt_state, cfg.MODEL_DIR / f"{cfg.MODEL_NAME}_last.opt.pth")

        row = dict(epoch=epoch, lr=lr_now,
                   **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(cfg.LOG_DIR / "train_log.csv", index=False)

        if epochs_no_best >= cfg.EARLY_STOP_PATIENCE:
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best %s CSI:%.3f CC:%.3f MAE:%.3f",
             monitor,
             best_selected["csi"] if best_selected else 0,
             best_selected["cc"] if best_selected else 0,
             best_selected["mae"] if best_selected else 0)
