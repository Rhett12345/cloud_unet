"""
losses.py
=========
Dual-head loss for precipitation regression: BCE (rain probability) + weighted MSE.

Usage:
    from losses import build_loss
    loss_fn = build_loss()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class DualHeadLoss(nn.Module):
    """BCEWithLogits for rain/no-rain + weighted MSE for precipitation amount.

    Total = bce_weight * BCEWithLogits(prob_logits, rain_mask)
          + mse_weight * weighted_MSE(rain_pred, target)

    Rain pixels (target > threshold) get rain_weight × higher MSE weight.
    """

    def __init__(
        self,
        rain_weight: float = 2.0,
        bce_weight: float = 0.4,
        mse_weight: float = 0.6,
        threshold: float = cfg.RAIN_THRESHOLD,
    ):
        super().__init__()
        self.rain_weight = rain_weight
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.threshold = threshold

    def forward(
        self,
        prob_logits: torch.Tensor,
        rain_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prob_logits: (B, 1, H, W)  raw logits (no sigmoid) — safe with AMP
            rain_pred:   (B, 1, H, W)  softplus precipitation amount (mm/h)
            target:      (B, 1, H, W)  GPM precipitation (mm/h), NaN→0
        """
        rain_mask = (target > self.threshold).float()

        # BCEWithLogits: numerically safe with AMP autocast
        bce = F.binary_cross_entropy_with_logits(
            prob_logits, rain_mask, reduction="mean"
        )

        # Weighted MSE: rain pixels get higher weight
        pixel_weight = 1.0 + (self.rain_weight - 1.0) * rain_mask
        diff = rain_pred - target
        mse = (pixel_weight * diff * diff).mean()

        return self.bce_weight * bce + self.mse_weight * mse


def build_loss() -> DualHeadLoss:
    return DualHeadLoss()


if __name__ == "__main__":
    loss_fn = build_loss()
    logits = torch.randn(2, 1, 128, 128)       # raw logits
    rain = torch.rand(2, 1, 128, 128) * 5      # softplus output
    target = torch.rand(2, 1, 128, 128) * 10
    target[target < 0.05] = 0.0
    loss = loss_fn(logits, rain, target)
    print(f"Loss: {loss.item():.4f}")
