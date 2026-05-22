"""
model.py
========
U-Net for GPM precipitation regression from AGRI tiles.

Architecture (~1.9M params):
  Input: (B, 9, 128, 128) — 7 BT channels + 2 geo (lat, lon)

  Encoder:
    enc1: Conv3×3×2(9→32)  + BN + ReLU   → (B, 32, 128, 128)
    pool1: MaxPool2×2                      → (B, 32, 64, 64)
    enc2: Conv3×3×2(32→64) + BN + ReLU    → (B, 64, 64, 64)
    pool2: MaxPool2×2                      → (B, 64, 32, 32)
    enc3: Conv3×3×2(64→128)+ BN + ReLU    → (B, 128, 32, 32)
    pool3: MaxPool2×2                      → (B, 128, 16, 16)

  Bottleneck:
    Conv3×3×2(128→256) + BN + ReLU        → (B, 256, 16, 16)

  Decoder:
    up3: Upsample + Conv + cat(enc3)       → (B, 256, 32, 32)
         Conv3×3×2(256→128) + BN + ReLU   → (B, 128, 32, 32)
    up2: Upsample + Conv + cat(enc2)       → (B, 128, 64, 64)
         Conv3×3×2(128→64) + BN + ReLU    → (B, 64, 64, 64)
    up1: Upsample + Conv + cat(enc1)       → (B, 64, 128, 128)
         Conv3×3×2(64→32) + BN + ReLU     → (B, 32, 128, 128)

  Heads:
    prob: Conv1×1(32→1) + Sigmoid         → (B, 1, 128, 128)
    rain: Conv1×1(32→1) + Softplus        → (B, 1, 128, 128)
    final = prob * rain                    → (B, 1, 128, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ConvBlock(nn.Module):
    """Two Conv3×3 + BN + ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PrecipUNet(nn.Module):
    """U-Net for precipitation regression from AGRI tiles.

    Input:  (B, 9, 128, 128)
    Output: (B, 1, 128, 128)  precipitation map (mm/h)
    """

    def __init__(self, in_channels: int = cfg.IN_CHANNELS):
        super().__init__()

        # ── Encoder ──
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(128, 256)

        # ── Decoder ──
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 1, bias=False),
        )
        self.dec3 = ConvBlock(256, 128)  # 128(up) + 128(skip) = 256

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 1, bias=False),
        )
        self.dec2 = ConvBlock(128, 64)   # 64(up) + 64(skip) = 128

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 1, bias=False),
        )
        self.dec1 = ConvBlock(64, 32)    # 32(up) + 32(skip) = 64

        # ── Dual heads ──
        self.head_logits = nn.Conv2d(32, 1, 1)    # raw logits for BCEWithLogits
        self.head_rain = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        """Return (prob_logits, rain_map), each (B, 1, H, W).
        prob_logits are raw (no sigmoid) — safe with AMP autocast.
        """
        # Encoder
        e1 = self.enc1(x)           # (B, 32, 128, 128)
        e2 = self.enc2(self.pool1(e1))   # (B, 64, 64, 64)
        e3 = self.enc3(self.pool2(e2))   # (B, 128, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # (B, 256, 16, 16)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 128, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 128, 128)

        prob_logits = self.head_logits(d1)   # (B, 1, 128, 128) — raw logits
        rain = self.head_rain(d1)            # (B, 1, 128, 128) — >= 0

        return prob_logits, rain

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return final precipitation map: sigmoid(logits) * rain."""
        prob_logits, rain = self.forward(x)
        return torch.sigmoid(prob_logits) * rain


def build_model() -> PrecipUNet:
    return PrecipUNet(in_channels=cfg.IN_CHANNELS)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, cfg.IN_CHANNELS, 128, 128)
    prob, rain = model(dummy)
    pred = model.predict(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Input:  {dummy.shape}")
    print(f"Prob:   {prob.shape}")
    print(f"Rain:   {rain.shape}")
    print(f"Pred:   {pred.shape}")
    print(f"Params: {total:,}")
