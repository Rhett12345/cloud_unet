"""
model.py
========
Simple U-Net for cloud property retrieval from AGRI data.

Architecture:
  Standard U-Net with double-conv blocks, max-pooling down, bilinear up,
  skip connections via concatenation.  Geo fields (lat, lon, VZA, SZA)
  are concatenated to the BT channels before the first encoder block.
"""

import torch
import torch.nn as nn

import config as cfg

GEO_CHANNELS = 4   # lat, lon, VZA, SZA


class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet(nn.Module):
    """Standard U-Net with concatenation skip connections.

    Input:  agri (B, AGRI_CHANNELS, H, W), geo (B, 4, H, W)
    Output: clp_logits (B, CLP_CLASSES, H, W), comp_out (B, COMP_CHANNELS, H, W)
    """

    def __init__(
        self,
        agri_channels: int = cfg.AGRI_CHANNELS,
        geo_channels: int = GEO_CHANNELS,
        clp_classes: int = cfg.CLP_CLASSES,
        comp_channels: int = cfg.COMP_CHANNELS,
        base_ch: int = cfg.UNET_BASE_CHANNELS,
    ):
        super().__init__()
        self.clp_classes = clp_classes
        self.comp_channels = comp_channels
        out_ch = clp_classes + comp_channels
        C = base_ch
        in_ch = agri_channels + geo_channels

        self.enc1 = DoubleConv(in_ch, C)
        self.enc2 = DoubleConv(C, 2 * C)
        self.enc3 = DoubleConv(2 * C, 4 * C)
        self.enc4 = DoubleConv(4 * C, 8 * C)
        self.bottleneck = DoubleConv(8 * C, 16 * C)

        self.dec3 = DoubleConv(16 * C + 8 * C, 8 * C)
        self.dec2 = DoubleConv(8 * C + 4 * C, 4 * C)
        self.dec1 = DoubleConv(4 * C + 2 * C, 2 * C)
        self.dec0 = DoubleConv(2 * C + C, C)

        self.head = nn.Conv2d(C, out_ch, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, agri: torch.Tensor, geo: torch.Tensor = None):
        if geo is not None:
            x = torch.cat([agri, geo], dim=1)
        else:
            # pad with zeros to match expected input channels
            missing = self.enc1.conv[0].in_channels - agri.shape[1]
            zeros = torch.zeros(agri.shape[0], missing, *agri.shape[2:],
                                device=agri.device, dtype=agri.dtype)
            x = torch.cat([agri, zeros], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d3 = self.dec3(torch.cat([self.up(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e2], dim=1))
        d0 = self.dec0(torch.cat([self.up(d1), e1], dim=1))

        out = self.head(d0)
        clp_logits = out[:, :self.clp_classes]
        comp_out = out[:, self.clp_classes:]

        return clp_logits, comp_out


def build_model() -> SimpleUNet:
    return SimpleUNet(
        agri_channels=cfg.AGRI_CHANNELS,
        geo_channels=GEO_CHANNELS,
        clp_classes=cfg.CLP_CLASSES,
        comp_channels=cfg.COMP_CHANNELS,
        base_ch=cfg.UNET_BASE_CHANNELS,
    )


if __name__ == "__main__":
    model = build_model()
    dummy_agri = torch.randn(4, cfg.AGRI_CHANNELS, 32, 32)
    dummy_geo = torch.randn(4, GEO_CHANNELS, 32, 32)
    clp, comp = model(dummy_agri, dummy_geo)
    total = sum(p.numel() for p in model.parameters())
    print(f"CLP  shape : {clp.shape}")
    print(f"COMP shape : {comp.shape}")
    print(f"Parameters : {total:,}")

