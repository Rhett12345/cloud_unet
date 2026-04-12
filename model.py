"""
model.py
========
HIR_COMP_UNet_DATrans_CLPenhance – AGRI-only (no GIIRS) variant.

Architecture:
  ConvNeXt encoder × 4 → DA-Block (PAM + CAM) → Transformer bottleneck →
  ConvNeXt decoder × 3 → split head: CLP (CrossEntropy) + COMP (SmoothL1)
  with CLP feature enhancement injected into the regression branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvNextBlock(nn.Module):
    """
    Depthwise → LayerNorm → pointwise×2 (GELU) with residual scaling.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw   = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1  = nn.Conv2d(channels, 2 * channels, 1)
        self.gelu = nn.GELU()
        self.pw2  = nn.Conv2d(2 * channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pw2(self.gelu(self.pw1(x)))
        return res + self.gamma * x


class PAM_Module(nn.Module):
    """Position Attention Module (non-local self-attention)."""
    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 8)
        self.q  = nn.Conv2d(channels, mid, 1)
        self.k  = nn.Conv2d(channels, mid, 1)
        self.v  = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.k(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)          # (B, HW, HW)
        v    = self.v(x).view(B, -1, H * W)
        out  = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class CAM_Module(nn.Module):
    """Channel Attention Module."""
    def __init__(self, channels: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)
        attn = torch.softmax(torch.bmm(flat, flat.permute(0, 2, 1)), dim=-1)
        out  = torch.bmm(attn, flat).view(B, C, H, W)
        return self.beta * out + x


class DA_Block(nn.Module):
    """Dual Attention Block: PAM ⊕ CAM."""
    def __init__(self, channels: int):
        super().__init__()
        self.pam  = PAM_Module(channels)
        self.cam  = CAM_Module(channels)
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.pam(x) + self.cam(x))


class TransformerEncoder(nn.Module):
    """Lightweight ViT-style encoder operating on flattened spatial tokens."""
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
                ),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)              # (B, T, C)
        for norm1, attn, norm2, mlp in self.layers:
            res, _ = attn(norm1(t), norm1(t), norm1(t))
            t = t + res
            t = t + mlp(norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class CloudPropertyNet(nn.Module):
    """
    HIR_COMP_UNet – AGRI-only variant.

    Outputs:
        clp_logits  : (B, CLP_CLASSES, H, W)  – raw logits for CrossEntropy
        comp_out    : (B, COMP_CHANNELS, H, W) – normalised regression values
                      channels: [CER, COT, CTH]
    """

    def __init__(
        self,
        agri_channels:  int = cfg.AGRI_CHANNELS,
        clp_classes:    int = cfg.CLP_CLASSES,
        comp_channels:  int = cfg.COMP_CHANNELS,
        base_ch:        int = cfg.MODEL_BASE_CHANNELS,
        trans_depth:    int = cfg.TRANSFORMER_DEPTH,
        trans_heads:    int = cfg.TRANSFORMER_HEADS,
        trans_mlp_dim:  int = cfg.TRANSFORMER_MLP_DIM,
    ):
        super().__init__()
        self.clp_classes   = clp_classes
        self.comp_channels = comp_channels
        C = base_ch

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = self._block(agri_channels, C)
        self.enc2 = self._block(C,     2 * C)
        self.enc3 = self._block(2 * C, 4 * C)
        self.enc4 = self._block(4 * C, 8 * C)

        # ── DA-Blocks ────────────────────────────────────────────────────
        self.da1 = DA_Block(C)
        self.da2 = DA_Block(2 * C)
        self.da3 = DA_Block(4 * C)
        self.da4 = DA_Block(8 * C)

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = TransformerEncoder(
            dim=8 * C, depth=trans_depth, heads=trans_heads, mlp_dim=trans_mlp_dim
        )

        # ── Decoder ──────────────────────────────────────────────────────
        self.dec3 = self._up_block(8 * C, 4 * C)
        self.dec2 = self._up_block(4 * C, 2 * C)
        self.dec1 = self._up_block(2 * C, C)

        # ── Output heads ─────────────────────────────────────────────────
        self.head = nn.Conv2d(C, clp_classes + comp_channels, 1)

        # ── CLP-enhance branch ───────────────────────────────────────────
        self.clp_enhance = nn.Sequential(
            nn.Conv2d(clp_classes, comp_channels, 1),
            nn.BatchNorm2d(comp_channels),
            nn.ReLU(),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            ConvNextBlock(out_ch),
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            ConvNextBlock(out_ch),
        )

    @staticmethod
    def _crop(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spatial crop x to match target's H×W."""
        _, _, H, W = target.shape
        return x[:, :, :H, :W]

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, agri: torch.Tensor):
        """
        agri : (B, AGRI_CHANNELS, H, W)
        Returns:
            clp_logits : (B, CLP_CLASSES,   H, W)
            comp_out   : (B, COMP_CHANNELS,  H, W)
        """
        # Encoder
        e1 = self.da1(self.enc1(agri))
        e2 = self.da2(self.enc2(F.max_pool2d(e1, 2)))
        e3 = self.da3(self.enc3(F.max_pool2d(e2, 2)))
        e4 = self.da4(self.enc4(F.max_pool2d(e3, 2)))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d3 = self.dec3(b  + self._crop(e4, b))
        d2 = self.dec2(d3 + self._crop(e3, d3))
        d1 = self.dec1(d2 + self._crop(e2, d2))

        out = self.head(d1)
        clp_logits = out[:, :self.clp_classes, :, :]
        comp_raw   = out[:, self.clp_classes:,  :, :]

        # Enhance regression with classification features
        comp_out = comp_raw + self.clp_enhance(clp_logits)

        return clp_logits, comp_out


def build_model() -> CloudPropertyNet:
    return CloudPropertyNet(
        agri_channels=cfg.AGRI_CHANNELS,
        clp_classes=cfg.CLP_CLASSES,
        comp_channels=cfg.COMP_CHANNELS,
        base_ch=cfg.MODEL_BASE_CHANNELS,
        trans_depth=cfg.TRANSFORMER_DEPTH,
        trans_heads=cfg.TRANSFORMER_HEADS,
        trans_mlp_dim=cfg.TRANSFORMER_MLP_DIM,
    )


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, cfg.AGRI_CHANNELS, 32, 32)
    clp, comp = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"CLP  shape : {clp.shape}")
    print(f"COMP shape : {comp.shape}")
    print(f"Parameters : {total:,}")
