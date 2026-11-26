"""Improved Harmonization U-Net v2 (Hybrid): GatedConv + FG-BG Cross-Attention + Color Affine Head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from iharmony.models.blocks import AttentionGate, BilinearUp, GatedDoubleConv, StandardDoubleConv


class BottleneckFGBGCrossAttention(nn.Module):
    def __init__(self, channels: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, feat: Tensor, mask: Tensor) -> Tensor:
        bs, ch, h, w = feat.shape
        m_low = F.adaptive_avg_pool2d(mask, (h, w))
        m_flat = m_low.view(bs, h * w)
        tokens = feat.flatten(2).transpose(1, 2)
        q, kv = self.norm_q(tokens), self.norm_kv(tokens)

        key_padding_mask = m_flat > 0.5
        all_fg = key_padding_mask.all(dim=1)
        if all_fg.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_fg] = False

        attn_out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
        fg_gate_tokens = (m_flat > 0.2).unsqueeze(-1).to(feat.dtype)
        tokens = tokens + self.gamma1 * (attn_out * fg_gate_tokens)
        tokens = tokens + self.gamma2 * (self.ffn(self.norm_ffn(tokens)) * fg_gate_tokens)
        return tokens.transpose(1, 2).view(bs, ch, h, w)


class GlobalLocalColorAffineHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.local_conv = nn.Conv2d(in_ch, 12, kernel_size=3, padding=1)
        self.global_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, 12))

    def forward(self, feat: Tensor, comp: Tensor) -> Tensor:
        bs, _, h, w = feat.shape
        params = self.local_conv(feat) + self.global_fc(feat).view(bs, 12, 1, 1)
        dw = params[:, :9].view(bs, 3, 3, h, w)
        db = params[:, 9:].view(bs, 3, h, w)
        eye = torch.eye(3, device=feat.device, dtype=feat.dtype).view(1, 3, 3, 1, 1)
        pred = torch.einsum("bijhw,bjhw->bihw", (eye + dw).float(), comp.float()) + db.float()
        return torch.clamp(pred, 0.0, 1.0).to(feat.dtype)


class ImprovedHarmonizationUNet(nn.Module):
    def __init__(self, base_channels: int = 32, num_heads: int = 8):
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.enc1 = GatedDoubleConv(4, c1)
        self.enc2 = GatedDoubleConv(c1, c2)
        self.enc3 = GatedDoubleConv(c2, c3)
        self.enc4 = GatedDoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = GatedDoubleConv(c4, c5)
        self.fg_bg_attn = BottleneckFGBGCrossAttention(channels=c5, num_heads=num_heads)

        self.up4 = BilinearUp(c5, c4)
        self.att4 = AttentionGate(g_ch=c4, x_ch=c4)
        self.dec4 = StandardDoubleConv(c4 * 2, c4)

        self.up3 = BilinearUp(c4, c3)
        self.att3 = AttentionGate(g_ch=c3, x_ch=c3)
        self.dec3 = StandardDoubleConv(c3 * 2, c3)

        self.up2 = BilinearUp(c3, c2)
        self.att2 = AttentionGate(g_ch=c2, x_ch=c2)
        self.dec2 = StandardDoubleConv(c2 * 2, c2)

        self.up1 = BilinearUp(c2, c1)
        self.att1 = AttentionGate(g_ch=c1, x_ch=c1)
        self.dec1 = StandardDoubleConv(c1 * 2, c1)

        self.head = GlobalLocalColorAffineHead(c1)

    def forward(self, x: Tensor) -> Tensor:
        comp, mask = x[:, :3], x[:, 3:4]
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        bot = self.fg_bg_attn(self.bottleneck(self.pool(s4)), mask)

        g4 = self.up4(bot)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, s4)], dim=1))
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, s3)], dim=1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, s2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, s1)], dim=1))
        return self.head(d1, comp)
