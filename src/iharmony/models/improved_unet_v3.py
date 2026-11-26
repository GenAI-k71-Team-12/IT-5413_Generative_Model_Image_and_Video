"""Improved Harmonization U-Net v3 (DINOv2 + Lab): Frozen DINOv2 Guide + Hybrid Affine Residual Head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from iharmony.models.blocks import AttentionGate, BilinearUp, GatedDoubleConv, StandardDoubleConv


class FrozenDINOv2Guide(nn.Module):
    def __init__(self, out_channels: int = 512):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        self.proj = nn.Sequential(
            nn.Conv2d(384, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, rgb: Tensor, target_hw: tuple[int, int]) -> Tensor:
        with torch.no_grad():
            x = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            feats = self.dino.forward_features(x)["x_norm_patchtokens"]
            bs, num_tokens, ch = feats.shape
            h_p = w_p = int(math.sqrt(num_tokens))
            feat_map = feats.transpose(1, 2).view(bs, ch, h_p, w_p)

        if (h_p, w_p) != target_hw:
            feat_map = F.interpolate(feat_map, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(feat_map)


class SemanticGuidedFGBGCrossAttention(nn.Module):
    def __init__(self, channels: int = 512, num_heads: int = 8):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 256, channels) * 0.02)
        self.norm_sem_q = nn.LayerNorm(channels)
        self.norm_sem_k = nn.LayerNorm(channels)
        self.norm_val = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_feat: Tensor, sem_feat: Tensor, mask: Tensor) -> Tensor:
        bs, ch, h, w = spatial_feat.shape
        m_low = F.adaptive_avg_pool2d(mask, (h, w))
        m_flat = m_low.view(bs, h * w)

        sp_tokens = spatial_feat.flatten(2).transpose(1, 2)
        sem_tokens = sem_feat.flatten(2).transpose(1, 2)

        pos = (
            self.pos_emb
            if h * w == self.pos_emb.shape[1]
            else F.interpolate(
                self.pos_emb.transpose(1, 2).view(1, ch, 16, 16),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .flatten(2)
            .transpose(1, 2)
        )

        q = self.norm_sem_q(sem_tokens + pos)
        k = self.norm_sem_k(sem_tokens + pos)
        v = self.norm_val(sp_tokens)

        bg_key_padding_mask = m_flat > 0.5
        all_fg = bg_key_padding_mask.all(dim=1)
        if all_fg.any():
            bg_key_padding_mask = bg_key_padding_mask.clone()
            bg_key_padding_mask[all_fg] = False

        attn_out, _ = self.attn(
            query=q, key=k, value=v, key_padding_mask=bg_key_padding_mask, need_weights=False
        )

        fg_gate = (m_flat > 0.2).unsqueeze(-1).to(spatial_feat.dtype)
        sp_tokens = sp_tokens + self.gamma1 * (attn_out * fg_gate)
        sp_tokens = sp_tokens + self.gamma2 * (self.ffn(self.norm_ffn(sp_tokens)) * fg_gate)
        return sp_tokens.transpose(1, 2).view(bs, ch, h, w)


class HybridColorAffineResidualHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.local_conv = nn.Conv2d(in_ch, 12, kernel_size=3, padding=1)
        self.global_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, 12))

        self.chroma_res = nn.Sequential(
            nn.Conv2d(in_ch + 3, in_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, in_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat: Tensor, comp: Tensor, mask: Tensor) -> Tensor:
        bs, _, h, w = feat.shape
        params = self.local_conv(feat) + self.global_fc(feat).view(bs, 12, 1, 1)
        dw = params[:, :9].view(bs, 3, 3, h, w)
        db = params[:, 9:].view(bs, 3, h, w)
        eye = torch.eye(3, device=feat.device, dtype=feat.dtype).view(1, 3, 3, 1, 1)

        affine_pred = torch.einsum("bijhw,bjhw->bihw", (eye + dw).float(), comp.float()) + db.float()
        delta_rgb = self.chroma_res(
            torch.cat([feat, affine_pred.clamp(0.0, 1.0).to(feat.dtype)], dim=1)
        ).float()

        pred = affine_pred + self.res_scale.float() * delta_rgb * mask.float()
        return torch.clamp(pred, 0.0, 1.0).to(feat.dtype)


class ImprovedHarmonizationUNetV3(nn.Module):
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
        self.enc3 = StandardDoubleConv(c2, c3)
        self.enc4 = StandardDoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = StandardDoubleConv(c4, c5)
        self.sem_guide = FrozenDINOv2Guide(out_channels=c5)
        self.fg_bg_attn = SemanticGuidedFGBGCrossAttention(channels=c5, num_heads=num_heads)

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

        self.head = HybridColorAffineResidualHead(c1)

    def forward(self, x: Tensor) -> Tensor:
        comp, mask = x[:, :3], x[:, 3:4]
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        bot_spatial = self.bottleneck(self.pool(s4))
        bot_sem = self.sem_guide(comp, target_hw=bot_spatial.shape[-2:])
        bot = self.fg_bg_attn(bot_spatial, bot_sem, mask)

        g4 = self.up4(bot)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, s4)], dim=1))
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, s3)], dim=1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, s2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, s1)], dim=1))
        return self.head(d1, comp, mask)
