"""Self-contained Harmony Transformer (HT_2H9L_allihd) matching official latest_net_G.pth."""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchRearrange(nn.Module):
    """Pure-PyTorch equivalent of einops Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)')."""

    def __init__(self, p: int = 4):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        bs, ch, h, w = x.shape
        p = self.p
        return (
            x.reshape(bs, ch, h // p, p, w // p, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(bs, (h // p) * (w // p), p * p * ch)
        )


class HTTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class HTTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, pos=pos)
        return out


class HTTransformerEncoders(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 2,
        num_encoder_layers: int = 9,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        layer = HTTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = HTTransformerEncoder(layer, num_encoder_layers)

    def forward(self, src: Tensor, src_pos: Optional[Tensor] = None) -> Tensor:
        return self.encoder(src, pos=src_pos)


def build_ht_sine_pos(size: int = 256, ksize: int = 4, stride: int = 4, dim: int = 256) -> Tensor:
    """Build fixed 2D sine-cosine patch positional embedding of shape (H_f*W_f, 1, dim)."""
    feature_h = (size - ksize) // stride + 1
    num_pos_feats = dim // 2
    ones = torch.ones((feature_h, feature_h), dtype=torch.float32)
    y_embed = ones.cumsum(0, dtype=torch.float32)
    x_embed = ones.cumsum(1, dtype=torch.float32)
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = 10000.0 ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
    return pos.flatten(1).transpose(0, 1).unsqueeze(1).contiguous()


class HTGenerator(nn.Module):
    """Harmony Transformer generator expecting concat([comp, mask]) in [0, 1]."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ksize: int = 4,
        dim: int = 256,
        nhead: int = 2,
        num_layers: int = 9,
        dim_forward: int = 2,
        size: int = 256,
    ):
        super().__init__()
        self.patch_to_embedding = nn.Sequential(
            PatchRearrange(p=ksize),
            nn.Linear(ksize * ksize * (input_nc + 1), dim),
        )
        self.transformer_enc = HTTransformerEncoders(
            d_model=dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim * dim_forward,
            dropout=0.0,
            activation="gelu",
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, output_nc, kernel_size=ksize, stride=ksize, padding=0),
            nn.Tanh(),
        )
        self.register_buffer(
            "base_pos", build_ht_sine_pos(size=size, ksize=ksize, stride=ksize, dim=dim), persistent=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        patch_embedding = self.patch_to_embedding(inputs)
        bs, seq_len, ch = patch_embedding.size()
        pixel_pos = self.base_pos.expand(-1, bs, -1)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2), src_pos=pixel_pos)
        side = int(math.sqrt(seq_len))
        return self.dec(content.permute(1, 2, 0).contiguous().view(bs, ch, side, side))
