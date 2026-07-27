"""Compact conditional U-Net for layout diffusion (plan/generation.md,
Stage 1). Time + tag conditioning enter every residual block via FiLM
(per-channel scale/shift), self-attention at the bottleneck. Sized for a
128px 9-channel raster on an 8 GB GPU (~20M params at base=64).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        out, _ = self.attn(flat, flat, flat, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 9, base: int = 64,
                 mults: tuple[int, ...] = (1, 2, 4), emb_dim: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )
        self.cond_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                       nn.Linear(emb_dim, emb_dim))

        chans = [base * m for m in mults]
        self.stem = nn.Conv2d(in_channels, chans[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i, ch in enumerate(chans):
            prev = chans[max(i - 1, 0)]
            self.down_blocks.append(nn.ModuleList([
                ResBlock(prev if i else ch, ch, emb_dim), ResBlock(ch, ch, emb_dim),
            ]))
            self.downsamplers.append(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity())

        mid = chans[-1]
        self.mid1 = ResBlock(mid, mid, emb_dim)
        self.mid_attn = SelfAttention2d(mid)
        self.mid2 = ResBlock(mid, mid, emb_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in reversed(range(len(chans))):
            ch = chans[i]
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch * 2, ch, emb_dim), ResBlock(ch, ch, emb_dim),
            ]))
            self.upsamplers.append(
                nn.ConvTranspose2d(ch, chans[max(i - 1, 0)], 4, stride=2, padding=1)
                if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, chans[0])
        self.out_conv = nn.Conv2d(chans[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t) + self.cond_proj(cond)
        h = self.stem(x)
        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for blocks, up in zip(self.up_blocks, self.upsamplers):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, emb)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))
