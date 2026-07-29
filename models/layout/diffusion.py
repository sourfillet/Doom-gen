"""DDPM training objective + DDIM sampling with classifier-free guidance
(plan/generation.md, Stage 1). Model space is [-1, 1]; the dataset's
[0, 1] rasters are mapped in by the caller (x * 2 - 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cumprod[t][:, None, None, None]
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def loss(self, x0: torch.Tensor, cond: torch.Tensor,
             channel_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Epsilon-prediction MSE at a uniformly sampled t. channel_weights
        (C,) upweights specific channels' denoising accuracy - used to make
        the model prioritize the sharp binary occupancy channel over the
        smooth height channels, so it actually learns void."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t, cond)
        if channel_weights is None:
            return F.mse_loss(pred, noise)
        w = channel_weights.to(x0.device).view(1, -1, 1, 1)
        return (F.mse_loss(pred, noise, reduction="none") * w).mean()

    @torch.no_grad()
    def ddim_sample(self, shape, cond: torch.Tensor, uncond: torch.Tensor,
                    steps: int = 50, guidance: float = 2.0,
                    device: str = "cpu") -> torch.Tensor:
        """Deterministic DDIM (eta=0). guidance = CFG scale: 0 is fully
        unconditional; higher leans harder on the tags at the cost of
        corpus-wide structural prior (the knob from plan/generation.md,
        "Blending tag-conditioning with corpus-wide quality")."""
        x = torch.randn(shape, device=device)
        taus = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i in range(len(taus)):
            t = taus[i].expand(shape[0])
            eps_c = self.model(x, t, cond)
            if guidance > 0:
                eps_u = self.model(x, t, uncond)
                eps = (1 + guidance) * eps_c - guidance * eps_u
            else:
                eps = eps_c
            ac_t = self.alphas_cumprod[taus[i]]
            x0_pred = ((x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()).clamp(-1, 1)
            if i == len(taus) - 1:
                x = x0_pred
            else:
                ac_prev = self.alphas_cumprod[taus[i + 1]]
                x = ac_prev.sqrt() * x0_pred + (1 - ac_prev).sqrt() * eps
        return x
