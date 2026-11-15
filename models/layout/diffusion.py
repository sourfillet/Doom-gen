"""DDPM training objective + DDIM sampling with classifier-free guidance
(Stage 1). Model space is [-1, 1]; the dataset's
[0, 1] rasters are mapped in by the caller (x * 2 - 1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    # Sampling semantics are independent of the trained weights. Old positive
    # guidance g corresponds to scale g+1 in this convention.
    guidance_convention = "unconditional_plus_scale_v1"

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
        w = channel_weights.to(device=x0.device, dtype=pred.dtype)
        if w.shape != (x0.shape[1],):
            raise ValueError(f"channel_weights shape {tuple(w.shape)} must match channels ({x0.shape[1]},)")
        return (F.mse_loss(pred, noise, reduction="none") * w.view(1, -1, 1, 1)).mean()

    @torch.no_grad()
    def ddim_sample(self, shape, cond: torch.Tensor, uncond: torch.Tensor,
                    steps: int = 50, guidance: float = 2.0,
                    device: str | torch.device | None = None) -> torch.Tensor:
        """Deterministic DDIM (eta=0). CFG scale: 0 is unconditional,
        1 is conditional, >1 extrapolates toward the tags at the cost of
        corpus-wide structural prior."""
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and non-negative")
        if not 1 <= steps <= self.timesteps:
            raise ValueError(f"steps must be between 1 and {self.timesteps}")
        if device is None:
            device = cond.device
        x = torch.randn(shape, device=device)
        taus = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i in range(len(taus)):
            t = taus[i].expand(shape[0])
            if guidance == 0:
                eps = self.model(x, t, uncond)
            elif guidance == 1:
                eps = self.model(x, t, cond)
            else:
                eps_c = self.model(x, t, cond)
                eps_u = self.model(x, t, uncond)
                eps = eps_u + guidance * (eps_c - eps_u)
            ac_t = self.alphas_cumprod[taus[i]]
            x0_pred = ((x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()).clamp(-1, 1)
            if i == len(taus) - 1:
                x = x0_pred
            else:
                eps_hat = (x - ac_t.sqrt() * x0_pred) / (1 - ac_t).sqrt()
                ac_prev = self.alphas_cumprod[taus[i + 1]]
                x = ac_prev.sqrt() * x0_pred + (1 - ac_prev).sqrt() * eps_hat
        return x
