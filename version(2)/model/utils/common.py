"""
Shared neural network components: condition encoder and sinusoidal time
embedding. Both diffusion models use these so they live in one place.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


N_DAYS    = 7
N_MONTHS  = 12
N_LEAPS   = 2
N_DECADES = 41


class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic sinusoidal positional encoding for the diffusion timestep,
    followed by a small MLP. The output goes into the transformer as an
    extra "context" token.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class ConditionTokens(nn.Module):
    """
    Embeds the 4 condition tokens (day, month, leap, decade) into a sequence
    of 4 vectors of dim d_model. Used as a prefix to the transformer input
    so attention can flow from the date tokens to the conditions.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(N_DAYS,    d_model)
        self.month_emb  = nn.Embedding(N_MONTHS,  d_model)
        self.leap_emb   = nn.Embedding(N_LEAPS,   d_model)
        self.decade_emb = nn.Embedding(N_DECADES, d_model)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        # conditions: (B, 4) [day_idx, month_idx, leap_idx, decade_idx]
        day    = self.day_emb(conditions[:, 0]).unsqueeze(1)
        month  = self.month_emb(conditions[:, 1]).unsqueeze(1)
        leap   = self.leap_emb(conditions[:, 2]).unsqueeze(1)
        decade = self.decade_emb(conditions[:, 3]).unsqueeze(1)
        return torch.cat([day, month, leap, decade], dim=1)   # (B, 4, d_model)


def cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for bar_alpha (cumulative product of (1 - beta_t)).
    Output is a (T+1,) tensor where bar_alpha[0] = 1 and bar_alpha[T] is small.
    Used by both D3PM (as token survival probability) and Diffusion-LM
    (as signal scaling factor for x_0).
    """
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T + s) / (1 + s)) * (math.pi / 2)) ** 2
    return (f / f[0]).clamp(min=1e-5, max=1.0)
