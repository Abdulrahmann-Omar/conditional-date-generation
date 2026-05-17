"""
Conditional GAN for date generation.

The generator takes a noise vector + condition embeddings and outputs
soft distributions over (day, year_last_digit). The discriminator sees
a date vector (one-hot or soft) alongside the same conditions and tries
to distinguish real from fake.

I'm using label smoothing on the real side to avoid overconfident D early on,
and training G twice per D step since the date space is small and D learns fast.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import N_DAY_VALS, N_YEAR_DIGITS, DATE_VEC_DIM


NOISE_DIM: int = 64
COND_DIM:  int = 64

# vocabulary sizes for each condition
N_DAYS    = 7
N_MONTHS  = 12
N_LEAPS   = 2
N_DECADES = 41


class ConditionEncoder(nn.Module):
    """
    Embeds the four condition tokens and projects to a fixed-size vector.
    Each condition has its own embedding table since they come from very different
    distributions (7 days vs 41 decades, for example).
    """

    def __init__(self, cond_dim: int = COND_DIM) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(N_DAYS,    16)
        self.month_emb  = nn.Embedding(N_MONTHS,  16)
        self.leap_emb   = nn.Embedding(N_LEAPS,    4)
        self.decade_emb = nn.Embedding(N_DECADES, 32)
        self.proj = nn.Sequential(
            nn.Linear(16 + 16 + 4 + 32, cond_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        # conditions: (B, 4) with [day_idx, month_idx, leap_idx, decade_idx]
        day, month, leap, decade = (
            conditions[:, 0], conditions[:, 1],
            conditions[:, 2], conditions[:, 3],
        )
        emb = torch.cat([
            self.day_emb(day),
            self.month_emb(month),
            self.leap_emb(leap),
            self.decade_emb(decade),
        ], dim=-1)
        return self.proj(emb)  # (B, cond_dim)


class Generator(nn.Module):
    """
    G: noise(64) + condition(64) -> date logits(41)

    The 41 logits are split into day_logits(31) and year_digit_logits(10).
    During training the generator outputs soft distributions (post-softmax).
    At inference we take argmax.
    """

    def __init__(self, noise_dim: int = NOISE_DIM, cond_dim: int = COND_DIM) -> None:
        super().__init__()
        self.noise_dim    = noise_dim
        self.cond_encoder = ConditionEncoder(cond_dim)

        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, DATE_VEC_DIM),
        )

    def forward(self, noise: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        cond = self.cond_encoder(conditions)
        x    = torch.cat([noise, cond], dim=-1)
        return self.net(x)  # raw logits (B, 41)

    def soft_date(self, noise: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Apply per-group softmax so D sees something that looks like one-hots."""
        logits     = self.forward(noise, conditions)
        day_soft   = F.softmax(logits[:, :N_DAY_VALS],     dim=-1)
        year_soft  = F.softmax(logits[:, N_DAY_VALS:],     dim=-1)
        return torch.cat([day_soft, year_soft], dim=-1)

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate day_idx and year_last_digit from conditions.
        Returns two (B,) tensors.
        """
        device = next(self.parameters()).device
        noise  = torch.randn(conditions.size(0), self.noise_dim, device=device)
        logits = self.forward(noise, conditions)
        day_idx    = logits[:, :N_DAY_VALS].argmax(dim=-1)
        year_digit = logits[:, N_DAY_VALS:].argmax(dim=-1)
        return day_idx, year_digit


class Discriminator(nn.Module):
    """
    D: date_vec(41) + condition(64) -> scalar logit

    Uses dropout to slow down D a bit so G has room to learn.
    """

    def __init__(self, cond_dim: int = COND_DIM) -> None:
        super().__init__()
        self.cond_encoder = ConditionEncoder(cond_dim)

        self.net = nn.Sequential(
            nn.Linear(DATE_VEC_DIM + cond_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, date_vec: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        cond = self.cond_encoder(conditions)
        x    = torch.cat([date_vec, cond], dim=-1)
        return self.net(x)  # (B, 1)
