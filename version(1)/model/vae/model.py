"""
Conditional VAE for date generation.

The encoder takes (condition embedding || date one-hot) and maps it to a
latent distribution (mu, log_var). The decoder takes (condition embedding || z)
and reconstructs the date as logits over (day, year_last_digit).

At inference we sample z ~ N(0, I) so the model generates different valid dates
each time it's called with the same conditions.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import N_DAY_VALS, N_YEAR_DIGITS, DATE_VEC_DIM


LATENT_DIM: int = 32
COND_DIM:   int = 64

N_DAYS    = 7
N_MONTHS  = 12
N_LEAPS   = 2
N_DECADES = 41


class ConditionEncoder(nn.Module):
    """Same condition encoder used across GAN and VAE so the embedding style is consistent."""

    def __init__(self, cond_dim: int = COND_DIM) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(N_DAYS,    16)
        self.month_emb  = nn.Embedding(N_MONTHS,  16)
        self.leap_emb   = nn.Embedding(N_LEAPS,    4)
        self.decade_emb = nn.Embedding(N_DECADES, 32)
        self.proj = nn.Sequential(
            nn.Linear(16 + 16 + 4 + 32, cond_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
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
        return self.proj(emb)


class Encoder(nn.Module):
    """
    Maps (cond || date_one_hot) to the latent distribution parameters.
    Input dim: cond_dim + DATE_VEC_DIM = 64 + 41 = 105
    """

    def __init__(self, cond_dim: int = COND_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        in_dim = cond_dim + DATE_VEC_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, cond: torch.Tensor, date_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x      = torch.cat([cond, date_vec], dim=-1)
        h      = self.net(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Maps (cond || z) back to date logits.
    Outputs separate logits for day (31) and year_last_digit (10).
    """

    def __init__(self, cond_dim: int = COND_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        in_dim = cond_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, DATE_VEC_DIM),
        )

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cond, z], dim=-1)
        return self.net(x)   # (B, 41) raw logits


class ConditionalVAE(nn.Module):

    def __init__(self, cond_dim: int = COND_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.cond_encoder = ConditionEncoder(cond_dim)
        self.encoder      = Encoder(cond_dim, latent_dim)
        self.decoder      = Decoder(cond_dim, latent_dim)
        self.latent_dim   = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, conditions: torch.Tensor, date_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits, mu, logvar).
        logits has shape (B, 41).
        """
        cond   = self.cond_encoder(conditions)
        mu, logvar = self.encoder(cond, date_vec)
        z      = self.reparameterize(mu, logvar)
        logits = self.decoder(cond, z)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample new dates for the given conditions."""
        device = next(self.parameters()).device
        cond   = self.cond_encoder(conditions)
        z      = torch.randn(conditions.size(0), self.latent_dim, device=device)
        logits = self.decoder(cond, z)
        day_idx    = logits[:, :N_DAY_VALS].argmax(dim=-1)
        year_digit = logits[:, N_DAY_VALS:].argmax(dim=-1)
        return day_idx, year_digit


def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = reconstruction loss + beta * KL divergence.
    targets: (B, 2) [day_idx, year_digit]
    logits:  (B, 41)
    """
    # cross-entropy for each output group separately
    day_loss  = F.cross_entropy(logits[:, :N_DAY_VALS], targets[:, 0])
    year_loss = F.cross_entropy(logits[:, N_DAY_VALS:], targets[:, 1])
    recon     = day_loss + year_loss

    # standard KL term
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + beta * kld, recon, kld
