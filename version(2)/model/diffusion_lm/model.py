"""
Diffusion-LM: continuous diffusion on token embeddings.

Paper: Li et al. 2022, "Diffusion-LM Improves Controllable Text Generation"
(https://arxiv.org/abs/2205.14217).

Each discrete token is embedded into a small continuous vector. We then run
a standard DDPM forward/reverse process in that embedding space. At
inference time, after we've reconstructed the continuous x_0, we "round" by
picking the token whose embedding is closest.

Two key training signals:
  - L_simple: MSE between predicted x_0 and the ground-truth embedding
  - L_round:  cross-entropy on rounding x_0 back to the original token
              (uses the same embedding table as the input embedding layer)

Both losses use the SHARED embedding so the geometry stays consistent
between the input encoder and the output rounder. This is critical for the
model to produce x_0 vectors that actually round to the right token.

We don't use the asymmetric Markov transition stuff from the paper; we just
use the standard x_0-prediction parameterisation with a cosine schedule.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import VOCAB_SIZE, SEQ_LEN, MASK_ID, DASH_ID
from utils.common import SinusoidalTimeEmbedding, ConditionTokens, cosine_schedule


D_MODEL    = 128
D_EMBED    = 32    # embedding dim of each token (the latent we diffuse in)
N_HEADS    = 4
N_LAYERS   = 4
DIM_FF     = 256
DROPOUT    = 0.1

DEFAULT_T = 200    # more steps than D3PM because continuous diffusion benefits from it


class DiffusionLMTransformer(nn.Module):
    """
    Predicts x_0 (continuous embedding) given x_t, t, and conditions.

    Architecture:
      - input projection from D_EMBED to D_MODEL
      - prepend [cond_1..4, time_token]
      - transformer encoder
      - output projection back to D_EMBED for the date positions
    """

    def __init__(
        self,
        d_model:  int = D_MODEL,
        d_embed:  int = D_EMBED,
        n_heads:  int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_ff:   int = DIM_FF,
        dropout:  float = DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed

        # learnable token embedding. This is also used for "rounding" at the
        # end of generation, so it has to live in the same space as the
        # things being diffused.
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_embed)

        self.in_proj  = nn.Linear(d_embed, d_model)
        self.out_proj = nn.Linear(d_model, d_embed)

        self.pos_emb = nn.Embedding(SEQ_LEN + 5, d_model)
        self.cond_tokens = ConditionTokens(d_model)
        self.time_emb    = SinusoidalTimeEmbedding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings. Used to make x_0 from a target sequence."""
        return self.tok_emb(x)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        x_t:        (B, L, d_embed) - noisy embeddings
        t:          (B,)
        conditions: (B, 4)
        Returns predicted x_0 of the same shape as x_t.
        """
        B, L, _ = x_t.shape
        device = x_t.device

        x_proj  = self.in_proj(x_t)                             # (B, L, d_model)

        cond_seq = self.cond_tokens(conditions)                 # (B, 4, d_model)
        time_tok = self.time_emb(t).unsqueeze(1)                # (B, 1, d_model)

        seq = torch.cat([cond_seq, time_tok, x_proj], dim=1)    # (B, 5+L, d_model)
        positions = torch.arange(seq.size(1), device=device)
        seq = seq + self.pos_emb(positions)[None]

        out = self.transformer(seq)
        out = self.norm(out)
        date_out = out[:, 5:]                                   # (B, L, d_model)
        return self.out_proj(date_out)                          # (B, L, d_embed)

    def round_to_tokens(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Pick the token whose embedding is nearest each predicted x_0 vector.
        We exclude MASK from the candidate set (we never want to emit MASK).
        Returns LongTensor of shape (B, L).
        """
        # candidate embeddings: every token except MASK
        emb_table = self.tok_emb.weight        # (V, d_embed)
        candidates = torch.arange(VOCAB_SIZE, device=x_0.device)
        keep = candidates != MASK_ID
        valid_emb = emb_table[keep]             # (V-1, d_embed)
        valid_ids = candidates[keep]            # (V-1,)

        # distance from each x_0 vector to each candidate embedding
        # x_0: (B, L, d), valid_emb: (V-1, d)
        dist = torch.cdist(x_0, valid_emb[None].expand(x_0.size(0), -1, -1))
        nearest = dist.argmin(dim=-1)           # (B, L)
        return valid_ids[nearest]               # map back to original token IDs


# ---------------- forward process ----------------

def q_sample_continuous(
    x_0: torch.Tensor, t: torch.Tensor, bar_alpha: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard DDPM forward in continuous space:
       x_t = sqrt(bar_alpha[t]) * x_0 + sqrt(1 - bar_alpha[t]) * eps
    Returns (x_t, eps).
    """
    bar_alpha_t = bar_alpha[t][:, None, None]   # (B, 1, 1)
    sqrt_bar    = bar_alpha_t.sqrt()
    sqrt_one_m  = (1.0 - bar_alpha_t).sqrt()
    eps = torch.randn_like(x_0)
    x_t = sqrt_bar * x_0 + sqrt_one_m * eps
    return x_t, eps


def diffusion_lm_loss(
    model:      DiffusionLMTransformer,
    target_ids: torch.Tensor,
    conditions: torch.Tensor,
    bar_alpha:  torch.Tensor,
    T:          int,
    round_weight: float = 1.0,
) -> torch.Tensor:
    """
    L_simple (MSE on x_0 prediction) + L_round (CE on rounding).
    L_round uses cosine-similarity-based logits so the loss is consistent
    with the L2 nearest-neighbour rounding used at inference.
    """
    B = target_ids.size(0)
    t = torch.randint(1, T + 1, (B,), device=target_ids.device)

    x_0_target = model.embed_tokens(target_ids)                   # (B, L, d)
    x_t, _ = q_sample_continuous(x_0_target, t, bar_alpha)
    x_0_pred = model(x_t, t, conditions)                          # (B, L, d)

    # L_simple
    l_simple = F.mse_loss(x_0_pred, x_0_target)

    # L_round: distance-based logits to every token, cross-entropy to target
    emb_table = model.tok_emb.weight                               # (V, d)
    # squared distance to each token embedding
    sq_dist = ((x_0_pred[:, :, None, :] - emb_table[None, None]) ** 2).sum(-1)  # (B, L, V)
    # mask out MASK token with an additive bias (avoid in-place ops for autograd)
    mask_bias = torch.zeros(emb_table.size(0), device=x_0_pred.device)
    mask_bias[MASK_ID] = 1e6
    logits  = -sq_dist - mask_bias[None, None, :]
    l_round = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
    )

    return l_simple + round_weight * l_round


# ---------------- reverse / sampling ----------------

@torch.no_grad()
def diffusion_lm_sample(
    model:      DiffusionLMTransformer,
    conditions: torch.Tensor,
    bar_alpha:  torch.Tensor,
    T:          int,
) -> torch.Tensor:
    """
    DDPM ancestral sampling using x_0 prediction.

    x_{t-1} | x_t, x_0_pred has mean
        mu = sqrt(bar_alpha_{t-1}) * (1 - alpha_t) / (1 - bar_alpha_t)  * x_0_pred
           + sqrt(alpha_t) * (1 - bar_alpha_{t-1}) / (1 - bar_alpha_t)  * x_t
    where alpha_t = bar_alpha_t / bar_alpha_{t-1}.
    """
    device = conditions.device
    B = conditions.size(0)
    L = SEQ_LEN
    d = model.d_embed

    x_t = torch.randn(B, L, d, device=device)

    for t in range(T, 0, -1):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        x_0_pred = model(x_t, t_batch, conditions)

        bar_t   = bar_alpha[t].item()
        bar_tm1 = bar_alpha[t - 1].item() if t > 1 else 1.0
        alpha_t = bar_t / max(bar_tm1, 1e-8)

        if t > 1:
            mean = (
                (bar_tm1 ** 0.5) * (1 - alpha_t) / max(1 - bar_t, 1e-8) * x_0_pred
                + (alpha_t ** 0.5) * (1 - bar_tm1) / max(1 - bar_t, 1e-8) * x_t
            )
            var = max((1 - alpha_t) * (1 - bar_tm1) / max(1 - bar_t, 1e-8), 1e-8)
            noise = torch.randn_like(x_t)
            x_t = mean + (var ** 0.5) * noise
        else:
            x_t = x_0_pred

    # round to tokens
    return model.round_to_tokens(x_t)


def build_schedule(T: int = DEFAULT_T) -> torch.Tensor:
    return cosine_schedule(T)
