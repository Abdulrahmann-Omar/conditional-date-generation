"""
D3PM with an absorbing-state forward process.

Paper: Austin et al. 2021, "Structured Denoising Diffusion Models in Discrete
State-Spaces" (https://arxiv.org/abs/2107.03006).

The forward process is the simplest of the three variants discussed in the
paper: at each step, with some probability, each token gets replaced by a
special [MASK] token. Once masked, a token stays masked (absorbing).

Concretely, with a cosine schedule of cumulative probabilities bar_alpha_t:
    q(x_t | x_0) = each token independently is original with prob bar_alpha_t,
                   else [MASK].

The reverse model predicts the distribution over the original tokens at
masked positions, conditioned on (x_t, t, conditions).

The architecture is a small Transformer encoder. Inputs are concatenated as:

    [cond_1, cond_2, cond_3, cond_4, time_token, x_t_1, ..., x_t_10]

The output for positions 5..14 (the date positions) is projected to a
distribution over the 11 non-mask tokens.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import VOCAB_SIZE, SEQ_LEN, MASK_ID, CHAR_VOCAB
from utils.common import SinusoidalTimeEmbedding, ConditionTokens, cosine_schedule


D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
DIM_FF     = 256
DROPOUT    = 0.1

# diffusion timesteps. Date sequences are short so we don't need 1000 steps.
DEFAULT_T = 100


class D3PMTransformer(nn.Module):
    """
    The reverse-process denoising network.
    Takes (x_t, t, conditions) and outputs logits over the original token
    vocabulary for every position. We never predict [MASK] as an output, so
    the head has dimension VOCAB_SIZE - 1 = 11.
    """

    def __init__(
        self,
        d_model:  int = D_MODEL,
        n_heads:  int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_ff:   int = DIM_FF,
        dropout:  float = DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.tok_emb  = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb  = nn.Embedding(SEQ_LEN + 5, d_model)
        # positions 0..3 are conditions, position 4 is the time token,
        # positions 5..14 are the date characters

        self.cond_tokens = ConditionTokens(d_model)
        self.time_emb    = SinusoidalTimeEmbedding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # head predicts non-mask tokens only (0..VOCAB_SIZE-2 = 0..10)
        self.head = nn.Linear(d_model, VOCAB_SIZE - 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        x_t:        (B, SEQ_LEN) token IDs
        t:          (B,) integer timesteps in [0, T]
        conditions: (B, 4)
        Returns logits of shape (B, SEQ_LEN, VOCAB_SIZE - 1)
        """
        B, L = x_t.shape
        device = x_t.device

        cond_seq = self.cond_tokens(conditions)              # (B, 4, d)
        time_tok = self.time_emb(t).unsqueeze(1)             # (B, 1, d)
        date_seq = self.tok_emb(x_t)                         # (B, L, d)

        seq = torch.cat([cond_seq, time_tok, date_seq], dim=1)   # (B, 5+L, d)

        positions = torch.arange(seq.size(1), device=device)
        seq = seq + self.pos_emb(positions)[None]

        out = self.transformer(seq)
        out = self.norm(out)
        date_out = out[:, 5:]    # only the date positions
        return self.head(date_out)


# ---------------- forward process ----------------

def q_sample(
    x_0: torch.Tensor, t: torch.Tensor, bar_alpha: torch.Tensor
) -> torch.Tensor:
    """
    Absorbing forward process: each token is kept with probability
    bar_alpha[t], else replaced with [MASK].
    """
    bar_alpha_t = bar_alpha[t][:, None]           # (B, 1)
    survive_mask = torch.rand_like(x_0, dtype=torch.float32) < bar_alpha_t
    x_t = torch.where(survive_mask, x_0, torch.full_like(x_0, MASK_ID))
    return x_t


def d3pm_loss(
    model: D3PMTransformer,
    x_0: torch.Tensor,
    conditions: torch.Tensor,
    bar_alpha: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    Cross-entropy loss on the model's prediction of x_0, computed only at
    the positions that were corrupted (masked) at the sampled timestep.
    This is the variational simple loss for the absorbing-state D3PM.
    """
    B = x_0.size(0)
    t = torch.randint(1, T + 1, (B,), device=x_0.device)
    x_t = q_sample(x_0, t, bar_alpha)

    logits = model(x_t, t, conditions)            # (B, L, V-1)
    was_masked = (x_t == MASK_ID)

    # only count loss at masked positions
    if was_masked.any():
        loss = F.cross_entropy(
            logits[was_masked].reshape(-1, logits.size(-1)),
            x_0[was_masked].reshape(-1),
        )
    else:
        loss = torch.zeros((), device=x_0.device)
    return loss


# ---------------- reverse / generation ----------------

@torch.no_grad()
def d3pm_sample(
    model: D3PMTransformer,
    conditions: torch.Tensor,
    bar_alpha: torch.Tensor,
    T: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Iteratively denoise from an all-masked sequence.

    At each reverse step t -> t-1, for every position still masked in x_t:
      - predict x_0 from the model
      - with probability (bar_alpha[t-1] - bar_alpha[t]) / (1 - bar_alpha[t])
        unmask that position using a sample from the predicted distribution
    At t = 1 -> 0 we force-unmask any positions still left.
    """
    device = conditions.device
    B = conditions.size(0)
    x_t = torch.full((B, SEQ_LEN), MASK_ID, dtype=torch.long, device=device)

    for t in range(T, 0, -1):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        logits  = model(x_t, t_batch, conditions)            # (B, L, V-1)

        if temperature != 1.0:
            logits = logits / temperature

        # sample x_0 predictions at every position
        probs = F.softmax(logits, dim=-1)
        x_0_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
        x_0_pred = x_0_pred.reshape(B, SEQ_LEN)

        masked = (x_t == MASK_ID)

        bar_t   = bar_alpha[t].item()
        bar_tm1 = bar_alpha[t - 1].item()
        # probability of unmasking each currently-masked token at this step
        if t > 1:
            p_unmask = max(0.0, (bar_tm1 - bar_t) / max(1.0 - bar_t, 1e-8))
            unmask_now = (torch.rand(B, SEQ_LEN, device=device) < p_unmask) & masked
        else:
            unmask_now = masked   # force final fill at t=1

        x_t = torch.where(unmask_now, x_0_pred, x_t)

    return x_t


def build_schedule(T: int = DEFAULT_T) -> torch.Tensor:
    """Convenience wrapper for the cosine schedule used by D3PM."""
    return cosine_schedule(T)
