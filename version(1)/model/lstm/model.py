"""
Seq2Seq LSTM for date generation.

The encoder is just an embedding layer that maps the 4 conditions to a single
context vector (by concatenation + linear projection). This context vector
initializes the decoder LSTM hidden state.

The decoder is an LSTM that generates the date character by character.
At each step it takes the previous character embedding as input and outputs
logits over the character vocabulary.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import (
    CHAR_VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID,
    MAX_SEQ_LEN, detokenize_date,
)


HIDDEN_SIZE = 256
N_LAYERS    = 2
EMBED_DIM   = 64    # character embedding dim
COND_DIM    = 128   # condition embedding dim

N_DAYS    = 7
N_MONTHS  = 12
N_LEAPS   = 2
N_DECADES = 41


class ConditionToHidden(nn.Module):
    """
    Takes the 4 condition indices and produces the initial (h0, c0) for
    each decoder LSTM layer.

    The mapping is: embed each condition, concat, then project to
    (N_LAYERS * HIDDEN_SIZE * 2) to fill all layer/cell pairs.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        n_layers:    int = N_LAYERS,
        cond_dim:    int = COND_DIM,
    ) -> None:
        super().__init__()
        self.n_layers    = n_layers
        self.hidden_size = hidden_size

        self.day_emb    = nn.Embedding(N_DAYS,    16)
        self.month_emb  = nn.Embedding(N_MONTHS,  16)
        self.leap_emb   = nn.Embedding(N_LEAPS,    4)
        self.decade_emb = nn.Embedding(N_DECADES, 32)

        self.proj = nn.Sequential(
            nn.Linear(16 + 16 + 4 + 32, cond_dim),
            nn.Tanh(),
            # output: n_layers * hidden_size * 2  (for h and c of each layer)
            nn.Linear(cond_dim, n_layers * hidden_size * 2),
        )

    def forward(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # conditions: (B, 4)
        emb = torch.cat([
            self.day_emb(conditions[:, 0]),
            self.month_emb(conditions[:, 1]),
            self.leap_emb(conditions[:, 2]),
            self.decade_emb(conditions[:, 3]),
        ], dim=-1)   # (B, 68)

        out = self.proj(emb)  # (B, n_layers * hidden_size * 2)

        # split into h0 and c0, each (n_layers, B, hidden_size)
        h_and_c = out.view(-1, self.n_layers, 2, self.hidden_size)
        h0 = h_and_c[:, :, 0, :].permute(1, 0, 2).contiguous()
        c0 = h_and_c[:, :, 1, :].permute(1, 0, 2).contiguous()
        return h0, c0


class DateLSTM(nn.Module):

    def __init__(
        self,
        hidden_size: int   = HIDDEN_SIZE,
        n_layers:    int   = N_LAYERS,
        embed_dim:   int   = EMBED_DIM,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        self.cond_encoder = ConditionToHidden(hidden_size, n_layers)
        self.char_emb     = nn.Embedding(CHAR_VOCAB_SIZE, embed_dim, padding_idx=PAD_ID)
        self.lstm         = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, CHAR_VOCAB_SIZE)

    def forward(
        self,
        conditions: torch.Tensor,
        tgt:        torch.Tensor,
    ) -> torch.Tensor:
        """
        conditions: (B, 4)
        tgt:        (B, T)  -- decoder input starting with BOS
        Returns logits (B, T, vocab_size).
        """
        h0, c0  = self.cond_encoder(conditions)
        emb     = self.char_emb(tgt)            # (B, T, embed_dim)
        out, _  = self.lstm(emb, (h0, c0))      # (B, T, hidden_size)
        return self.output_proj(out)             # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, max_len: int = MAX_SEQ_LEN) -> List[str]:
        """Greedy autoregressive decoding."""
        device = next(self.parameters()).device
        B      = conditions.size(0)

        h, c   = self.cond_encoder(conditions)
        token  = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        done   = torch.zeros(B, dtype=torch.bool, device=device)

        all_tokens = []   # will accumulate generated tokens (excluding BOS)

        for _ in range(max_len):
            emb         = self.char_emb(token)            # (B, 1, embed)
            out, (h, c) = self.lstm(emb, (h, c))          # (B, 1, hidden)
            logits      = self.output_proj(out[:, 0, :])  # (B, vocab)
            next_tok    = logits.argmax(dim=-1)            # (B,)

            done = done | (next_tok == EOS_ID)
            all_tokens.append(next_tok)

            token = next_tok.unsqueeze(1)
            if done.all():
                break

        # reconstruct date strings
        results = []
        for b in range(B):
            tokens   = [t[b].item() for t in all_tokens]
            date_str = detokenize_date(tokens)
            results.append(date_str)
        return results
