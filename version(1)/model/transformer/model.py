"""
Seq2Seq Transformer for date generation.

Source sequence: the 4 condition tokens (length 4).
Target sequence: the date string character by character (max length 12 including BOS/EOS).

Each condition position has its own embedding table since they're from different
vocabularies (7 days, 12 months, 2 leap values, 41 decades).
The decoder is a standard causal transformer that generates one character at a time.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import (
    CHAR_VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID,
    MAX_SEQ_LEN, detokenize_date,
)


D_MODEL    = 128
N_HEADS    = 4
N_ENC_LAYERS = 3
N_DEC_LAYERS = 3
DIM_FF     = 256
DROPOUT    = 0.1

N_DAYS    = 7
N_MONTHS  = 12
N_LEAPS   = 2
N_DECADES = 41


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Nothing fancy - just the classic Vaswani et al. formula.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConditionEmbedding(nn.Module):
    """
    Embeds the 4 condition tokens into a sequence of length 4.
    Each condition position uses its own embedding table so the model
    knows which slot it's reading from.
    """

    def __init__(self, d_model: int = D_MODEL, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(N_DAYS,    d_model)
        self.month_emb  = nn.Embedding(N_MONTHS,  d_model)
        self.leap_emb   = nn.Embedding(N_LEAPS,   d_model)
        self.decade_emb = nn.Embedding(N_DECADES, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=8, dropout=dropout)
        self.scale      = math.sqrt(d_model)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        # conditions: (B, 4) [day_idx, month_idx, leap_idx, decade_idx]
        day    = self.day_emb(conditions[:, 0]).unsqueeze(1)    # (B, 1, d)
        month  = self.month_emb(conditions[:, 1]).unsqueeze(1)
        leap   = self.leap_emb(conditions[:, 2]).unsqueeze(1)
        decade = self.decade_emb(conditions[:, 3]).unsqueeze(1)
        # stack into (B, 4, d_model)
        seq = torch.cat([day, month, leap, decade], dim=1) * self.scale
        return self.pos_enc(seq)


class DateTransformer(nn.Module):
    """
    Encoder-decoder transformer.
    Encoder: encodes the 4 condition tokens.
    Decoder: autoregressively generates the date character sequence.
    """

    def __init__(
        self,
        d_model:      int = D_MODEL,
        nhead:        int = N_HEADS,
        n_enc_layers: int = N_ENC_LAYERS,
        n_dec_layers: int = N_DEC_LAYERS,
        dim_ff:       int = DIM_FF,
        dropout:      float = DROPOUT,
    ) -> None:
        super().__init__()

        self.cond_emb = ConditionEmbedding(d_model, dropout)

        # character embedding for the decoder side
        self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos_enc  = PositionalEncoding(d_model, max_len=MAX_SEQ_LEN + 4, dropout=dropout)
        self.scale    = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        self.output_proj = nn.Linear(d_model, CHAR_VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, conditions: torch.Tensor) -> torch.Tensor:
        src = self.cond_emb(conditions)   # (B, 4, d_model)
        return self.encoder(src)          # (B, 4, d_model)

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T   = tgt.size(1)
        emb = self.char_emb(tgt) * self.scale    # (B, T, d)
        emb = self.pos_enc(emb)

        # causal mask so each position can only see past positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)

        out = self.decoder(
            emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(out)   # (B, T, vocab_size)

    def forward(self, conditions: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        conditions: (B, 4)
        tgt:        (B, T) -- input to decoder, starts with BOS
        """
        pad_mask = (tgt == PAD_ID)
        memory   = self.encode(conditions)
        return self.decode(memory, tgt, tgt_key_padding_mask=pad_mask)

    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, max_len: int = MAX_SEQ_LEN) -> list:
        """
        Greedy decoding. Returns a list of date strings (one per batch element).
        """
        device = next(self.parameters()).device
        B      = conditions.size(0)
        memory = self.encode(conditions)

        # start every sequence with BOS
        tgt    = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        done   = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits     = self.decode(memory, tgt)       # (B, t, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            tgt        = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            done       = done | (next_token == EOS_ID)
            if done.all():
                break

        results = []
        for b in range(B):
            tokens   = tgt[b, 1:].tolist()   # skip BOS
            date_str = detokenize_date(tokens)
            results.append(date_str)
        return results
