"""
Dataset class for date generation.

I keep raw conditions and date strings accessible so we can use them
during validation (computing CSR requires the original condition dicts,
not just the encoded indices).
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import (
    parse_line,
    encode_conditions,
    encode_date_vec,
    tokenize_date,
    pad_sequence,
    MAX_SEQ_LEN,
)


class DateDataset(Dataset):
    """
    Loads data.txt and returns encoded (conditions, date) pairs.

    mode='vec'  ->  date is returned as (day_idx, year_last_digit) LongTensor of shape (2,)
    mode='seq'  ->  date is returned as a padded char-ID sequence of shape (MAX_SEQ_LEN,)
    """

    def __init__(self, data_path: str, mode: str = 'seq') -> None:
        assert mode in ('vec', 'seq'), f"mode must be 'vec' or 'seq', got {mode}"
        self.mode = mode

        # keep raw data around for CSR evaluation
        self.conditions: List[Dict[str, str]] = []
        self.date_strs:  List[str]            = []

        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cond, date_str = parse_line(line)
                if date_str is None:
                    continue
                self.conditions.append(cond)
                self.date_strs.append(date_str)

    def __len__(self) -> int:
        return len(self.date_strs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cond     = self.conditions[idx]
        date_str = self.date_strs[idx]

        day_idx, month_idx, leap_idx, decade_idx = encode_conditions(cond)
        cond_tensor = torch.tensor(
            [day_idx, month_idx, leap_idx, decade_idx], dtype=torch.long
        )

        if self.mode == 'seq':
            tokens     = tokenize_date(date_str)
            padded     = pad_sequence(tokens, MAX_SEQ_LEN)
            date_tensor = torch.tensor(padded, dtype=torch.long)
        else:
            day_comp, year_digit = encode_date_vec(date_str)
            date_tensor = torch.tensor([day_comp, year_digit], dtype=torch.long)

        return cond_tensor, date_tensor


def load_input_file(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Load example_input.txt (conditions only, no dates).
    Returns (list_of_condition_dicts, list_of_raw_lines).
    """
    conditions: List[Dict[str, str]] = []
    raw_lines:  List[str]            = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cond, _ = parse_line(line)
            conditions.append(cond)
            raw_lines.append(line)

    return conditions, raw_lines
