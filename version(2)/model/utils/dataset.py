"""
Dataset for the diffusion models.

Returns (cond_tensor, fixed_date_tensor) where:
  cond_tensor       : LongTensor of shape (4,)   [day, month, leap, decade]
  fixed_date_tensor : LongTensor of shape (10,)  -- DD-MM-YYYY char IDs
"""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import (
    parse_line, encode_conditions, tokenize_fixed, SEQ_LEN,
)


class DateDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        self.conditions: List[Dict[str, str]] = []
        self.date_strs:  List[str] = []

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
        date_tensor = torch.tensor(tokenize_fixed(date_str), dtype=torch.long)
        return cond_tensor, date_tensor


def load_input_file(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    conditions, raw = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cond, _ = parse_line(line)
            conditions.append(cond)
            raw.append(line)
    return conditions, raw
