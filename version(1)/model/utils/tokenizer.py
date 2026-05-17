"""
Tokenizer for conditions and dates.

The input conditions are four bracket-wrapped tokens like [MON] [DEC] [False] [196].
The output is a date string like 3-12-1962 (no zero-padding on day or month).

I'm keeping two parallel encodings:
  - "vec" mode: encode the date as (day_idx, year_last_digit) for GAN/VAE
  - "seq" mode: encode the date as a character sequence for Transformer/LSTM
"""

from typing import Dict, List, Optional, Tuple
import datetime


DAY_TOKENS: List[str] = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

MONTH_TOKENS: List[str] = [
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
]

# decade token is the first 3 digits of the year, e.g. [196] -> 1960-1969
DECADE_MIN: int = 180
DECADE_MAX: int = 220
NUM_DECADES: int = DECADE_MAX - DECADE_MIN + 1   # 41 total decades

# how many days each month can have at most (Feb gets 29 as an upper bound)
MAX_DAYS_PER_MONTH: List[int] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# character vocabulary for the sequence models
# digits 0-9, separator '-', then three special tokens
DATE_CHARS: List[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
SPECIAL_TOKENS: List[str] = ['<PAD>', '<BOS>', '<EOS>']
CHAR_VOCAB: List[str] = DATE_CHARS + SPECIAL_TOKENS
CHAR_VOCAB_SIZE: int = len(CHAR_VOCAB)   # 14

PAD_ID: int = CHAR_VOCAB.index('<PAD>')
BOS_ID: int = CHAR_VOCAB.index('<BOS>')
EOS_ID: int = CHAR_VOCAB.index('<EOS>')

# longest possible date string is "31-12-2200" = 10 chars; add 2 for BOS/EOS
MAX_DATE_LEN: int = 10
MAX_SEQ_LEN: int = MAX_DATE_LEN + 2

# for GAN/VAE vector encoding
N_DAY_VALS: int = 31     # days 1-31 -> indices 0-30
N_YEAR_DIGITS: int = 10  # last digit of year: 0-9
DATE_VEC_DIM: int = N_DAY_VALS + N_YEAR_DIGITS   # 41


# ------- parsing -------

def parse_line(line: str) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Split one line into a condition dict and an optional date string.
    Works for both data.txt (has date) and example_input.txt (no date).
    """
    parts = line.strip().split()
    conditions = {
        'day':    parts[0][1:-1],   # strip [ and ]
        'month':  parts[1][1:-1],
        'leap':   parts[2][1:-1],   # 'True' or 'False'
        'decade': parts[3][1:-1],   # e.g. '196'
    }
    date_str = parts[4] if len(parts) > 4 else None
    return conditions, date_str


def format_line(conditions: Dict[str, str], date_str: str) -> str:
    """Reconstruct the original line format from conditions + date."""
    return (
        f"[{conditions['day']}] [{conditions['month']}] "
        f"[{conditions['leap']}] [{conditions['decade']}] {date_str}"
    )


# ------- condition encoding -------

def encode_conditions(cond: Dict[str, str]) -> Tuple[int, int, int, int]:
    """Map condition strings to integer indices."""
    day_idx    = DAY_TOKENS.index(cond['day'])
    month_idx  = MONTH_TOKENS.index(cond['month'])
    leap_idx   = 1 if cond['leap'] == 'True' else 0
    decade_idx = int(cond['decade']) - DECADE_MIN
    return day_idx, month_idx, leap_idx, decade_idx


# ------- date encoding for vec models (GAN/VAE) -------

def encode_date_vec(date_str: str) -> Tuple[int, int]:
    """
    Encode a date as (day_idx, year_last_digit).
    Month and decade are dropped here since they come directly from the conditions.
    day_idx = day - 1, so day=1 -> idx=0
    """
    d, m, y = date_str.split('-')
    day_idx        = int(d) - 1
    year_last_digit = int(y) % 10
    return day_idx, year_last_digit


def decode_date_vec(day_idx: int, month_idx: int, decade_idx: int, year_last_digit: int) -> str:
    """Reconstruct a date string from the four components."""
    day          = day_idx + 1
    month        = month_idx + 1
    decade_value = decade_idx + DECADE_MIN
    year         = decade_value * 10 + year_last_digit
    return f"{day}-{month}-{year}"


# ------- date encoding for seq models (Transformer/LSTM) -------

def tokenize_date(date_str: str) -> List[int]:
    """Tokenize a date string to a list of char IDs, wrapped in BOS/EOS."""
    return [BOS_ID] + [CHAR_VOCAB.index(c) for c in date_str] + [EOS_ID]


def detokenize_date(token_ids: List[int]) -> str:
    """Convert char IDs back to a date string (stops at EOS or PAD)."""
    chars = []
    for tid in token_ids:
        if tid in (EOS_ID, PAD_ID, BOS_ID):
            if tid == EOS_ID:
                break
            continue
        chars.append(CHAR_VOCAB[tid])
    return ''.join(chars)


def pad_sequence(tokens: List[int], max_len: int) -> List[int]:
    return tokens + [PAD_ID] * (max_len - len(tokens))


# ------- calendar helpers -------

def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def max_days_in_month(month: int, year: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return MAX_DAYS_PER_MONTH[month - 1]
