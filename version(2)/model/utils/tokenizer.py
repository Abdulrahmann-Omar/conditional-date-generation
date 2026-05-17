"""
Tokenizer for the diffusion models in version 2.

The main difference from version 1 is that we use a FIXED-LENGTH encoding
of the date string. Diffusion models (especially D3PM) are much easier to
implement when every example has the same length. So we zero-pad day and
month to two digits, year is always four digits, and the format becomes:

    DD-MM-YYYY    (length 10)

For example:
    3-12-1962  ->  03-12-1962
    1-1-1800   ->  01-01-1800
    31-12-2200 ->  31-12-2200

For D3PM we also need a [MASK] token. The vocabulary becomes:

    digits 0-9, '-', [MASK]    -> 12 tokens
"""

from typing import Dict, List, Optional, Tuple
import datetime


DAY_TOKENS: List[str] = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

MONTH_TOKENS: List[str] = [
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
]

DECADE_MIN: int = 180
DECADE_MAX: int = 220
NUM_DECADES: int = DECADE_MAX - DECADE_MIN + 1   # 41

MAX_DAYS_PER_MONTH: List[int] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# -------------------- fixed-length char vocab --------------------

DATE_CHARS: List[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
MASK_TOKEN: str = '[MASK]'
CHAR_VOCAB: List[str] = DATE_CHARS + [MASK_TOKEN]
VOCAB_SIZE: int = len(CHAR_VOCAB)   # 12

MASK_ID: int = CHAR_VOCAB.index(MASK_TOKEN)
DASH_ID: int = CHAR_VOCAB.index('-')

# fixed-length date string: DD-MM-YYYY
SEQ_LEN: int = 10


# -------------------- parsing & conditions --------------------

def parse_line(line: str) -> Tuple[Dict[str, str], Optional[str]]:
    parts = line.strip().split()
    cond = {
        'day':    parts[0][1:-1],
        'month':  parts[1][1:-1],
        'leap':   parts[2][1:-1],
        'decade': parts[3][1:-1],
    }
    date_str = parts[4] if len(parts) > 4 else None
    return cond, date_str


def format_line(cond: Dict[str, str], date_str: str) -> str:
    return (
        f"[{cond['day']}] [{cond['month']}] "
        f"[{cond['leap']}] [{cond['decade']}] {date_str}"
    )


def encode_conditions(cond: Dict[str, str]) -> Tuple[int, int, int, int]:
    day_idx    = DAY_TOKENS.index(cond['day'])
    month_idx  = MONTH_TOKENS.index(cond['month'])
    leap_idx   = 1 if cond['leap'] == 'True' else 0
    decade_idx = int(cond['decade']) - DECADE_MIN
    return day_idx, month_idx, leap_idx, decade_idx


# -------------------- fixed-length date encoding --------------------

def to_fixed_str(date_str: str) -> str:
    """Convert d-m-yyyy to DD-MM-YYYY with zero padding."""
    d, m, y = date_str.split('-')
    return f"{int(d):02d}-{int(m):02d}-{int(y):04d}"


def from_fixed_str(fixed_str: str) -> str:
    """Convert DD-MM-YYYY back to d-m-yyyy (strip leading zeros)."""
    try:
        d, m, y = fixed_str.split('-')
        return f"{int(d)}-{int(m)}-{int(y)}"
    except (ValueError, IndexError):
        return fixed_str   # if malformed, return as-is so caller can validate


def tokenize_fixed(date_str: str) -> List[int]:
    """Tokenize a d-m-yyyy date as 10 char IDs (zero-padded internally)."""
    fixed = to_fixed_str(date_str)
    return [CHAR_VOCAB.index(c) for c in fixed]


def detokenize_fixed(token_ids: List[int]) -> str:
    """Convert 10 char IDs back to a d-m-yyyy string. MASK tokens become '?'."""
    chars = []
    for tid in token_ids:
        if 0 <= tid < len(CHAR_VOCAB):
            c = CHAR_VOCAB[tid]
            if c == MASK_TOKEN:
                chars.append('?')
            else:
                chars.append(c)
        else:
            chars.append('?')
    fixed_str = ''.join(chars)
    return from_fixed_str(fixed_str)


# -------------------- calendar helpers --------------------

def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def max_days_in_month(month: int, year: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return MAX_DAYS_PER_MONTH[month - 1]
