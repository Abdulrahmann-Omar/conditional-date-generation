"""
Evaluation utilities.

Since there are many valid dates per condition set, accuracy against the ground truth
doesn't make much sense. Instead I use the Condition Satisfaction Rate (CSR):
what percentage of generated dates actually satisfy all four conditions.
"""

import datetime
from typing import Dict, List, Tuple

from .tokenizer import MONTH_TOKENS, DAY_TOKENS, is_leap_year, max_days_in_month, DECADE_MIN


# datetime.weekday() returns 0=Monday, 6=Sunday
_WEEKDAY_MAP: Dict[int, str] = {
    0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'
}


def get_weekday(day: int, month: int, year: int) -> str:
    """Return the three-letter weekday token for a given date."""
    try:
        return _WEEKDAY_MAP[datetime.date(year, month, day).weekday()]
    except ValueError:
        return 'INVALID'


def check_date(date_str: str, cond: Dict[str, str]) -> Dict[str, bool]:
    """
    Check each condition individually.
    Returns a dict with keys: valid, day, month, leap, decade, all.
    """
    result = {k: False for k in ('valid', 'day', 'month', 'leap', 'decade', 'all')}

    try:
        parts = date_str.strip().split('-')
        if len(parts) != 3:
            return result

        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])

        # basic sanity
        if not (1 <= month <= 12):
            return result
        if not (1 <= day <= max_days_in_month(month, year)):
            return result
        if not (1800 <= year <= 2200):
            return result

        result['valid'] = True

        result['month']  = (MONTH_TOKENS[month - 1] == cond['month'])
        result['decade'] = (str(year // 10) == cond['decade'])
        result['leap']   = (is_leap_year(year) == (cond['leap'] == 'True'))
        result['day']    = (get_weekday(day, month, year) == cond['day'])
        result['all']    = all(result[k] for k in ('valid', 'day', 'month', 'leap', 'decade'))

    except (ValueError, IndexError):
        pass

    return result


def compute_csr(
    predictions: List[str],
    conditions: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Compute Condition Satisfaction Rate across a list of predictions.
    Returns fractions (0.0 to 1.0) for each condition and for 'all'.
    """
    n = len(predictions)
    if n == 0:
        return {}

    totals: Dict[str, int] = {k: 0 for k in ('valid', 'day', 'month', 'leap', 'decade', 'all')}

    for pred, cond in zip(predictions, conditions):
        checks = check_date(pred, cond)
        for k in totals:
            if checks.get(k, False):
                totals[k] += 1

    return {k: v / n for k, v in totals.items()}


def print_csr(csr: Dict[str, float], prefix: str = '') -> None:
    """Print a readable CSR summary."""
    parts = [f"{k}={v:.3f}" for k, v in csr.items()]
    print(f"{prefix}CSR: {' | '.join(parts)}")
