"""
predict.py - run inference with any of the four trained models.

Usage:
    python predict.py -i path/to/input.txt -o path/to/output.txt
    python predict.py -i path/to/input.txt -o path/to/output.txt --model transformer

Available models: transformer (default), lstm, gan, vae

The output format matches data.txt exactly:
    [DAY] [MONTH] [LEAP] [DECADE] dd-mm-yyyy
in the same order as the input file.

After generation, each date is validated against its conditions. If a date
is invalid (e.g. day 31 in April) the script falls back to a simple
calendar-search to find the nearest valid date for those conditions.
"""

import argparse
import os
import sys
import datetime
from typing import Dict, List, Tuple

import torch

# all model code lives alongside this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.tokenizer import (
    encode_conditions, format_line,
    MONTH_TOKENS, DAY_TOKENS, DECADE_MIN,
    is_leap_year, max_days_in_month,
)
from utils.dataset  import load_input_file
from utils.evaluate import check_date, compute_csr, print_csr


# ---- lazy model imports (only load the one we need) ----

def load_transformer(weights_path: str) -> 'DateTransformer':
    from transformer.model import DateTransformer
    model = DateTransformer()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def load_lstm(weights_path: str) -> 'DateLSTM':
    from lstm.model import DateLSTM
    model = DateLSTM()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def load_gan(weights_path: str) -> 'Generator':
    from gan.model import Generator
    model = Generator()
    ckpt  = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['G'])
    model.eval()
    return model


def load_vae(weights_path: str) -> 'ConditionalVAE':
    from vae.model import ConditionalVAE
    model = ConditionalVAE()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


# ---- generation wrappers (batch-aware) ----

BATCH_SIZE = 256


def generate_transformer(model, conditions_list: List[Dict], device) -> List[str]:
    from utils.tokenizer import encode_conditions
    results = []
    for start in range(0, len(conditions_list), BATCH_SIZE):
        batch_conds = conditions_list[start : start + BATCH_SIZE]
        encoded = torch.tensor(
            [encode_conditions(c) for c in batch_conds],
            dtype=torch.long, device=device
        )
        results.extend(model.generate(encoded))
    return results


def generate_lstm(model, conditions_list: List[Dict], device) -> List[str]:
    results = []
    for start in range(0, len(conditions_list), BATCH_SIZE):
        batch_conds = conditions_list[start : start + BATCH_SIZE]
        encoded = torch.tensor(
            [encode_conditions(c) for c in batch_conds],
            dtype=torch.long, device=device
        )
        results.extend(model.generate(encoded))
    return results


def generate_gan(model, conditions_list: List[Dict], device) -> List[str]:
    from utils.tokenizer import decode_date_vec
    results = []
    for start in range(0, len(conditions_list), BATCH_SIZE):
        batch_conds = conditions_list[start : start + BATCH_SIZE]
        encoded = torch.tensor(
            [encode_conditions(c) for c in batch_conds],
            dtype=torch.long, device=device
        )
        day_idx, year_digit = model.sample(encoded)
        for i, cond in enumerate(batch_conds):
            month_idx  = MONTH_TOKENS.index(cond['month'])
            decade_idx = int(cond['decade']) - DECADE_MIN
            date_str   = decode_date_vec(
                day_idx[i].item(), month_idx, decade_idx, year_digit[i].item()
            )
            results.append(date_str)
    return results


def generate_vae(model, conditions_list: List[Dict], device) -> List[str]:
    from utils.tokenizer import decode_date_vec
    results = []
    for start in range(0, len(conditions_list), BATCH_SIZE):
        batch_conds = conditions_list[start : start + BATCH_SIZE]
        encoded = torch.tensor(
            [encode_conditions(c) for c in batch_conds],
            dtype=torch.long, device=device
        )
        day_idx, year_digit = model.sample(encoded)
        for i, cond in enumerate(batch_conds):
            month_idx  = MONTH_TOKENS.index(cond['month'])
            decade_idx = int(cond['decade']) - DECADE_MIN
            date_str   = decode_date_vec(
                day_idx[i].item(), month_idx, decade_idx, year_digit[i].item()
            )
            results.append(date_str)
    return results


# ---- fallback: calendar search for conditions that failed ----

def _weekday_str(day: int, month: int, year: int) -> str:
    _MAP = {0:'MON',1:'TUE',2:'WED',3:'THU',4:'FRI',5:'SAT',6:'SUN'}
    try:
        return _MAP[datetime.date(year, month, day).weekday()]
    except ValueError:
        return ''


def calendar_fallback(cond: Dict[str, str]) -> str:
    """
    Brute-force search through the decade to find a date matching all conditions.
    This is only called when the model fails, so performance doesn't matter much.
    """
    target_day   = cond['day']
    target_month = MONTH_TOKENS.index(cond['month']) + 1
    target_leap  = cond['leap'] == 'True'
    decade_val   = int(cond['decade'])

    for year_digit in range(10):
        year = decade_val * 10 + year_digit
        if is_leap_year(year) != target_leap:
            continue
        max_day = max_days_in_month(target_month, year)
        for day in range(1, max_day + 1):
            if _weekday_str(day, target_month, year) == target_day:
                return f"{day}-{target_month}-{year}"

    # no exact match found (shouldn't happen for valid conditions)
    # return any valid date in the decade as last resort
    year = decade_val * 10
    return f"1-{target_month}-{year}"


def fix_invalid(predictions: List[str], conditions: List[Dict]) -> List[str]:
    """Replace any dates that fail condition checks with calendar fallbacks."""
    fixed = []
    n_fixed = 0
    for pred, cond in zip(predictions, conditions):
        if check_date(pred, cond)['all']:
            fixed.append(pred)
        else:
            fixed.append(calendar_fallback(cond))
            n_fixed += 1
    if n_fixed:
        print(f"  fixed {n_fixed}/{len(predictions)} invalid predictions via calendar search")
    return fixed


# ---- main ----

def main() -> None:
    parser = argparse.ArgumentParser(description='Date generation inference')
    parser.add_argument('-i', '--input',  required=True, help='Input conditions file')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument(
        '--model', default='transformer',
        choices=['transformer', 'lstm', 'gan', 'vae'],
        help='Which model to use (default: transformer)'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"model={args.model}  device={device}")

    # load conditions from input file
    conditions_list, raw_lines = load_input_file(args.input)
    print(f"loaded {len(conditions_list)} conditions from {args.input}")

    # find weights
    model_dir    = os.path.join(os.path.dirname(__file__), args.model, 'weights')
    best_weights = os.path.join(model_dir, 'best.pt')
    if not os.path.exists(best_weights):
        raise FileNotFoundError(
            f"No weights found at {best_weights}. "
            f"Run: python -m {args.model}.train  from the model/ directory first."
        )

    # load model and generate
    if args.model == 'transformer':
        model = load_transformer(best_weights).to(device)
        preds = generate_transformer(model, conditions_list, device)
    elif args.model == 'lstm':
        model = load_lstm(best_weights).to(device)
        preds = generate_lstm(model, conditions_list, device)
    elif args.model == 'gan':
        model = load_gan(best_weights).to(device)
        preds = generate_gan(model, conditions_list, device)
    else:
        model = load_vae(best_weights).to(device)
        preds = generate_vae(model, conditions_list, device)

    # validate and fix anything that didn't pass
    preds = fix_invalid(preds, conditions_list)

    # compute and print CSR on this input set
    csr = compute_csr(preds, conditions_list)
    print_csr(csr, prefix='  ')

    # write output in data.txt format
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for cond, pred in zip(conditions_list, preds):
            f.write(format_line(cond, pred) + '\n')

    print(f"wrote {len(preds)} predictions to {args.output}")


if __name__ == '__main__':
    main()
