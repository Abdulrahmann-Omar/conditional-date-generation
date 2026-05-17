"""
Local inference using checkpoints downloaded from the Modal volume.

After running `modal run modal_train.py`, pull the weights back with:
    modal volume get date-diffusion-weights /d3pm ./weights
    modal volume get date-diffusion-weights /diffusion_lm ./weights

Then run:
    python predict.py -i data/example_input.txt -o out.txt --model d3pm
    python predict.py -i data/example_input.txt -o out.txt --model diffusion_lm

There's also a calendar fallback (same idea as version 1) that handles the
edge cases where the model outputs something nonsensical.
"""

import argparse
import datetime
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from utils.dataset   import load_input_file
from utils.evaluate  import check_date, compute_csr, print_csr
from utils.tokenizer import (
    encode_conditions, MONTH_TOKENS, DECADE_MIN,
    detokenize_fixed, is_leap_year, max_days_in_month,
)


BATCH = 256


def _load_d3pm(path: str, device):
    from d3pm.model import D3PMTransformer
    ckpt = torch.load(path, map_location='cpu')
    model = D3PMTransformer().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    bar_alpha = ckpt['bar_alpha'].to(device)
    T = ckpt['T']
    return model, bar_alpha, T


def _load_diffusion_lm(path: str, device):
    from diffusion_lm.model import DiffusionLMTransformer
    ckpt = torch.load(path, map_location='cpu')
    model = DiffusionLMTransformer().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    bar_alpha = ckpt['bar_alpha'].to(device)
    T = ckpt['T']
    return model, bar_alpha, T


def _gen_d3pm(model, conds_list: List[Dict], bar_alpha, T, device) -> List[str]:
    from d3pm.model import d3pm_sample
    out = []
    for i in range(0, len(conds_list), BATCH):
        batch = conds_list[i:i + BATCH]
        cond_t = torch.tensor(
            [encode_conditions(c) for c in batch],
            dtype=torch.long, device=device,
        )
        samples = d3pm_sample(model, cond_t, bar_alpha, T)
        for b in range(samples.size(0)):
            out.append(detokenize_fixed(samples[b].tolist()))
    return out


def _gen_diffusion_lm(model, conds_list: List[Dict], bar_alpha, T, device) -> List[str]:
    from diffusion_lm.model import diffusion_lm_sample
    out = []
    for i in range(0, len(conds_list), BATCH):
        batch = conds_list[i:i + BATCH]
        cond_t = torch.tensor(
            [encode_conditions(c) for c in batch],
            dtype=torch.long, device=device,
        )
        samples = diffusion_lm_sample(model, cond_t, bar_alpha, T)
        for b in range(samples.size(0)):
            out.append(detokenize_fixed(samples[b].tolist()))
    return out


# ---------- calendar fallback (same as v1) ----------

def _weekday_str(day, month, year):
    m = {0:'MON',1:'TUE',2:'WED',3:'THU',4:'FRI',5:'SAT',6:'SUN'}
    try:
        return m[datetime.date(year, month, day).weekday()]
    except ValueError:
        return ''


def calendar_fallback(cond: Dict[str, str]) -> str:
    target_day   = cond['day']
    target_month = MONTH_TOKENS.index(cond['month']) + 1
    target_leap  = cond['leap'] == 'True'
    decade_val   = int(cond['decade'])
    for year_digit in range(10):
        year = decade_val * 10 + year_digit
        if is_leap_year(year) != target_leap:
            continue
        for day in range(1, max_days_in_month(target_month, year) + 1):
            if _weekday_str(day, target_month, year) == target_day:
                return f"{day}-{target_month}-{year}"
    return f"1-{target_month}-{decade_val * 10}"


def fix_invalid(preds: List[str], conds: List[Dict]) -> List[str]:
    fixed = []
    n_fixed = 0
    for p, c in zip(preds, conds):
        if check_date(p, c)['all']:
            fixed.append(p)
        else:
            fixed.append(calendar_fallback(c))
            n_fixed += 1
    if n_fixed:
        print(f"  fixed {n_fixed}/{len(preds)} invalid predictions via calendar fallback")
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument(
        '--model', default='d3pm',
        choices=['d3pm', 'diffusion_lm'],
    )
    parser.add_argument(
        '--weights', default=None,
        help='path to .pt file. defaults to weights/<model>/best.pt',
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"model={args.model} device={device}")

    conds_list, _ = load_input_file(args.input)
    print(f"loaded {len(conds_list)} conditions")

    weights_path = args.weights or os.path.join(
        os.path.dirname(__file__), "weights", args.model, "best.pt"
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"no weights at {weights_path}")

    if args.model == 'd3pm':
        model, bar_alpha, T = _load_d3pm(weights_path, device)
        preds = _gen_d3pm(model, conds_list, bar_alpha, T, device)
    else:
        model, bar_alpha, T = _load_diffusion_lm(weights_path, device)
        preds = _gen_diffusion_lm(model, conds_list, bar_alpha, T, device)

    preds = fix_invalid(preds, conds_list)
    csr   = compute_csr(preds, conds_list)
    print_csr(csr, prefix='  ')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        for c, p in zip(conds_list, preds):
            f.write(f"[{c['day']}] [{c['month']}] [{c['leap']}] [{c['decade']}] {p}\n")
    print(f"wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
