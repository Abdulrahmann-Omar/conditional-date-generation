"""
Diffusion-LM training routine. Mirrors d3pm/trainer.py.
"""

import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from utils.dataset  import DateDataset
from utils.evaluate import compute_csr, print_csr
from utils.tokenizer import detokenize_fixed
from diffusion_lm.model import (
    DiffusionLMTransformer, diffusion_lm_loss, diffusion_lm_sample,
    build_schedule, DEFAULT_T,
)


def _validate(model, val_loader, val_conds, bar_alpha, T, device) -> Dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for cond_batch, _ in val_loader:
            cond_batch = cond_batch.to(device)
            samples = diffusion_lm_sample(model, cond_batch, bar_alpha, T)
            for b in range(samples.size(0)):
                preds.append(detokenize_fixed(samples[b].tolist()))
    return compute_csr(preds, val_conds)


def run_training(
    data_path:  str,
    ckpt_dir:   str,
    epochs:     int = 40,
    batch_size: int = 256,
    lr:         float = 1e-4,
    T:          int = DEFAULT_T,
    seed:       int = 42,
    num_workers: int = 4,
    round_weight: float = 1.0,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[dlm] device={device}, epochs={epochs}, batch={batch_size}, lr={lr}, T={T}")

    full_ds  = DateDataset(data_path)
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(
        full_ds, [trn_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    val_conds  = [full_ds.conditions[i] for i in val_ds.indices]

    model     = DiffusionLMTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bar_alpha = build_schedule(T).to(device)

    os.makedirs(ckpt_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dlm] parameters: {n_params:,}")

    best_csr_all  = 0.0
    best_csr_dict: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for cond, x_0_ids in trn_loader:
            cond     = cond.to(device)
            x_0_ids  = x_0_ids.to(device)
            loss     = diffusion_lm_loss(
                model, x_0_ids, cond, bar_alpha, T,
                round_weight=round_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        csr = _validate(model, val_loader, val_conds, bar_alpha, T, device)
        dt  = time.time() - t0

        print(
            f"[dlm] epoch {epoch:02d}/{epochs} "
            f"| loss={np.mean(losses):.4f} | time={dt:.1f}s",
            end='  ',
        )
        print_csr(csr)

        if csr.get('all', 0) > best_csr_all:
            best_csr_all  = csr['all']
            best_csr_dict = dict(csr)
            torch.save({
                'model':     model.state_dict(),
                'T':         T,
                'bar_alpha': bar_alpha.cpu(),
                'csr':       best_csr_dict,
                'epoch':     epoch,
            }, os.path.join(ckpt_dir, 'best.pt'))

    torch.save({
        'model':     model.state_dict(),
        'T':         T,
        'bar_alpha': bar_alpha.cpu(),
        'csr':       csr,
        'epoch':     epochs,
    }, os.path.join(ckpt_dir, 'final.pt'))

    print(f"[dlm] DONE. best CSR(all)={best_csr_all:.4f}")
    return best_csr_dict if best_csr_dict else csr
