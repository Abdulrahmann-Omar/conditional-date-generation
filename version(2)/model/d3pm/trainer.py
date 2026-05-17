"""
D3PM training routine. Called from modal_train.py either locally or on Modal.

Designed so the entire training run is one self-contained function that takes
hyperparameters as arguments and writes checkpoints to a directory.
"""

import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.dataset  import DateDataset
from utils.evaluate import compute_csr, print_csr
from utils.tokenizer import detokenize_fixed, SEQ_LEN
from d3pm.model import (
    D3PMTransformer, d3pm_loss, d3pm_sample, build_schedule, DEFAULT_T,
)


def _validate(model, val_loader, val_conds, bar_alpha, T, device) -> Dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for cond_batch, _ in val_loader:
            cond_batch = cond_batch.to(device)
            samples = d3pm_sample(model, cond_batch, bar_alpha, T)
            for b in range(samples.size(0)):
                date_str = detokenize_fixed(samples[b].tolist())
                preds.append(date_str)
    return compute_csr(preds, val_conds)


def run_training(
    data_path:  str,
    ckpt_dir:   str,
    epochs:     int = 40,
    batch_size: int = 256,
    lr:         float = 3e-4,
    T:          int = DEFAULT_T,
    seed:       int = 42,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Train D3PM end-to-end. Returns the best validation CSR dict so the
    caller can report it.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[d3pm] device={device}, epochs={epochs}, batch={batch_size}, lr={lr}, T={T}")

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

    model     = D3PMTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bar_alpha = build_schedule(T).to(device)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_csr_all = 0.0
    best_csr_dict: Dict[str, float] = {}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[d3pm] parameters: {n_params:,}")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for cond, x_0 in trn_loader:
            cond = cond.to(device)
            x_0  = x_0.to(device)
            loss = d3pm_loss(model, x_0, cond, bar_alpha, T)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        csr = _validate(model, val_loader, val_conds, bar_alpha, T, device)
        dt  = time.time() - t0

        print(
            f"[d3pm] epoch {epoch:02d}/{epochs} "
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

    # also save final
    torch.save({
        'model':     model.state_dict(),
        'T':         T,
        'bar_alpha': bar_alpha.cpu(),
        'csr':       csr,
        'epoch':     epochs,
    }, os.path.join(ckpt_dir, 'final.pt'))

    print(f"[d3pm] DONE. best CSR(all)={best_csr_all:.4f}")
    return best_csr_dict if best_csr_dict else csr
