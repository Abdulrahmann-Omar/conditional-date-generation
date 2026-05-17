"""
Training loop for the Seq2Seq LSTM.

Run from model/:
    python -m lstm.train

Same teacher-forcing setup as the Transformer.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset  import DateDataset
from utils.evaluate import compute_csr, print_csr
from utils.tokenizer import PAD_ID, CHAR_VOCAB_SIZE
from lstm.model import DateLSTM


SEED       = 42
BATCH_SIZE = 512
N_EPOCHS   = 60
LR         = 1e-3

SAVE_DIR  = os.path.join(os.path.dirname(__file__), 'weights')
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data', 'data.txt'
)


def validate(model: DateLSTM, val_loader: DataLoader, val_conditions, device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for cond_batch, _ in val_loader:
            cond_batch = cond_batch.to(device)
            preds.extend(model.generate(cond_batch))
    return compute_csr(preds, val_conditions)


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    full_ds  = DateDataset(DATA_PATH, mode='seq')
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(
        full_ds, [trn_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    val_conditions = [full_ds.conditions[i] for i in val_ds.indices]

    model     = DateLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # reduce LR when CSR plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_csr = 0.0

    for epoch in range(N_EPOCHS):
        model.train()
        losses = []

        for cond, date_seq in trn_loader:
            cond     = cond.to(device)
            date_seq = date_seq.to(device)

            tgt_in  = date_seq[:, :-1]
            tgt_out = date_seq[:, 1:]

            logits = model(cond, tgt_in)   # (B, T, vocab_size)

            loss = F.cross_entropy(
                logits.reshape(-1, CHAR_VOCAB_SIZE),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss.item())

        csr = validate(model, val_loader, val_conditions, device)
        scheduler.step(csr.get('all', 0))

        print(
            f"epoch {epoch+1:02d}/{N_EPOCHS} "
            f"| loss={np.mean(losses):.4f}",
            end='  ',
        )
        print_csr(csr)

        if csr.get('all', 0) > best_csr:
            best_csr = csr['all']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pt'))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final.pt'))
    print(f"\nDone. Best val CSR (all): {best_csr:.4f}")


if __name__ == '__main__':
    train()
