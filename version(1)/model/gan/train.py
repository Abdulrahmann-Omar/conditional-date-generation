"""
Training loop for the Conditional GAN.

Run from the model/ directory:
    python -m gan.train

Saves best checkpoint (by validation CSR) to gan/weights/best.pt
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# make sure we can import utils and gan from model/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset  import DateDataset
from utils.evaluate import compute_csr, print_csr
from utils.tokenizer import decode_date_vec, N_DAY_VALS
from gan.model import Generator, Discriminator, NOISE_DIM


# ---- hyperparameters ----
SEED       = 42
BATCH_SIZE = 512
N_EPOCHS   = 60
LR         = 2e-4
BETA1      = 0.5
BETA2      = 0.999

SAVE_DIR  = os.path.join(os.path.dirname(__file__), 'weights')
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data', 'data.txt'
)


def date_to_onehot(date_comp: torch.Tensor) -> torch.Tensor:
    """Convert (B, 2) [day_idx, year_digit] to a (B, 41) one-hot vector."""
    day_oh   = F.one_hot(date_comp[:, 0], num_classes=31).float()
    year_oh  = F.one_hot(date_comp[:, 1], num_classes=10).float()
    return torch.cat([day_oh, year_oh], dim=-1)


def validate(G: Generator, val_loader: DataLoader, val_conditions, device) -> dict:
    G.eval()
    preds = []
    with torch.no_grad():
        for cond_batch, _ in val_loader:
            cond_batch = cond_batch.to(device)
            day_idx, year_digit = G.sample(cond_batch)
            for i in range(len(cond_batch)):
                date_str = decode_date_vec(
                    day_idx[i].item(),
                    cond_batch[i, 1].item(),    # month comes from condition
                    cond_batch[i, 3].item(),    # decade too
                    year_digit[i].item(),
                )
                preds.append(date_str)
    return compute_csr(preds, val_conditions)


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # data
    full_ds  = DateDataset(DATA_PATH, mode='vec')
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(
        full_ds, [trn_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # get the raw condition dicts for validation CSR
    val_conditions = [full_ds.conditions[i] for i in val_ds.indices]

    # models
    G     = Generator().to(device)
    D     = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_csr = 0.0

    for epoch in range(N_EPOCHS):
        G.train()
        D.train()
        d_losses, g_losses = [], []

        for cond, date_comp in trn_loader:
            cond      = cond.to(device)
            date_comp = date_comp.to(device)
            B         = cond.size(0)

            real = date_to_onehot(date_comp)

            # ----- train D -----
            noise = torch.randn(B, NOISE_DIM, device=device)
            fake  = G.soft_date(noise, cond).detach()

            # label smoothing: real=0.9, fake=0.1 - helps D not get overconfident
            real_labels = torch.full((B, 1), 0.9, device=device)
            fake_labels = torch.full((B, 1), 0.1, device=device)

            opt_D.zero_grad()
            d_loss = (
                F.binary_cross_entropy_with_logits(D(real, cond), real_labels) +
                F.binary_cross_entropy_with_logits(D(fake, cond), fake_labels)
            )
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            # ----- train G (twice per D step) -----
            for _ in range(2):
                noise  = torch.randn(B, NOISE_DIM, device=device)
                fake   = G.soft_date(noise, cond)
                opt_G.zero_grad()
                g_loss = F.binary_cross_entropy_with_logits(
                    D(fake, cond),
                    torch.ones(B, 1, device=device),
                )
                g_loss.backward()
                opt_G.step()
            g_losses.append(g_loss.item())

        csr = validate(G, val_loader, val_conditions, device)
        print(
            f"epoch {epoch+1:02d}/{N_EPOCHS} "
            f"| D={np.mean(d_losses):.4f} G={np.mean(g_losses):.4f}",
            end='  ')
        print_csr(csr)

        if csr.get('all', 0) > best_csr:
            best_csr = csr['all']
            torch.save({'G': G.state_dict(), 'D': D.state_dict()},
                       os.path.join(SAVE_DIR, 'best.pt'))

    torch.save({'G': G.state_dict(), 'D': D.state_dict()},
               os.path.join(SAVE_DIR, 'final.pt'))
    print(f"\nDone. Best val CSR (all): {best_csr:.4f}")


if __name__ == '__main__':
    train()
