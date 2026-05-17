"""
Training loop for the Conditional VAE.

Run from model/:
    python -m vae.train

The beta parameter controls the KL weight. I'm using beta annealing:
starting at 0 and slowly increasing to 1 over the first half of training.
This prevents the KL term from collapsing the latent space too early.
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
from utils.tokenizer import decode_date_vec, N_DAY_VALS, DATE_VEC_DIM
from vae.model import ConditionalVAE, vae_loss


SEED       = 42
BATCH_SIZE = 512
N_EPOCHS   = 60
LR         = 1e-3
BETA_MAX   = 1.0     # final KL weight
BETA_ANNEAL_EPOCHS = 30   # ramp beta up over first 30 epochs

SAVE_DIR  = os.path.join(os.path.dirname(__file__), 'weights')
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data', 'data.txt'
)


def date_to_onehot(date_comp: torch.Tensor) -> torch.Tensor:
    day_oh  = F.one_hot(date_comp[:, 0], num_classes=31).float()
    year_oh = F.one_hot(date_comp[:, 1], num_classes=10).float()
    return torch.cat([day_oh, year_oh], dim=-1)


def validate(model: ConditionalVAE, val_loader: DataLoader, val_conditions, device) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for cond_batch, _ in val_loader:
            cond_batch = cond_batch.to(device)
            day_idx, year_digit = model.sample(cond_batch)
            for i in range(len(cond_batch)):
                date_str = decode_date_vec(
                    day_idx[i].item(),
                    cond_batch[i, 1].item(),
                    cond_batch[i, 3].item(),
                    year_digit[i].item(),
                )
                preds.append(date_str)
    return compute_csr(preds, val_conditions)


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    full_ds  = DateDataset(DATA_PATH, mode='vec')
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(
        full_ds, [trn_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    val_conditions = [full_ds.conditions[i] for i in val_ds.indices]

    model     = ConditionalVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_csr = 0.0

    for epoch in range(N_EPOCHS):
        model.train()
        total_losses, recon_losses, kld_losses = [], [], []

        # anneal beta from 0 to BETA_MAX over the first BETA_ANNEAL_EPOCHS epochs
        beta = min(BETA_MAX, BETA_MAX * epoch / max(BETA_ANNEAL_EPOCHS, 1))

        for cond, date_comp in trn_loader:
            cond      = cond.to(device)
            date_comp = date_comp.to(device)
            date_vec  = date_to_onehot(date_comp)

            logits, mu, logvar = model(cond, date_vec)
            loss, recon, kld   = vae_loss(logits, date_comp, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            # clip gradients to avoid occasional spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_losses.append(loss.item())
            recon_losses.append(recon.item())
            kld_losses.append(kld.item())

        csr = validate(model, val_loader, val_conditions, device)
        print(
            f"epoch {epoch+1:02d}/{N_EPOCHS} "
            f"| loss={np.mean(total_losses):.4f} "
            f"recon={np.mean(recon_losses):.4f} "
            f"kld={np.mean(kld_losses):.4f} "
            f"beta={beta:.3f}",
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
