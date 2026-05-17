"""
Modal training for the four version-1 models (GAN, VAE, Transformer, LSTM).

The local train.py scripts in each model folder still work as-is for
running training on your own machine. This file just wraps the same logic
for execution on a Modal A10G GPU and captures the best CSR so we can
report numbers in the README.

Usage from the project root:
    modal run "version(1)/modal_train.py" --model all      --epochs 25
    modal run "version(1)/modal_train.py" --model gan      --epochs 25
    modal run "version(1)/modal_train.py" --model vae      --epochs 25
    modal run "version(1)/modal_train.py" --model transformer
    modal run "version(1)/modal_train.py" --model lstm
"""

from pathlib import Path

import modal


LOCAL_DIR = Path(__file__).parent

app = modal.App("date-models-v1")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.2.0", "numpy<2", "tqdm")
    .add_local_dir(str(LOCAL_DIR / "model"), remote_path="/code/model")
    .add_local_dir(str(LOCAL_DIR / "data"),  remote_path="/code/data")
)

volume = modal.Volume.from_name("date-models-v1-weights", create_if_missing=True)

GPU_TYPE = "A10G"
TIMEOUT  = 3 * 60 * 60


# ---------- GAN ----------
@app.function(image=image, gpu=GPU_TYPE, volumes={"/ckpt": volume}, timeout=TIMEOUT)
def train_gan_remote(epochs: int = 25, batch_size: int = 512, lr: float = 2e-4):
    import sys, os, time, json
    sys.path.insert(0, "/code/model")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    from utils.dataset  import DateDataset
    from utils.evaluate import compute_csr
    from utils.tokenizer import decode_date_vec
    from gan.model import Generator, Discriminator, NOISE_DIM

    SEED, SAVE_DIR = 42, "/ckpt/gan"
    BETA1, BETA2 = 0.5, 0.999
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    device = torch.device("cuda")
    full_ds  = DateDataset("/code/data/data.txt", mode="vec")
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(SEED))
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_conds  = [full_ds.conditions[i] for i in val_ds.indices]

    G, D = Generator().to(device), Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(BETA1, BETA2))

    def date_to_onehot(date_comp):
        day_oh  = F.one_hot(date_comp[:, 0], num_classes=31).float()
        year_oh = F.one_hot(date_comp[:, 1], num_classes=10).float()
        return torch.cat([day_oh, year_oh], dim=-1)

    def validate():
        G.eval()
        preds = []
        with torch.no_grad():
            for cond_batch, _ in val_loader:
                cond_batch = cond_batch.to(device)
                day_idx, year_digit = G.sample(cond_batch)
                for i in range(len(cond_batch)):
                    preds.append(decode_date_vec(
                        day_idx[i].item(),
                        cond_batch[i, 1].item(),
                        cond_batch[i, 3].item(),
                        year_digit[i].item(),
                    ))
        return compute_csr(preds, val_conds)

    best_csr_all = 0.0
    best_csr     = {}

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        d_losses, g_losses = [], []
        t0 = time.time()
        for cond, date_comp in trn_loader:
            cond = cond.to(device); date_comp = date_comp.to(device)
            B = cond.size(0)
            real = date_to_onehot(date_comp)

            noise = torch.randn(B, NOISE_DIM, device=device)
            fake = G.soft_date(noise, cond).detach()
            rl = torch.full((B, 1), 0.9, device=device)
            fl = torch.full((B, 1), 0.1, device=device)
            opt_D.zero_grad()
            d_loss = (F.binary_cross_entropy_with_logits(D(real, cond), rl) +
                      F.binary_cross_entropy_with_logits(D(fake, cond), fl))
            d_loss.backward(); opt_D.step()
            d_losses.append(d_loss.item())

            for _ in range(2):
                noise = torch.randn(B, NOISE_DIM, device=device)
                fake = G.soft_date(noise, cond)
                opt_G.zero_grad()
                g_loss = F.binary_cross_entropy_with_logits(
                    D(fake, cond), torch.ones(B, 1, device=device))
                g_loss.backward(); opt_G.step()
            g_losses.append(g_loss.item())

        csr = validate()
        print(f"[gan] epoch {epoch:02d}/{epochs} | D={np.mean(d_losses):.4f} G={np.mean(g_losses):.4f}"
              f" | {time.time()-t0:.1f}s | CSR all={csr.get('all', 0):.4f}")
        if csr.get("all", 0) > best_csr_all:
            best_csr_all = csr["all"]
            best_csr     = dict(csr)
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "csr": best_csr},
                       os.path.join(SAVE_DIR, "best.pt"))

    json.dump(best_csr, open(os.path.join(SAVE_DIR, "best_csr.json"), "w"))
    volume.commit()
    return best_csr


# ---------- VAE ----------
@app.function(image=image, gpu=GPU_TYPE, volumes={"/ckpt": volume}, timeout=TIMEOUT)
def train_vae_remote(epochs: int = 25, batch_size: int = 512, lr: float = 1e-3):
    import sys, os, time, json
    sys.path.insert(0, "/code/model")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    from utils.dataset  import DateDataset
    from utils.evaluate import compute_csr
    from utils.tokenizer import decode_date_vec
    from vae.model import ConditionalVAE, vae_loss

    SEED, SAVE_DIR = 42, "/ckpt/vae"
    BETA_MAX, BETA_ANNEAL = 1.0, max(epochs // 2, 1)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    device = torch.device("cuda")
    full_ds  = DateDataset("/code/data/data.txt", mode="vec")
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(SEED))
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_conds  = [full_ds.conditions[i] for i in val_ds.indices]

    def date_to_onehot(date_comp):
        day_oh  = F.one_hot(date_comp[:, 0], num_classes=31).float()
        year_oh = F.one_hot(date_comp[:, 1], num_classes=10).float()
        return torch.cat([day_oh, year_oh], dim=-1)

    model = ConditionalVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def validate():
        model.eval()
        preds = []
        with torch.no_grad():
            for cond_batch, _ in val_loader:
                cond_batch = cond_batch.to(device)
                day_idx, year_digit = model.sample(cond_batch)
                for i in range(len(cond_batch)):
                    preds.append(decode_date_vec(
                        day_idx[i].item(),
                        cond_batch[i, 1].item(),
                        cond_batch[i, 3].item(),
                        year_digit[i].item(),
                    ))
        return compute_csr(preds, val_conds)

    best_csr_all = 0.0
    best_csr     = {}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        beta = min(BETA_MAX, BETA_MAX * (epoch - 1) / max(BETA_ANNEAL, 1))

        for cond, date_comp in trn_loader:
            cond = cond.to(device); date_comp = date_comp.to(device)
            date_vec = date_to_onehot(date_comp)
            logits, mu, logvar = model(cond, date_vec)
            loss, _, _ = vae_loss(logits, date_comp, mu, logvar, beta=beta)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

        csr = validate()
        print(f"[vae] epoch {epoch:02d}/{epochs} | loss={np.mean(losses):.4f}"
              f" | beta={beta:.3f} | {time.time()-t0:.1f}s | CSR all={csr.get('all', 0):.4f}")
        if csr.get("all", 0) > best_csr_all:
            best_csr_all = csr["all"]
            best_csr     = dict(csr)
            torch.save({"model": model.state_dict(), "csr": best_csr},
                       os.path.join(SAVE_DIR, "best.pt"))

    json.dump(best_csr, open(os.path.join(SAVE_DIR, "best_csr.json"), "w"))
    volume.commit()
    return best_csr


# ---------- Transformer ----------
@app.function(image=image, gpu=GPU_TYPE, volumes={"/ckpt": volume}, timeout=TIMEOUT)
def train_transformer_remote(epochs: int = 25, batch_size: int = 512, lr: float = 5e-4):
    import sys, os, time, json
    sys.path.insert(0, "/code/model")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    from utils.dataset   import DateDataset
    from utils.evaluate  import compute_csr
    from utils.tokenizer import PAD_ID, CHAR_VOCAB_SIZE
    from transformer.model import DateTransformer

    SEED, SAVE_DIR = 42, "/ckpt/transformer"
    WARMUP = 2000
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    device = torch.device("cuda")
    full_ds  = DateDataset("/code/data/data.txt", mode="seq")
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(SEED))
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_conds  = [full_ds.conditions[i] for i in val_ds.indices]

    model = DateTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, s / max(WARMUP, 1)))

    def validate():
        model.eval()
        preds = []
        with torch.no_grad():
            for cond_batch, _ in val_loader:
                cond_batch = cond_batch.to(device)
                preds.extend(model.generate(cond_batch))
        return compute_csr(preds, val_conds)

    best_csr_all = 0.0
    best_csr     = {}
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for cond, date_seq in trn_loader:
            cond = cond.to(device); date_seq = date_seq.to(device)
            tgt_in  = date_seq[:, :-1]
            tgt_out = date_seq[:, 1:]
            logits = model(cond, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, CHAR_VOCAB_SIZE),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()
            losses.append(loss.item())
            step += 1

        csr = validate()
        print(f"[trf] epoch {epoch:02d}/{epochs} | loss={np.mean(losses):.4f}"
              f" | {time.time()-t0:.1f}s | CSR all={csr.get('all', 0):.4f}")
        if csr.get("all", 0) > best_csr_all:
            best_csr_all = csr["all"]
            best_csr     = dict(csr)
            torch.save({"model": model.state_dict(), "csr": best_csr},
                       os.path.join(SAVE_DIR, "best.pt"))

    json.dump(best_csr, open(os.path.join(SAVE_DIR, "best_csr.json"), "w"))
    volume.commit()
    return best_csr


# ---------- LSTM ----------
@app.function(image=image, gpu=GPU_TYPE, volumes={"/ckpt": volume}, timeout=TIMEOUT)
def train_lstm_remote(epochs: int = 25, batch_size: int = 512, lr: float = 1e-3):
    import sys, os, time, json
    sys.path.insert(0, "/code/model")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    from utils.dataset  import DateDataset
    from utils.evaluate import compute_csr
    from utils.tokenizer import PAD_ID, CHAR_VOCAB_SIZE
    from lstm.model import DateLSTM

    SEED, SAVE_DIR = 42, "/ckpt/lstm"
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    device = torch.device("cuda")
    full_ds  = DateDataset("/code/data/data.txt", mode="seq")
    val_size = len(full_ds) // 10
    trn_size = len(full_ds) - val_size
    trn_ds, val_ds = random_split(full_ds, [trn_size, val_size],
                                  generator=torch.Generator().manual_seed(SEED))
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_conds  = [full_ds.conditions[i] for i in val_ds.indices]

    model = DateLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def validate():
        model.eval()
        preds = []
        with torch.no_grad():
            for cond_batch, _ in val_loader:
                cond_batch = cond_batch.to(device)
                preds.extend(model.generate(cond_batch))
        return compute_csr(preds, val_conds)

    best_csr_all = 0.0
    best_csr     = {}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for cond, date_seq in trn_loader:
            cond = cond.to(device); date_seq = date_seq.to(device)
            tgt_in  = date_seq[:, :-1]
            tgt_out = date_seq[:, 1:]
            logits = model(cond, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, CHAR_VOCAB_SIZE),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

        csr = validate()
        print(f"[lstm] epoch {epoch:02d}/{epochs} | loss={np.mean(losses):.4f}"
              f" | {time.time()-t0:.1f}s | CSR all={csr.get('all', 0):.4f}")
        if csr.get("all", 0) > best_csr_all:
            best_csr_all = csr["all"]
            best_csr     = dict(csr)
            torch.save({"model": model.state_dict(), "csr": best_csr},
                       os.path.join(SAVE_DIR, "best.pt"))

    json.dump(best_csr, open(os.path.join(SAVE_DIR, "best_csr.json"), "w"))
    volume.commit()
    return best_csr


@app.local_entrypoint()
def main(model: str = "all", epochs: int = 25):
    """model in {gan, vae, transformer, lstm, all}"""
    if model == "gan":
        print("gan CSR:", train_gan_remote.remote(epochs=epochs))
    elif model == "vae":
        print("vae CSR:", train_vae_remote.remote(epochs=epochs))
    elif model == "transformer":
        print("transformer CSR:", train_transformer_remote.remote(epochs=epochs))
    elif model == "lstm":
        print("lstm CSR:", train_lstm_remote.remote(epochs=epochs))
    elif model == "all":
        h_g = train_gan_remote.spawn(epochs=epochs)
        h_v = train_vae_remote.spawn(epochs=epochs)
        h_t = train_transformer_remote.spawn(epochs=epochs)
        h_l = train_lstm_remote.spawn(epochs=epochs)
        print("gan         CSR:", h_g.get())
        print("vae         CSR:", h_v.get())
        print("transformer CSR:", h_t.get())
        print("lstm        CSR:", h_l.get())
    else:
        raise ValueError(f"unknown model: {model}")
