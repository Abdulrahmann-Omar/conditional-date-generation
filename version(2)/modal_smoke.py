"""Quick smoke test: imports the model code on Modal and prints param counts."""

from pathlib import Path
import modal

LOCAL_DIR = Path(__file__).parent

app = modal.App("date-diffusion-smoke")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.2.0", "numpy<2", "tqdm")
    .add_local_dir(str(LOCAL_DIR / "model"), remote_path="/code/model")
    .add_local_dir(str(LOCAL_DIR / "data"),  remote_path="/code/data")
)


@app.function(image=image, gpu="A10G", timeout=300)
def smoke():
    import sys, torch
    sys.path.insert(0, "/code/model")

    from d3pm.model         import D3PMTransformer, build_schedule, d3pm_loss, d3pm_sample
    from diffusion_lm.model import DiffusionLMTransformer, build_schedule as dlm_sched, diffusion_lm_loss, diffusion_lm_sample
    from utils.tokenizer    import VOCAB_SIZE, SEQ_LEN, MASK_ID

    print(f"torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    print(f"vocab_size={VOCAB_SIZE}, seq_len={SEQ_LEN}, mask_id={MASK_ID}")

    device = torch.device("cuda")

    # ---- D3PM forward pass smoke test ----
    d3pm   = D3PMTransformer().to(device)
    bar_a  = build_schedule(100).to(device)
    x0     = torch.randint(0, 11, (4, SEQ_LEN), device=device)
    cond   = torch.tensor([[0, 0, 0, 0]] * 4, device=device)
    loss   = d3pm_loss(d3pm, x0, cond, bar_a, T=100)
    print(f"d3pm loss: {loss.item():.4f}, params: {sum(p.numel() for p in d3pm.parameters()):,}")
    samp = d3pm_sample(d3pm, cond, bar_a, T=100)
    print(f"d3pm sample shape: {samp.shape}, first row: {samp[0].tolist()}")

    # ---- Diffusion-LM forward pass smoke test ----
    dlm    = DiffusionLMTransformer().to(device)
    bar_a2 = dlm_sched(200).to(device)
    loss2  = diffusion_lm_loss(dlm, x0, cond, bar_a2, T=200)
    print(f"dlm  loss: {loss2.item():.4f}, params: {sum(p.numel() for p in dlm.parameters()):,}")
    samp2 = diffusion_lm_sample(dlm, cond, bar_a2, T=200)
    print(f"dlm  sample shape: {samp2.shape}, first row: {samp2[0].tolist()}")

    # check data file
    import os
    n_lines = sum(1 for _ in open("/code/data/data.txt"))
    print(f"data.txt lines: {n_lines}")

    return {"status": "ok", "n_lines": n_lines}


@app.local_entrypoint()
def main():
    res = smoke.remote()
    print("RESULT:", res)
