"""
Modal entry point for training the diffusion models.

Usage from the local machine:
    modal run modal_train.py --model d3pm --epochs 40
    modal run modal_train.py --model diffusion_lm --epochs 40
    modal run modal_train.py --model both --epochs 40

The training function runs on a Modal A10G GPU, mounts the local code and
data, and writes checkpoints to a persistent Modal Volume. After training
you can pull weights back to your laptop with:

    modal volume get date-diffusion-weights /d3pm ./weights/
    modal volume get date-diffusion-weights /diffusion_lm ./weights/
"""

from pathlib import Path

import modal


LOCAL_DIR = Path(__file__).parent

app = modal.App("date-diffusion")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "tqdm",
    )
    # bake the code into the image so workers can import from /code
    .add_local_dir(str(LOCAL_DIR / "model"), remote_path="/code/model")
    .add_local_dir(str(LOCAL_DIR / "data"),  remote_path="/code/data")
)

# persistent volume for checkpoints
volume = modal.Volume.from_name("date-diffusion-weights", create_if_missing=True)

GPU_TYPE  = "A10G"
TIMEOUT_S = 3 * 60 * 60   # 3 hours, plenty for 40 epochs


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/ckpt": volume},
    timeout=TIMEOUT_S,
)
def train_d3pm_remote(
    epochs:     int   = 40,
    batch_size: int   = 256,
    lr:         float = 3e-4,
    T:          int   = 100,
):
    import sys
    sys.path.insert(0, "/code/model")
    from d3pm.trainer import run_training
    csr = run_training(
        data_path="/code/data/data.txt",
        ckpt_dir="/ckpt/d3pm",
        epochs=epochs, batch_size=batch_size, lr=lr, T=T,
    )
    volume.commit()
    return csr


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/ckpt": volume},
    timeout=TIMEOUT_S,
)
def train_diffusion_lm_remote(
    epochs:     int   = 40,
    batch_size: int   = 256,
    lr:         float = 1e-4,
    T:          int   = 200,
):
    import sys
    sys.path.insert(0, "/code/model")
    from diffusion_lm.trainer import run_training
    csr = run_training(
        data_path="/code/data/data.txt",
        ckpt_dir="/ckpt/diffusion_lm",
        epochs=epochs, batch_size=batch_size, lr=lr, T=T,
    )
    volume.commit()
    return csr


@app.local_entrypoint()
def main(model: str = "both", epochs: int = 40):
    """Choose model in {d3pm, diffusion_lm, both}."""
    if model == "d3pm":
        csr = train_d3pm_remote.remote(epochs=epochs)
        print(f"d3pm best CSR: {csr}")
    elif model == "diffusion_lm":
        csr = train_diffusion_lm_remote.remote(epochs=epochs)
        print(f"diffusion_lm best CSR: {csr}")
    elif model == "both":
        # run both in parallel using spawn, then collect results
        h1 = train_d3pm_remote.spawn(epochs=epochs)
        h2 = train_diffusion_lm_remote.spawn(epochs=epochs)
        csr_d3 = h1.get()
        csr_dl = h2.get()
        print(f"d3pm         best CSR: {csr_d3}")
        print(f"diffusion_lm best CSR: {csr_dl}")
    else:
        raise ValueError(f"unknown model: {model}")
