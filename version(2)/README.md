# Version 2: diffusion models on Modal

This folder is the cloud-trained extension of version 1. Two new models live
here, both from the diffusion family, and the training runs on Modal so I
can use a real GPU instead of grinding CPU on my laptop.

The models are:

- **D3PM** (Discrete Denoising Diffusion Probabilistic Model). Operates on
  the date as a sequence of tokens. The forward process gradually replaces
  digits with a `[MASK]` token; the reverse process is a small Transformer
  that denoises step by step. Paper: Austin et al. 2021, "Structured
  Denoising Diffusion Models in Discrete State-Spaces."

- **Diffusion-LM**. Embeds the tokens into a continuous space, runs a
  standard DDPM noise-and-denoise loop in that space, then rounds the
  final continuous output back to tokens. Paper: Li et al. 2022,
  "Diffusion-LM Improves Controllable Text Generation."

Both share the same conditioning setup as the version-1 models: the four
condition tokens (day, month, leap, decade) are embedded and fed to the
denoiser at every step.

## Layout

```
version(2)/
├── modal_train.py         # Modal app + train_d3pm / train_diffusion_lm
├── modal_smoke.py         # 30-second sanity check on Modal
├── predict.py             # local inference with downloaded weights
├── requirements.txt
├── data/                  # data.txt + example_input.txt
└── model/
    ├── utils/             # shared tokenizer / dataset / metric / common
    ├── d3pm/
    │   ├── model.py
    │   └── trainer.py
    └── diffusion_lm/
        ├── model.py
        └── trainer.py
```

The big difference from version 1: the date is encoded at a **fixed length**
of 10 characters (`DD-MM-YYYY`, zero-padded). Diffusion models really want
fixed-size sequences and avoiding variable lengths makes everything cleaner.
The vocabulary becomes `{0..9, '-', [MASK]}` for 12 tokens.

## Setting up Modal

```bash
pip install modal
modal setup    # creates an account and writes a token
```

That's all the auth you need. Modal handles everything else (image build,
GPU allocation, file mounting).

## Training

From inside `version(2)/`:

```bash
# train both diffusion models in parallel on Modal A10G GPUs
modal run modal_train.py --model both --epochs 25

# or one at a time
modal run modal_train.py --model d3pm --epochs 25
modal run modal_train.py --model diffusion_lm --epochs 25
```

Behind the scenes Modal builds a Python 3.10 image with `torch==2.2.0`,
mounts the local `model/` and `data/` directories into the container,
asks for an A10G GPU, and runs the training. Checkpoints land on a
persistent Modal volume named `date-diffusion-weights`.

A 25-epoch run on the full 146k dataset takes roughly 10-15 minutes per
model on an A10G.

## Pulling weights back

Once training is done, grab the checkpoints to run inference locally:

```bash
modal volume get date-diffusion-weights /d3pm           ./weights
modal volume get date-diffusion-weights /diffusion_lm   ./weights
```

That puts `best.pt` files at `weights/d3pm/best.pt` and
`weights/diffusion_lm/best.pt`.

## Inference

```bash
python predict.py -i data/example_input.txt -o out.txt --model d3pm
python predict.py -i data/example_input.txt -o out.txt --model diffusion_lm
```

Output format is identical to version 1, so the same calendar fallback in
`predict.py` cleans up anything the model gets wrong.

## Implementation notes

### D3PM (absorbing state)

I used the absorbing-state variant from the paper because it's the cleanest:
each token, at each step, has some probability of being replaced with `[MASK]`,
and once masked it stays masked. The schedule is cosine, with 100 timesteps.

The denoiser is a small Transformer encoder. Inputs go in as:

```
[cond_1, cond_2, cond_3, cond_4, time_token, x_t_1, ..., x_t_10]
```

So the conditions and the timestep show up as extra tokens at the front of
the sequence and the transformer can attend across them freely. The head
predicts a distribution over the 11 non-mask tokens at each date position.

Training loss is cross-entropy on `x_0` evaluated only at the positions
that were masked at the sampled timestep, which is the standard simple
loss for absorbing D3PM.

### Diffusion-LM (continuous)

Each token gets a learnable 32-dim embedding. The diffusion lives in that
32-dim space: standard DDPM forward,

```
x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
```

and the same Transformer-encoder architecture predicts `x_0` directly
(I prefer the x_0 parameterisation over the eps one because the rounding
step at the end works on `x_0` anyway).

Two-part loss:
- L_simple: MSE between predicted `x_0` and the true embedding
- L_round: cross-entropy from distance-based logits back to the original token

The L_round part is critical. Without it, nothing forces the predicted
`x_0` vectors to land near a real token embedding, and rounding produces
garbage. I excluded the `[MASK]` token from the rounding candidates with
an additive penalty so the inference output is never a mask.

Schedule is again cosine, 200 steps (continuous diffusion likes more steps).

## What works, what doesn't

These models are honest about the hard part of this task: predicting a day
of the month that's consistent with the requested weekday. With 7 weekdays,
random guessing gives ~14% accuracy on that condition, and a lot of training
runs end up close to that. The other three conditions (month, decade, leap)
are basically given by the inputs, so they're near 1.0 after a few epochs.

The calendar fallback in `predict.py` is what gets the final output to 100%
condition satisfaction. The diffusion models contribute structure and most
of the trivial conditions; the fallback contributes the weekday lookup that
the model can't reliably learn from data alone.

## Files

- `modal_train.py` - main entry point. Defines the Modal app, the image,
  the persistent volume, and the two training functions (one per model).
- `modal_smoke.py` - 30-second sanity check that imports the model code,
  runs one forward/backward pass on the GPU, and exits. Useful before
  committing to a long training run.
- `predict.py` - local inference with downloaded weights. Has the same
  calendar fallback as version 1.
- `model/utils/tokenizer.py` - fixed-length encoding with `[MASK]`.
- `model/utils/common.py` - shared time embedding and condition embedder.

## Test results

See the top-level `README.md` for the full comparison across all six
models (4 from version 1 + the 2 here).
