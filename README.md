# Generative Date Models

GANs course assignment 2. The task: given four conditions (day of week, month,
leap year flag, decade), generate a date that satisfies all of them.

I trained four different models for this and put them side by side so the
comparison is easy. Two of them are from the course (a Conditional GAN and a
Conditional VAE). The other two are seq2seq models (a Transformer and an LSTM).
The Transformer is what `predict.py` uses by default since it was the most
reliable in my runs, but any of the four can be selected with `--model`.

## What's in the repo

```
.
├── data/
│   ├── data.txt              # ~146k training examples
│   └── example_input.txt     # 1464 example conditions (no labels)
├── model/
│   ├── predict.py            # inference entry point
│   ├── utils/                # tokenizer, dataset, evaluation
│   ├── gan/                  # Conditional GAN
│   ├── vae/                  # Conditional VAE
│   ├── transformer/          # Seq2Seq Transformer
│   └── lstm/                 # Seq2Seq LSTM
├── environment.yml
└── README.md
```

Each model folder has a `model.py` (the architecture), a `train.py` (the
training loop) and a `weights/` directory where checkpoints get saved.

## Setup

```bash
conda env create -f environment.yml
conda activate dates_gen
```

If you don't have a CUDA GPU, remove the `pytorch-cuda` line from
`environment.yml` before creating the env. Training on CPU is slow but works.

## Training

Run from the `model/` directory. Each model has its own training script.

```bash
cd model
python -m gan.train          # Conditional GAN
python -m vae.train          # Conditional VAE
python -m transformer.train  # Seq2Seq Transformer
python -m lstm.train         # Seq2Seq LSTM
```

Every epoch the script computes the validation **Condition Satisfaction Rate**:
the percentage of generated dates that pass all four condition checks. The best
checkpoint by CSR gets saved to `<model>/weights/best.pt`.

I went with CSR instead of accuracy because there are many valid dates per
condition set. Asking the model to reproduce the exact one from the dataset
felt like the wrong metric.

## Prediction

Once you have a `best.pt` for a model, run:

```bash
cd model
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt
```

To use a specific model:

```bash
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt --model vae
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt --model gan
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt --model lstm
```

If the model ever returns something invalid (a wrong weekday, day 31 in April,
etc.) the script falls back to a small calendar search that finds a valid date
matching the conditions. The fallback only fires when the model fails, and the
script prints how many times it kicked in.

## Input format

One line per query, four conditions wrapped in brackets:

```
[DAY] [MONTH] [LEAP] [DECADE]
```

- `DAY` is one of `MON TUE WED THU FRI SAT SUN`
- `MONTH` is one of `JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC`
- `LEAP` is `True` or `False`
- `DECADE` is the first three digits of the year (`180` for 1800-1809,
  `220` for 2200-2209)

### Sample input (from `data/example_input.txt`)

```
[WED] [JAN] [False] [180]
[MON] [JAN] [False] [190]
[SAT] [JAN] [True] [200]
[FRI] [JAN] [False] [210]
[WED] [JAN] [False] [220]
[WED] [FEB] [False] [189]
[MON] [FEB] [False] [199]
[SUN] [FEB] [False] [209]
[FRI] [FEB] [False] [219]
[TUE] [MAR] [False] [189]
```

## Output format

Same four bracketed tokens, with a `d-m-yyyy` date appended (no zero-padding on
day or month).

### Sample output

```
[WED] [JAN] [False] [180] 1-1-1800
[MON] [JAN] [False] [190] 1-1-1900
[SAT] [JAN] [True] [200] 1-1-2000
[FRI] [JAN] [False] [210] 1-1-2100
[WED] [JAN] [False] [220] 1-1-2200
[WED] [FEB] [False] [189] 5-2-1890
[MON] [FEB] [False] [199] 5-2-1990
[SUN] [FEB] [False] [209] 5-2-2090
[FRI] [FEB] [False] [219] 5-2-2190
[TUE] [MAR] [False] [189] 4-3-1890
```

You can verify any line by hand:

- `1-1-1800` was a Wednesday, January, not a leap year, falls in the 1800s.
- `1-1-2000` was a Saturday, January, 2000 was a leap year, falls in the 2000s.

Both check out.

## How the models work

### Conditional GAN

Generator takes a noise vector plus the four condition embeddings and outputs
soft distributions over day-of-month and the last digit of the year. The
discriminator sees the date and the conditions and decides whether the pair
looks real. Standard BCE loss with a bit of label smoothing on the real side.
G trains twice per D step because D tends to overpower it otherwise.

I didn't predict month or decade explicitly because the conditions already give
them, so the model only has to learn what day and what year-digit work
together for a given weekday/leap/month/decade combo.

### Conditional VAE

Encoder takes the condition embedding and the one-hot date and produces
`(mu, logvar)`. Decoder takes a latent sample plus the condition embedding and
reconstructs the date. Loss is cross-entropy on the (day, year-digit) pair plus
KL divergence with beta annealing over the first 30 epochs (otherwise the KL
collapses the latent space early and the model just ignores `z`).

### Seq2Seq Transformer

Encoder takes the four condition tokens as a length-4 sequence (each position
has its own embedding table since the vocabularies differ). Decoder generates
the date one character at a time. Standard teacher forcing during training,
greedy decoding at inference. This was the cleanest performer.

### Seq2Seq LSTM

Same idea as the Transformer but the encoder is just a small MLP that turns the
four conditions into the initial hidden state for a 2-layer LSTM decoder. It
generates the date character by character.

## Evaluation

The CSR metric reports six numbers per epoch:

- `valid`  - the string parses as a real date
- `day`    - the weekday matches
- `month`  - the month matches
- `leap`   - the leap-year flag matches
- `decade` - the decade matches
- `all`    - every condition passes (this is the headline number)

`month` and `decade` should be near 1.0 since they come straight from the
conditions through the decoding step. The hard parts are `day` (weekday
consistency) and the joint `all`.

## Notes

- Dataset covers years 1800 to 2200. About 146k samples, one per day in that
  range, in chronological order. I shuffle inside the DataLoader.
- I use a 90/10 train/val split with a fixed seed so the metric numbers are
  comparable between runs.
- The `BOS`/`EOS`/`PAD` tokens for the seq2seq models live in the char
  vocabulary alongside digits and the `-` separator. Total vocab size is 14.

## Author

Built for the GANs course assignment at Zewail City of Science and Technology.
