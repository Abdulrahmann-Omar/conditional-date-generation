# Generative date models

GANs course, assignment 2. Given four conditions (day of week, month, leap-year
flag, decade), generate a date that satisfies all of them.

I built four models so I could compare them. Two come from the course (a
Conditional GAN, which the assignment requires, and a Conditional VAE). The
other two are seq2seq models that aren't in the course: a Transformer and an
LSTM. The Transformer is what `predict.py` calls by default because it was the
most consistent on the weekday check in my runs, but you can pick any of the
four with `--model`.

The thing that took me the longest wasn't actually the models. It was figuring
out a metric that made sense. More on that below.

## What's in here

```
.
├── data/
│   ├── data.txt              # ~146k labelled examples
│   └── example_input.txt     # 1464 unlabelled queries
├── model/
│   ├── predict.py            # inference entry point
│   ├── utils/                # tokenizer, dataset, CSR metric
│   ├── gan/                  # Conditional GAN
│   ├── vae/                  # Conditional VAE
│   ├── transformer/          # Seq2Seq Transformer
│   └── lstm/                 # Seq2Seq LSTM
├── environment.yml
└── README.md
```

Inside each model folder there's a `model.py` for the architecture, a
`train.py` that handles the training loop, and a `weights/` folder where
checkpoints land.

## Setup

```bash
conda env create -f environment.yml
conda activate dates_gen
```

No CUDA GPU? Strip the `pytorch-cuda` line out of `environment.yml` before
running the create command. Training on CPU works, it's just slow.

## Training

Run from inside `model/`:

```bash
python -m gan.train
python -m vae.train
python -m transformer.train
python -m lstm.train
```

Each epoch prints the validation **CSR** (condition satisfaction rate): the
share of generated dates that pass all four condition checks. The best
checkpoint by CSR ends up at `<model>/weights/best.pt`.

I went with CSR over accuracy because lots of different dates satisfy the same
condition set, so asking the model to recover the exact one from the training
data didn't really test what we care about. CSR tests the thing we actually
want: did the model produce a date that fits the conditions, regardless of
which one.

## Prediction

Once a model has a saved `best.pt`:

```bash
cd model
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt
```

To use a specific model instead of the default:

```bash
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt --model vae
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt --model gan
python predict.py -i ../data/example_input.txt -o ../outputs/preds.txt --model lstm
```

If the model produces something invalid (wrong weekday, April 31st, that kind
of thing), the script runs a small calendar search to find a date that does
fit the conditions and uses that instead. It prints how many times it had to
do this, which is a nice sanity check on the model's quality.

## Input format

One line per query, four bracketed tokens:

```
[DAY] [MONTH] [LEAP] [DECADE]
```

- `DAY`: `MON TUE WED THU FRI SAT SUN`
- `MONTH`: `JAN FEB MAR ... DEC`
- `LEAP`: `True` or `False`
- `DECADE`: the first three digits of the year (`180` covers 1800-1809,
  `220` covers 2200-2209)

### Sample input

The first ten lines of `data/example_input.txt`:

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

Same conditions, with a `d-m-yyyy` date stuck on the end. Day and month aren't
zero-padded.

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

Pick any line and check by hand. `1-1-1800` was a Wednesday in January, the
year isn't a leap year, the decade is 1800-1809. `1-1-2000` was a Saturday,
January, 2000 *is* a leap year (divisible by 400), still in the 2000s decade.
Both work.

## The models

### Conditional GAN

Noise vector plus four condition embeddings go in. Out come soft distributions
over day-of-month and the last digit of the year. The discriminator gets the
date and the conditions and scores how real it looks. Standard BCE loss,
label-smoothed reals at 0.9 to keep D from getting cocky. G updates twice per
D step because in early experiments D was running away with it otherwise.

I'm not asking the model to predict month or decade. The conditions already
pin those down, so the generator only learns the part that's actually
unknown: day and the trailing year digit. That cuts the output dimension way
down.

### Conditional VAE

Encoder takes the condition embedding plus the one-hot date and outputs
`(mu, logvar)`. Decoder takes a `z` sample plus the condition embedding and
reconstructs the date. Loss is cross-entropy on (day, year-digit) plus the
usual KL term. I anneal beta from 0 up to 1 over the first 30 epochs because
without that the KL just collapses the latent space and the decoder learns to
ignore `z`.

### Seq2Seq Transformer

Encoder side: the four conditions become a length-4 source sequence. Each
position has its own embedding table because the four vocabularies don't
overlap (7 weekdays, 12 months, 2 leap values, 41 decades). Decoder side:
generate the date one character at a time with greedy decoding at inference.
This one was the most reliable across runs, which is why it's the default in
`predict.py`.

### Seq2Seq LSTM

Same generation idea as the Transformer but the encoder is just a small MLP
that produces the initial `(h0, c0)` for a 2-layer LSTM decoder. Cheaper to
train than the Transformer. Slightly worse weekday CSR in my runs but not by
a huge amount.

## What CSR actually measures

Per epoch the validation loop reports six numbers:

- `valid`: the output parses as a real calendar date
- `day`: the weekday lines up with the condition
- `month`: the month lines up
- `leap`: leap-year flag lines up
- `decade`: decade lines up
- `all`: everything passes simultaneously. This is the headline number.

`month` and `decade` should sit near 1.0 because they come from the conditions
through the decoding step. The hard ones are `day` (weekday consistency, which
depends on the full calendar) and `all` (the conjunction).

## Random notes

The dataset spans 1800 to 2200, one entry per day, in chronological order.
That's where the 146k comes from. Shuffling happens inside the DataLoader, not
in the file.

Train/val split is 90/10 with a fixed seed so CSR numbers between runs are
roughly comparable.

For the seq2seq models the character vocabulary is digits, the `-` separator,
plus `BOS`, `EOS`, `PAD`. 14 tokens total.

The calendar fallback in `predict.py` is a safety net, not the main thing
doing the work. A healthy model should only fire it on a handful of edge
cases. If you see it firing a lot, training probably didn't go well.
