# Generative date models

GANs course, assignment 2.

Given four conditions:

- day of the week (`MON`..`SUN`)
- month (`JAN`..`DEC`)
- leap-year flag (`True`/`False`)
- decade (the first three digits of the year)

generate a date `d-m-yyyy` that satisfies all of them.

The repo is split in two folders. Version 1 has the four models the
course actually asked for. Version 2 adds two diffusion models I wanted
to try and was curious whether they'd beat the rest.

```
.
├── version(1)/   GAN, VAE, Seq2Seq Transformer, Seq2Seq LSTM
├── version(2)/   D3PM and Diffusion-LM (trained on Modal.com)
└── Assignment.md
```

Both versions can train on Modal. Version 1 also has a local training
script (`python -m gan.train` and friends) in case you want to grind on
CPU instead.

## Results

All numbers below are best-epoch validation **Condition Satisfaction
Rate** (CSR): the fraction of generated dates that pass each condition
check. `all` is the fraction that pass every check simultaneously, which
is the metric I actually care about.

Trained on a 90/10 train/val split of the 146k examples in `data.txt`.
v1 models ran 20 epochs on an A10G; v2 models ran 25 epochs on an A10G.

<!-- RESULTS_TABLE_START -->

| Model                | valid | weekday | month | leap  | decade | **all**   |
|----------------------|------:|--------:|------:|------:|-------:|----------:|
| GAN (v1)             | 0.998 |   0.145 | 0.998 | 0.778 |  0.998 | **0.116** |
| VAE (v1)             | 0.980 |   0.144 | 0.980 | 0.755 |  0.980 | **0.111** |
| Transformer (v1)     | 1.000 |   0.149 | 1.000 | 1.000 |  1.000 | **0.149** |
| LSTM (v1)            | 1.000 |   0.151 | 1.000 | 1.000 |  1.000 | **0.151** |
| D3PM (v2)            | 0.992 |   0.150 | 0.992 | 0.991 |  0.992 | **0.150** |
| Diffusion-LM (v2)    | 0.988 |   0.145 | 0.988 | 0.961 |  0.988 | **0.141** |

<!-- RESULTS_TABLE_END -->

A few things jump out from this table. `month`, `decade`, `leap`, and
`valid` get most of the way to 1.0 for every model, because those
conditions are basically given by the inputs and the model just has to
not mess up the decoding step.

The weekday column is the one I actually care about, and it sits stuck
around 0.14-0.15 for every single model. That's not coincidence: random
chance on a 7-way weekday is 1/7 ≈ 0.143. None of these architectures
genuinely learns the calendar. They learn how a date string looks, they
nail the three easy conditions, and then they guess the day. So the
weekday CSR matches what you'd get from picking a day uniformly.

D3PM did beat the v1 sequence models very slightly (0.150 vs ~0.149).
Diffusion-LM was a touch worse (0.141). Within noise, honestly. The big
takeaway is that none of the six approaches escape the random-guess
floor on the calendar lookup.

That's the reason `predict.py` has a calendar fallback at the end:
the models give you a well-formed date with three correct conditions,
and the fallback fixes the weekday before anything gets written out.

## Input format

One line per query. Four bracketed tokens:

```
[DAY] [MONTH] [LEAP] [DECADE]
```

Concrete sample (the first ten lines of `data/example_input.txt`):

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

Same line, with a `d-m-yyyy` date appended (no zero-padding on day/month):

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

You can verify by hand: `1-1-1800` was a Wednesday, January, 1800 was
not a leap year, lands in decade 180. `1-1-2000` was a Saturday, January,
2000 *is* a leap year (divisible by 400), decade 200. Both check out.

## Setup

### Local environment (version 1)

```bash
cd "version(1)"
conda env create -f environment.yml
conda activate dates_gen
```

If you don't have a CUDA GPU, remove the `pytorch-cuda` line from
`environment.yml`. Training on CPU works but takes a while.

### Modal (both versions)

```bash
pip install modal
modal setup
```

That's all. Modal handles the GPU, Python, container image, and the
persistent volume for checkpoints.

## Training

### Version 1 (local or Modal)

Local:

```bash
cd "version(1)/model"
python -m gan.train
python -m vae.train
python -m transformer.train
python -m lstm.train
```

Modal (all four in parallel on A10G GPUs):

```bash
cd "version(1)"
modal run modal_train.py --model all --epochs 25
```

### Version 2 (Modal only)

```bash
cd "version(2)"
modal run modal_train.py --model both --epochs 25
```

Checkpoints land on Modal volumes named `date-models-v1-weights` and
`date-diffusion-weights` respectively. To pull a checkpoint back:

```bash
modal volume get date-models-v1-weights      /gan/best.pt  ./gan_best.pt
modal volume get date-diffusion-weights      /d3pm/best.pt ./d3pm_best.pt
```

## Inference

After downloading the relevant checkpoint:

```bash
# version 1
cd "version(1)/model"
python predict.py -i ../data/example_input.txt -o ../out.txt --model transformer

# version 2
cd "version(2)"
python predict.py -i data/example_input.txt -o out.txt --model d3pm
python predict.py -i data/example_input.txt -o out.txt --model diffusion_lm
```

If the model generates an invalid date or a wrong weekday, the script
runs a small calendar search to find a valid date matching the
conditions and uses that. It prints how many predictions had to be
fixed up, which doubles as a sanity check on model quality.

## CLOs

The encouraging part is that every model picks up the structure of a
date string surprisingly fast. The Transformer and LSTM hit 1.0 on
`valid` after a few epochs because they emit characters one at a time
and the format is easy. The GAN and VAE sit slightly below 1.0 because
they predict `(day, year_digit)` separately and occasionally produce a
day-of-month that doesn't exist (April 31, that kind of thing). The two
diffusion models behave like the seq2seq ones: high `valid`, high
`month`, high `decade`, all clearly thanks to the conditions being fed
in at every step.

The discouraging part is that the weekday never really moves. I tried
both the obvious thing (longer training) and the not-so-obvious thing
(bigger model, different schedule for D3PM), and the curve always
plateaus at 1/7. There's no signal in the data the way the conditions
encode it, because the model would need to internalise the Gregorian
calendar to do better. With 146k examples and no calendar inductive
bias, this just doesn't happen.

So practically, the calendar fallback in `predict.py` is what makes
the final output usable. The model takes care of three out of four
conditions; the fallback handles the weekday lookup. If a model is
actually trained, the fallback fires on roughly 85% of predictions
(because the weekday is wrong 6/7 of the time). If a model is broken
or untrained, it fires on 100%.

## Folder summaries

See `version(1)/README.md` for the four base models, and
`version(2)/README.md` for the diffusion models. Both folder READMEs
have implementation notes that are too detailed for this top-level
overview.
