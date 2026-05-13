# Text Diffusion Noise Schedule Screen

Branch: `pg-exp-textdiff-schedules`

This branch adds optional MDLM noise schedules controlled by `NOISE_SCHEDULE`.
Default behavior remains `NOISE_SCHEDULE=loglinear`, matching the existing
`log_linear_noise()` alpha and sigma path.

## Schedules

- `loglinear`: existing default, `alpha(t) = 1 - (1 - eps) t`
- `cosine`: `alpha(t) = eps + (1 - eps) cos(pi t / 2)^2`
- `power`: `alpha(t) = 1 - (1 - eps) t^p`, with `NOISE_POWER=2.0` by default

All schedules use `sigma = -log(alpha)`. The MDLM loss uses the analytic
`d sigma / dt`, and eval uses the same monotone alpha grid for the absorbing
mask ELBO reveal probabilities.

## Local CPU Screen Settings

The local environment reports `torch.backends.mps.is_available() == False`, so
these screens used `DEVICE=cpu`.

- `DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen`
- `TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model`
- `NUM_LAYERS=2`
- `MODEL_DIM=128`
- `NUM_HEADS=4`
- `MLP_MULT=2`
- `TRAIN_SEQ_LEN=256`
- `TRAIN_BATCH_TOKENS=2048`
- `MAX_TRAIN_TOKENS=262144`
- `MAX_VAL_TOKENS=262144`
- `ITERATIONS=80`
- `VAL_SEQS=8`
- `WARMUP_STEPS=5`
- `WARMDOWN_STEPS=20`
- `DIFFUSION_EVAL_STEPS=32`

Baseline for comparison: plain MDLM 32-step 3-seed mean `3.29302924`.

## Seed 42 Sweep

| run_id | schedule | seed | final_var_bpb | bits_per_token | train_loss | train_seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sched_loglinear_seed42` | loglinear | 42 | 3.32645576 | 8.85051632 | 9.20482862 | 4.41 |
| `sched_cosine_seed42` | cosine | 42 | 3.29386744 | 8.76381040 | 9.03502114 | 8.52 |
| `sched_power_seed42` | power | 42 | 3.31398527 | 8.81733680 | 9.32507070 | 9.72 |

Cosine was the best seed-42 schedule.

## Cosine 3-Seed Follow-Up

| run_id | seed | final_var_bpb | bits_per_token | train_loss | train_seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sched_cosine_seed0` | 0 | 3.26120524 | 8.67690790 | 7.91559970 | 5.07 |
| `sched_cosine_seed42` | 42 | 3.29386744 | 8.76381040 | 9.03502114 | 8.52 |
| `sched_cosine_seed314` | 314 | 3.28094748 | 8.72943497 | 8.47461052 | 5.15 |
| mean | - | 3.27867339 | 8.72338442 | 8.47507712 | 6.25 |

Cosine beats the plain MDLM 32-step mean by `0.01435585` BPB on this local
screen: `3.27867339` versus `3.29302924`.

## Verification

- `/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python -m unittest test_text_diffusion_mps.py`
- local CPU screen logs and JSON outputs in `logs_textdiff_schedule/`
