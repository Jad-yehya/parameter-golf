# Text Diffusion Hparam CPU Probes

Local-only hparam probes for `train_text_diffusion_mps.py` on branch
`pg-exp-textdiff-hparams`. No remote cluster was used.

## Setup

Worktree:

```text
/private/tmp/parameter-golf-worktrees/pg-exp-textdiff-hparams
```

Python:

```text
/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python
```

Import check:

```text
torch 2.11.0
torch.backends.mps.is_available() == False
```

The requested probes were therefore run on CPU with `DEVICE=cpu`.

Data:

```text
DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen
TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model
```

The screen dataset is a symlinked two-shard fixture:

```text
fineweb_train_000000.bin -> ../fineweb10B_sp1024/fineweb_train_000000.bin
fineweb_val_000000.bin -> ../fineweb10B_sp1024_val1m/fineweb_val_000000.bin
```

## Common Command

All runs used this shape and data budget unless the table lists a changed env
value:

```sh
env \
  DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen \
  TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model \
  DEVICE=cpu \
  COMPUTE_DTYPE=float32 \
  NUM_LAYERS=2 \
  MODEL_DIM=128 \
  NUM_HEADS=4 \
  MLP_MULT=2 \
  TRAIN_SEQ_LEN=256 \
  TRAIN_BATCH_TOKENS=2048 \
  MAX_TRAIN_TOKENS=262144 \
  MAX_VAL_TOKENS=262144 \
  VAL_SEQS=8 \
  ITERATIONS=80 \
  DIFFUSION_EVAL_STEPS=32 \
  WARMUP_STEPS=5 \
  WARMDOWN_STEPS=20 \
  SELF_CONDITION=0 \
  MASK_PATTERN=independent \
  TRAIN_LOG_EVERY=0 \
  OUT_DIR=logs_textdiff_hparam \
  RUN_ID=<run_id> \
  SEED=<seed> \
  <changed env values> \
  /Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python train_text_diffusion_mps.py
```

The exact stdout/file evidence is under `logs_textdiff_hparam/<run_id>.txt`
and `logs_textdiff_hparam/<run_id>.json`.

## Baseline

The baseline target from `text_diffusion_screen_results.md` is plain MDLM,
2L/128d, seq 256, 80 iterations, `DIFFUSION_EVAL_STEPS=32`, with this
three-seed mean:

| run | seed 0 | seed 42 | seed 314 | mean |
| --- | ---: | ---: | ---: | ---: |
| baseline plain MDLM | 3.27008804 | 3.32645576 | 3.28254391 | 3.29302924 |

The seed-42 baseline was rerun in this hparam log set as `baseline_s42` and
matched the prior result:

```text
final_var_bpb:3.32645576 bits_per_token:8.85051632 train_seconds:3.67
```

## Seed-42 Broad Sweep

All rows are `final_var_bpb` from 32-step eval. Lower is better.

| run_id | changed env values | seed 42 BPB |
| --- | --- | ---: |
| baseline_s42 | none | 3.32645576 |
| lr_2e4_s42 | `LR=2e-4` | 3.38112292 |
| lr_3e4_s42 | `LR=3e-4` | 3.33592464 |
| lr_1e3_s42 | `LR=1e-3` | 3.32758708 |
| lr_1p5e3_s42 | `LR=1.5e-3` | 3.32973627 |
| wd_0_s42 | `WEIGHT_DECAY=0` | 3.32643927 |
| wd_0p01_s42 | `WEIGHT_DECAY=0.01` | 3.32644084 |
| wd_0p3_s42 | `WEIGHT_DECAY=0.3` | 3.32648985 |
| warm0_down20_s42 | `WARMUP_STEPS=0 WARMDOWN_STEPS=20` | 3.32649344 |
| warm10_down20_s42 | `WARMUP_STEPS=10 WARMDOWN_STEPS=20` | 3.32680447 |
| warm5_down40_s42 | `WARMDOWN_STEPS=40` | 3.32277900 |
| warm5_down60_s42 | `WARMDOWN_STEPS=60` | 3.32033988 |
| warm5_down0_s42 | `WARMDOWN_STEPS=0` | 3.33083111 |
| bt1024_it160_s42 | `TRAIN_BATCH_TOKENS=1024 ITERATIONS=160` | 3.32262335 |
| bt4096_it40_s42 | `TRAIN_BATCH_TOKENS=4096 ITERATIONS=40` | 3.35314061 |
| eps_0p3_s42 | `NOISE_EPS=0.3` | 3.41707476 |
| eps_0p2_s42 | `NOISE_EPS=0.2` | 3.36751332 |
| eps_0p05_s42 | `NOISE_EPS=0.05` | 3.30552808 |
| eps_0p02_s42 | `NOISE_EPS=0.02` | 3.29614760 |
| eps_0p01_s42 | `NOISE_EPS=0.01` | 3.29011675 |
| eps_0p005_s42 | `NOISE_EPS=0.005` | 3.28604423 |
| eps_0p001_s42 | `NOISE_EPS=0.001` | 3.28522435 |
| eps_0p0001_s42 | `NOISE_EPS=0.0001` | 3.28379352 |
| eps_0p00001_s42 | `NOISE_EPS=0.00001` | 3.28353665 |
| eps_0_s42 | `NOISE_EPS=0` | 3.28358513 |

## Combination Sweep

The useful signal was low `NOISE_EPS`, then longer warmdown. LR and weight
decay did not help in the broad sweep.

| run_id | changed env values | seed 42 BPB |
| --- | --- | ---: |
| eps_0p02_lr3e4_s42 | `NOISE_EPS=0.02 LR=3e-4` | 3.30461483 |
| eps_0p02_lr1e3_s42 | `NOISE_EPS=0.02 LR=1e-3` | 3.29803840 |
| eps_0p02_warm5_down40_s42 | `NOISE_EPS=0.02 WARMDOWN_STEPS=40` | 3.29163245 |
| eps_0p02_bt1024_it160_s42 | `NOISE_EPS=0.02 TRAIN_BATCH_TOKENS=1024 ITERATIONS=160` | 3.28886540 |
| eps_0p02_bt4096_it40_s42 | `NOISE_EPS=0.02 TRAIN_BATCH_TOKENS=4096 ITERATIONS=40` | 3.32383303 |
| eps_0p001_warm5_down40_s42 | `NOISE_EPS=0.001 WARMDOWN_STEPS=40` | 3.28051140 |
| eps_0p001_warm5_down60_s42 | `NOISE_EPS=0.001 WARMDOWN_STEPS=60` | 3.27707712 |
| eps_0p001_warm5_down70_s42 | `NOISE_EPS=0.001 WARMDOWN_STEPS=70` | 3.27679019 |
| eps_0p001_warm5_down80_s42 | `NOISE_EPS=0.001 WARMDOWN_STEPS=80` | 3.27724764 |
| eps_0p001_warm0_down40_s42 | `NOISE_EPS=0.001 WARMUP_STEPS=0 WARMDOWN_STEPS=40` | 3.28016886 |
| eps_0p001_bt1024_it160_s42 | `NOISE_EPS=0.001 TRAIN_BATCH_TOKENS=1024 ITERATIONS=160` | 3.28109305 |
| eps_0p001_warm5_down60_lr1e3_s42 | `NOISE_EPS=0.001 WARMDOWN_STEPS=60 LR=1e-3` | 3.27805731 |
| eps_0p001_warm5_down60_lr3e4_s42 | `NOISE_EPS=0.001 WARMDOWN_STEPS=60 LR=3e-4` | 3.32524155 |
| eps_0p00001_warm5_down40_s42 | `NOISE_EPS=0.00001 WARMDOWN_STEPS=40` | 3.27876334 |
| eps_0p00001_warm5_down60_s42 | `NOISE_EPS=0.00001 WARMDOWN_STEPS=60` | 3.27531670 |
| eps_0p00001_warm5_down70_s42 | `NOISE_EPS=0.00001 WARMDOWN_STEPS=70` | 3.27503456 |
| eps_0p00001_warm5_down80_s42 | `NOISE_EPS=0.00001 WARMDOWN_STEPS=80` | 3.27549780 |
| eps_0_warm5_down40_s42 | `NOISE_EPS=0 WARMDOWN_STEPS=40` | 3.27881263 |
| eps_0_warm5_down60_s42 | `NOISE_EPS=0 WARMDOWN_STEPS=60` | 3.27536769 |

## Three-Seed Checks

Two near-tied candidates were checked. The best seed-42 row was
`NOISE_EPS=0.00001 WARMDOWN_STEPS=70`, but its three-seed mean was slightly
worse than `WARMDOWN_STEPS=60`.

| candidate | seed 0 | seed 42 | seed 314 | mean | delta vs 3.29302924 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline plain MDLM | 3.27008804 | 3.32645576 | 3.28254391 | 3.29302924 | 0.00000000 |
| `NOISE_EPS=0.00001 WARMDOWN_STEPS=60` | 3.22280514 | 3.27531670 | 3.22117460 | 3.23976548 | -0.05326376 |
| `NOISE_EPS=0.00001 WARMDOWN_STEPS=70` | 3.22404932 | 3.27503456 | 3.22188906 | 3.24032431 | -0.05270493 |

Exact best-mean commands:

```sh
env DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model DEVICE=cpu COMPUTE_DTYPE=float32 NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 MLP_MULT=2 TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=2048 MAX_TRAIN_TOKENS=262144 MAX_VAL_TOKENS=262144 VAL_SEQS=8 ITERATIONS=80 DIFFUSION_EVAL_STEPS=32 WARMUP_STEPS=5 WARMDOWN_STEPS=60 SELF_CONDITION=0 MASK_PATTERN=independent NOISE_EPS=0.00001 TRAIN_LOG_EVERY=0 OUT_DIR=logs_textdiff_hparam RUN_ID=best_eps1e5_warm60_seed0 SEED=0 /Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python train_text_diffusion_mps.py
env DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model DEVICE=cpu COMPUTE_DTYPE=float32 NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 MLP_MULT=2 TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=2048 MAX_TRAIN_TOKENS=262144 MAX_VAL_TOKENS=262144 VAL_SEQS=8 ITERATIONS=80 DIFFUSION_EVAL_STEPS=32 WARMUP_STEPS=5 WARMDOWN_STEPS=60 SELF_CONDITION=0 MASK_PATTERN=independent NOISE_EPS=0.00001 TRAIN_LOG_EVERY=0 OUT_DIR=logs_textdiff_hparam RUN_ID=best_eps1e5_warm60_seed42 SEED=42 /Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python train_text_diffusion_mps.py
env DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model DEVICE=cpu COMPUTE_DTYPE=float32 NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 MLP_MULT=2 TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=2048 MAX_TRAIN_TOKENS=262144 MAX_VAL_TOKENS=262144 VAL_SEQS=8 ITERATIONS=80 DIFFUSION_EVAL_STEPS=32 WARMUP_STEPS=5 WARMDOWN_STEPS=60 SELF_CONDITION=0 MASK_PATTERN=independent NOISE_EPS=0.00001 TRAIN_LOG_EVERY=0 OUT_DIR=logs_textdiff_hparam RUN_ID=best_eps1e5_warm60_seed314 SEED=314 /Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python train_text_diffusion_mps.py
```

## Takeaways

1. The best local CPU candidate is plain MDLM with `NOISE_EPS=0.00001` and
   `WARMDOWN_STEPS=60`. Its three-seed mean is `3.23976548`, improving the
   documented baseline mean by `0.05326376` BPB.
2. Most of the gain comes from lowering `NOISE_EPS`; longer warmdown adds a
   smaller but repeatable seed-42 gain.
3. LR and weight decay are not promising in this tiny screen. Weight decay was
   effectively flat around the baseline, and lower LR was clearly worse.
4. `TRAIN_BATCH_TOKENS=1024 ITERATIONS=160` helped seed 42 slightly at the
   default noise floor, but it was not competitive once low `NOISE_EPS` was
   enabled.

## Legality and Validation Risk

These runs do not use validation leakage in the training loop: the script samples
only the train shard during optimization and uses the val shard only for final
BPB measurement.

The main risk is interpretive rather than leakage-based. `NOISE_EPS` changes the
diffusion corruption and evaluation schedule. Very small epsilon values
substantially lower the local variational BPB in this harness, while making the
training loss scale much larger and not directly comparable to the baseline. I
would describe this setting explicitly in any submission notes and validate it
against the official scoring path before treating it as a leaderboard-quality
result.

I prefer `NOISE_EPS=0.00001` over exact zero despite similar seed-42 BPB, because
it avoids an exact-zero endpoint in the alpha/sigma schedule.
