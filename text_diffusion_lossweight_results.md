# Text Diffusion Loss-Weighting Screen

This branch continues the local text-diffusion probe from `pg-exp-text-diffusion`. It does not use Jean Zay, does not open an upstream PR, and is not official 8xH100 validation.

## Change

`train_text_diffusion_mps.py` now supports:

```bash
LOSS_WEIGHTING=mdlm              # original dsigma-weighted MDLM objective, default
LOSS_WEIGHTING=simple_mask_mean  # average CE over masked tokens only
```

The motivation is local-screen stability. The full MDLM objective is principled, but the `dsigma` factor can be high variance in tiny 80-step local screens. The simple masked-token CE is a diagnostic objective, not a claim that it should replace MDLM at full scale without validation.

## Common Setup

All runs are local CPU screens because this sandbox reports MPS as unavailable.

- data: `/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen`
- tokenizer: `/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model`
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
- `DIFFUSION_EVAL_STEPS=32`
- `WARMUP_STEPS=5`
- `WARMDOWN_STEPS=20`

Metric: `final_var_bpb` from the discrete variational eval.

## Three-Seed Results

| variant | seed 42 | seed 0 | seed 314 | mean | delta vs plain |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain MDLM, `NOISE_EPS=0.1` | 3.32645576 | 3.27008804 | 3.28254391 | 3.29302924 | 0.00000000 |
| simple CE, `NOISE_EPS=0.1` | 3.32161918 | 3.26974098 | 3.28182453 | 3.29106156 | -0.00196768 |
| simple CE, `NOISE_EPS=0.05` | 3.30094564 | 3.24603770 | 3.25019719 | 3.26572684 | -0.02730239 |
| simple CE, `NOISE_EPS=0.02` | 3.29112867 | 3.22826463 | 3.23175030 | 3.25038120 | -0.04264804 |
| simple CE, `NOISE_EPS=0.01` | 3.28550308 | 3.22182472 | 3.22851567 | 3.24528116 | -0.04774808 |

Best local result: `LOSS_WEIGHTING=simple_mask_mean NOISE_EPS=0.01`.

## One-Seed Side Probes

| variant | seed 42 |
| --- | ---: |
| simple CE, `LR=0.001`, `NOISE_EPS=0.1` | 3.32240707 |
| simple CE, `LR=0.0003`, `NOISE_EPS=0.1` | 3.33372160 |
| simple CE, `NOISE_EPS=0.2` | 3.36413760 |
| simple CE, `NOISE_EPS=0.005` | 3.28126053 |

The epsilon sweep is the important signal. Lowering from `0.1` to `0.01` helps, but `0.005` is worse on seed 42, so `0.01` is the current local choice.

## Example Command

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pg-textdiff-loss-pycache \
DEVICE=cpu \
COMPUTE_DTYPE=float32 \
DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen \
TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model \
OUT_DIR=logs_textdiff_loss \
RUN_ID=simpleloss_eps001_cpu32_seed42 \
SEED=42 \
NUM_LAYERS=2 \
MODEL_DIM=128 \
NUM_HEADS=4 \
MLP_MULT=2 \
TRAIN_SEQ_LEN=256 \
TRAIN_BATCH_TOKENS=2048 \
MAX_TRAIN_TOKENS=262144 \
MAX_VAL_TOKENS=262144 \
ITERATIONS=80 \
TRAIN_LOG_EVERY=0 \
VAL_SEQS=8 \
DIFFUSION_EVAL_STEPS=32 \
WARMUP_STEPS=5 \
WARMDOWN_STEPS=20 \
NOISE_EPS=0.01 \
MAX_WALLCLOCK_SECONDS=0 \
SELF_CONDITION=0 \
LOSS_WEIGHTING=simple_mask_mean \
/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python train_text_diffusion_mps.py
```

## Interpretation

This is the first local text-diffusion branch in this thread that clearly improves over the plain local MDLM screen. The result is still far from competitive with the AR local screens, and because the schedule changes the variational eval path too, it needs more scrutiny before being treated as a real submission direction.

Next useful checks:

- run a larger local model/screen to see whether the `NOISE_EPS=0.01` benefit survives scale
- compare 64/128 diffusion eval steps for the same trained variants
- combine this objective with any independent worker result that improves architecture or eval variance
