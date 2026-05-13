# Text Diffusion Eval Efficiency Screen

This branch tests cheaper diffusion evaluation masks for `train_text_diffusion_mps.py`. It is a local CPU screen only. No remote cluster was used.

## Change

`EVAL_MASK_MODE` controls evaluation masks only:

- `random`: existing stochastic eval path.
- `grid`: deterministic per-step, per-row low-discrepancy mask grid.
- `antithetic`: deterministic grid plus reversed-grid companion mask, averaged per eval step.

Training-time corruption still uses `MASK_PATTERN`; the new setting is only passed into `variational_elbo_bits`.

## Screen Setup

Data was read from the local cached SP1024 FineWeb shard already present at:

`/private/tmp/parameter-golf-worktrees/pg-mps-candidate/data/datasets/fineweb10B_sp1024`

Common settings:

```bash
DEVICE=cpu
COMPUTE_DTYPE=float32
NUM_LAYERS=2
MODEL_DIM=128
NUM_HEADS=4
MLP_MULT=2
TRAIN_SEQ_LEN=256
TRAIN_BATCH_TOKENS=2048
MAX_TRAIN_TOKENS=262144
MAX_VAL_TOKENS=262144
ITERATIONS=80
VAL_SEQS=8
WARMUP_STEPS=5
WARMDOWN_STEPS=20
MASK_PATTERN=independent
```

For each `EVAL_MASK_MODE`, I ran seeds `0`, `42`, and `314` at `DIFFUSION_EVAL_STEPS=8,16,32`. Because this harness does not persist checkpoints, "same run" comparison means the same deterministic CPU training seed and hyperparameters were rerun while changing only eval mode and step count.

## Results

Lower BPB is better.

| eval mask | steps | seed 0 | seed 42 | seed 314 | mean | stdev | MAE vs same-mode 32-step | preserves same-mode 32-step seed ranking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random | 8 | 3.31458530 | 3.29417767 | 3.29973322 | 3.30283206 | 0.00861471 | 0.03132155 | no |
| random | 16 | 3.28917138 | 3.32467217 | 3.28311414 | 3.29898590 | 0.01833050 | 0.00714572 | no |
| random | 32 | 3.27008804 | 3.32645576 | 3.28254391 | 3.29302924 | 0.02417694 | 0.00000000 | yes |
| grid | 8 | 3.27404586 | 3.29024902 | 3.28447235 | 3.28292241 | 0.00670509 | 0.00573470 | yes |
| grid | 16 | 3.28147398 | 3.29783095 | 3.29176005 | 3.29035500 | 0.00675121 | 0.01316728 | yes |
| grid | 32 | 3.26843627 | 3.28476609 | 3.27836077 | 3.27718771 | 0.00671803 | 0.00000000 | yes |
| antithetic | 8 | 3.28283609 | 3.29772916 | 3.29132661 | 3.29063062 | 0.00609996 | 0.00820845 | yes |
| antithetic | 16 | 3.28227473 | 3.29708903 | 3.29109560 | 3.29015312 | 0.00608452 | 0.00773096 | yes |
| antithetic | 32 | 3.27460811 | 3.28934369 | 3.28331469 | 3.28242216 | 0.00604879 | 0.00000000 | yes |

Against the original `random` 32-step estimates as a common reference, the low-step deterministic modes also preserve the seed ranking `(0, 314, 42)`, while random 8-step and 16-step do not:

| eval mask | steps | MAE vs random 32-step | preserves random 32-step seed ranking |
| --- | ---: | ---: | --- |
| random | 8 | 0.03132155 | no |
| random | 16 | 0.00714572 | no |
| grid | 8 | 0.01403100 | yes |
| grid | 16 | 0.01640896 | yes |
| antithetic | 8 | 0.01675245 | yes |
| antithetic | 16 | 0.01670170 | yes |

## Conclusion

The deterministic grid mode is the best cheap screen in this local setup. At 8 eval steps it preserves the 32-step ranking and has the lowest same-mode MAE to 32-step (`0.00573470` BPB), while stochastic random 8-step reverses the ranking and is much farther from 32-step (`0.03132155` BPB).

Antithetic mode also preserves ranking and gives stable seed spread, but it doubles the eval batch per step and does not beat grid 8-step on closeness to its own 32-step estimate.

Recommended default for cheap local screens: `EVAL_MASK_MODE=grid DIFFUSION_EVAL_STEPS=8`. Keep `EVAL_MASK_MODE=random` as the default behavior for compatibility.

## Verification

```bash
/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python -B -m unittest test_text_diffusion_mps.py
/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python -B -m py_compile train_text_diffusion_mps.py test_text_diffusion_mps.py
```
