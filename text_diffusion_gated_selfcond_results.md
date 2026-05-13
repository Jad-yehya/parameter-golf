# Text Diffusion Gated Self-Conditioning Screen

This branch tests a gated, optional low-rank self-conditioning path for the local
text-diffusion harness. It is a local CPU screen only, not an official Parameter
Golf validation result. No Jean Zay or remote cluster was used.

## Change

`SELF_CONDITION=1` now routes previous denoiser logits through a scalar sigmoid
gate initialized by `SELF_COND_GATE_INIT` and through either:

- full-rank projection when `SELF_COND_RANK=0`
- low-rank down/up projection when `SELF_COND_RANK>0`

The screened candidate used `SELF_COND_GATE_INIT=-4` and `SELF_COND_RANK=16`.
This initializes the active self-conditioning multiplier near `0.018`, so the
second denoising pass starts weak instead of dominating the token embedding.

## Screen Settings

Same local CPU screen settings as `text_diffusion_screen_results.md`:

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
- local `fineweb10B_sp1024_screen`

## Results

Metric: `final_var_bpb` with 32-step variational eval.

| variant | seed 42 | seed 0 | seed 314 | mean | delta vs plain MDLM mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain MDLM baseline | 3.32645576 | 3.27008804 | 3.28254391 | 3.29302924 | 0.00000000 |
| gated self-cond, rank 16, gate -4 | 3.23617167 | 3.32175673 | 3.29664605 | 3.28485815 | -0.00817109 |

Per-run logs and JSON summaries are in `logs_textdiff_gated_selfcond/`.

The final learned gate stayed close to its initial value:

| seed | final sigmoid gate | zlib state bytes |
| ---: | ---: | ---: |
| 42 | 0.01829417 | 2,874,707 |
| 0 | 0.01828749 | 2,875,750 |
| 314 | 0.01828633 | 2,868,288 |

## Interpretation

This is a small but positive local screen versus the plain MDLM 32-step mean.
The result is noisy: seed 42 improved strongly while seed 0 was worse than its
plain baseline. Still, unlike the earlier full self-conditioning 8-step screen
that was `+0.03481060` BPB worse, the gated low-rank path is not obviously
destructive and has a modest 3-seed mean win in this local setup.

The most likely interpretation is that the weak gate avoids early overuse of a
poor auxiliary denoising pass, while the low-rank path limits parameter and
optimization burden. This is worth a slightly stronger follow-up screen, but it
is not enough evidence for a record-facing direction by itself.

## Verification

- `/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python -m unittest test_text_diffusion_mps.py`
- `/Users/jadyehya/miniforge3/envs/parameter-golf-mps/bin/python -m py_compile train_text_diffusion_mps.py test_text_diffusion_mps.py`
