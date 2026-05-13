# Text Diffusion Tied Output Head Screen

This branch adds `TIE_OUTPUT_HEAD=1` to `train_text_diffusion_mps.py`. In tied mode, the diffusion output logits for real vocabulary tokens reuse `tok_emb.weight[:vocab_size]`; the mask-token column is a finite placeholder and remains excluded in `subs_log_probs`.

The tied projection is scaled by `1 / sqrt(model_dim)`. A first unscaled local diagnostic produced `6.28617927` BPB at seed 42, consistent with overly sharp random tied logits from the default embedding initialization.

## Local CPU Screen Settings

- `DEVICE=cpu`
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
- local screen data: `fineweb10B_sp1024_screen`

## Results

| variant | seed | final_var_bpb | bits/token | params | zlib state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain MDLM baseline | 42 | 3.32645576 | 8.85051632 | 804480 | 2807358 |
| tied output head, scaled | 42 | 3.54256788 | 9.42551386 | 665216 | 2438944 |

Plain MDLM 3-seed mean from the existing screen is `3.29302924` BPB.

The tied head saves `139264` parameters and `368414` compressed state bytes in this local model, but worsens seed-42 BPB by `+0.21611212` versus the matching plain seed-42 baseline.

## Conclusion

The tied output head is implemented and parameter-efficient, but this local seed-42 screen is not promising. I did not run the 3-seed screen because the first matching 32-step CPU result was clearly worse than both the plain seed-42 baseline and the plain 3-seed mean.
