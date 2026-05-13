# Text Diffusion Loss + Schedule Combination Screen

This branch combines two independently useful local text-diffusion changes:

- `LOSS_WEIGHTING=simple_mask_mean`
- alternate `NOISE_SCHEDULE` options from the schedule branch

It is local CPU screening only. No Jean Zay, no remote cluster, and no upstream PR.

## Common Setup

- local `fineweb10B_sp1024_screen`
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
- `LOSS_WEIGHTING=simple_mask_mean`
- `NOISE_EPS=0.01`

## Results

| variant | seed 42 | seed 0 | seed 314 | mean |
| --- | ---: | ---: | ---: | ---: |
| loglinear schedule | 3.28550308 | 3.22182472 | 3.22851567 | 3.24528116 |
| cosine schedule | 3.25314824 | - | - | - |
| power schedule, `NOISE_POWER=2.0` | 3.24413077 | 3.24602009 | 3.24620334 | 3.24545140 |
| loglinear schedule, `NOISE_EPS=0.00001`, `WARMDOWN_STEPS=60` | 3.27515881 | 3.22082567 | 3.21929975 | 3.23842808 |
| power schedule, `NOISE_EPS=0.00001`, `WARMDOWN_STEPS=60` | 3.23460642 | 3.24204677 | 3.24046431 | 3.23903917 |

Power schedule is much better than loglinear on seed 42, but the 3-seed mean is effectively tied and slightly worse than the current loglinear simple-loss branch. This remains true after applying the lower-epsilon/longer-warmdown setting.

## Conclusion

Do not replace the current lead with this branch. The best local text-diffusion setting remains:

```bash
LOSS_WEIGHTING=simple_mask_mean NOISE_EPS=0.00001 WARMDOWN_STEPS=60 NOISE_SCHEDULE=loglinear
```

The power schedule may still be useful if we care about variance or if it combines better with a larger model, but this small screen does not justify making it the default.
