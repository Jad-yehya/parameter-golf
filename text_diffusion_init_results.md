# Text Diffusion Initialization Screen

This branch tests whether the local diffusion harness was being hurt by AR-style zero initialization.

The existing local harness zero-initializes the diffusion head and residual projections. The older MDLM record used normal PyTorch linear initialization, so this branch adds:

- `ZERO_INIT_HEAD=0`
- `ZERO_INIT_RESIDUAL=0`

## Seed-42 Results

Common setup matches the 32-step local CPU screen:

- 2 layers, 128 dim, 4 heads
- seq len 256
- 80 iterations
- `DIFFUSION_EVAL_STEPS=32`
- local `fineweb10B_sp1024_screen`

| variant | seed 42 final_var_bpb |
| --- | ---: |
| default zero init | 3.32645576 |
| `ZERO_INIT_HEAD=0 ZERO_INIT_RESIDUAL=1` | 3.32717756 |
| `ZERO_INIT_HEAD=1 ZERO_INIT_RESIDUAL=0` | 3.32656660 |
| `ZERO_INIT_HEAD=0 ZERO_INIT_RESIDUAL=0` | 3.32775769 |

## Conclusion

This is not a lead. Default zero initialization remains slightly best on this local seed-42 screen.
