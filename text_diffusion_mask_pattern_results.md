# Text Diffusion Mask Pattern Screen

Local-only screen from branch `pg-exp-textdiff-mask-patterns`. No Jean Zay, no remote
cluster, no upstream PR.

## Setup

Baseline and new runs use the same small CPU protocol as `text_diffusion_screen_results.md`:

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
- local `fineweb10B_sp1024_screen`

New logs are in `logs_textdiff_mask_patterns/`.

## Mask Patterns Added

- `multi_span`: splits the target masked budget across multiple short spans.
- `segment_blocks`: uses fewer, longer random blocks as a sentence-ish segment proxy.
- `boundary_biased`: increases mask probability on inferred punctuation-boundary token
  IDs and the following token. The default inferred IDs on the local SP1024 tokenizer
  were `[475, 702, 853, 960, 987, 1009, 1013, 1015]`.

## Results

Metric: `final_var_bpb`, lower is better. Baseline rows are copied from
`text_diffusion_screen_results.md`; new rows are from this branch.

| variant | seed 42 | seed 0 | seed 314 | mean | delta vs independent |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent baseline | 3.32645576 | 3.27008804 | 3.28254391 | 3.29302924 | 0.00000000 |
| span baseline, `SPAN_LEN=4` | 3.29701645 | 3.31029382 | 3.29175638 | 3.29968888 | +0.00665964 |
| `multi_span`, `SPAN_LEN=4` | 3.28947269 | 3.30310159 | 3.29461371 | 3.29572933 | +0.00270009 |
| `segment_blocks`, `SPAN_LEN=4` | 3.29624007 | 3.29577715 | 3.28891460 | 3.29364394 | +0.00061470 |

Seed-42-only boundary checks:

| variant | seed 42 | note |
| --- | ---: | --- |
| `boundary_biased`, `BOUNDARY_MASK_BIAS=4.0` | 3.54735826 | clearly worse |
| `boundary_biased`, `BOUNDARY_MASK_BIAS=1.0` | 3.42562766 | still clearly worse |

## Conclusion

The text-specific patterns did not beat independent masking on the 3-seed 32-step CPU
screen. `segment_blocks` is the closest result, only `+0.00061470` BPB behind the
independent mean, but it is not an improvement. `multi_span` looked promising on seed
42, then regressed on seeds 0 and 314. Boundary-biased masking is not worth extending
in this form.

Current recommendation: keep independent masking as the local best and do not promote
these mask patterns without a stronger idea, such as coupling segment masks to a
different denoising objective or evaluation schedule.

## Verification

- `python -m unittest test_text_diffusion_mps.py`
- `python -m py_compile train_text_diffusion_mps.py test_text_diffusion_mps.py`
- seed-42 screen for `multi_span`, `segment_blocks`, and `boundary_biased`
- 3-seed screen for `multi_span` and `segment_blocks`
