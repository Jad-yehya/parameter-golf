# Text Diffusion Local Screen

This branch tests a local masked-diffusion direction for Parameter Golf. It is not an upstream submission and it is not official 8xH100 validation.

## Existing Diffusion Baseline In Repo

The repo already contains a strong MDLM diffusion non-record submission:

- `records/track_non_record_16mb/2026-03-29_LLaDA_MDLM_Diffusion`
- bidirectional MDLM with adaLN timestep conditioning and discrete absorbing-mask ELBO eval
- reported `val_var_bpb=1.1465` on 2xH100

This branch therefore does not claim text diffusion as new. It adds a small local harness to test diffusion-style variants quickly with exact Parameter Golf shard loading and SentencePiece byte accounting.

## Harness

Script: `train_text_diffusion_mps.py`

Key properties:

- single-process PyTorch path for CPU, MPS, or CUDA
- exact FineWeb shard header reader
- exact SentencePiece byte LUT accounting, not a constant bytes/token approximation
- bidirectional non-causal transformer
- MDLM log-linear absorbing-mask loss
- optional self-conditioning
- optional span corruption
- discrete variational BPB screen with configurable diffusion steps

The current sandbox reports `torch.backends.mps.is_available() == False`, so these screens were run on CPU. No Jean Zay or remote cluster was used.

Common CPU screen settings:

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
- local `fineweb10B_sp1024_screen`

## 32-Step Eval Results

Metric: `final_var_bpb` with `DIFFUSION_EVAL_STEPS=32`.

| variant | seed 42 | seed 0 | seed 314 | mean |
| --- | ---: | ---: | ---: | ---: |
| plain MDLM | 3.32645576 | 3.27008804 | 3.28254391 | 3.29302924 |
| span MDLM, span_len=4 | 3.29701645 | 3.31029382 | 3.29175638 | 3.29968888 |

Span masking is slightly worse than plain independent masking in this local screen: `+0.00665965` BPB on the 3-seed mean.

## 8-Step Side Screen

Metric: `final_var_bpb` with `DIFFUSION_EVAL_STEPS=8`.

| variant | seed 42 | seed 0 | seed 314 | mean |
| --- | ---: | ---: | ---: | ---: |
| plain MDLM | 3.29417767 | 3.31458530 | 3.29973322 | 3.30283206 |
| self-conditioned MDLM | 3.35268916 | 3.31857256 | 3.34166627 | 3.33764266 |
| span MDLM, span_len=4 | 3.30143069 | 3.31047904 | 3.29523743 | 3.30238239 |

Self-conditioning is clearly negative at this scale: `+0.03481060` BPB versus plain MDLM on the 3-seed mean.

Span masking was nearly tied at 8 eval steps, but the 32-step rerun moved it behind plain MDLM.

## Interpretation

The working harness is useful, but neither tested variant is a lead:

- Plain independent-mask MDLM remains the strongest of these local screens.
- Self-conditioning adds parameters and a second denoising pass but hurts BPB.
- Span corruption is not robustly better, and the stronger 32-step eval makes it slightly worse.

The next text-diffusion idea should probably target evaluation or parameter efficiency rather than masking pattern alone. Good candidates are fewer-step distillation, asymmetric denoising heads, or hybrid AR-prefix scoring around diffusion spans.

## Verification

- `python -m py_compile train_text_diffusion_mps.py test_text_diffusion_mps.py`
- `python -m unittest test_text_diffusion_mps.py`
- local smoke run: `smoke_textdiff_cpu`
- local three-seed screens listed above
