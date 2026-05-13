# Local MPS Screen: Random-Map Adapter + Causal Conv Memory

This is a local screening package, not official leaderboard validation. It packages the best local MPS architecture screen as a record-style folder so the candidate has a real `train_gpt.py` entrypoint before any upstream PR is considered.

## Summary

The tested variant combines:

- AttnOutGate on attention outputs.
- QK gain initialized to `5.0`.
- all-layer lightweight inter-head attention mixing.
- fixed random sinusoidal residual adapters after transformer blocks.
- a lightweight residual causal depthwise convolution memory path.

The broad ingredients overlap prior submissions and README requests, but this exact placement and composition was not found in the repo records during the local search.

## Local MPS Screen Setup

All three runs used a compact local screen, not the official 8xH100 setup:

- `DEVICE=mps`
- `COMPUTE_DTYPE=float32`
- `TORCH_COMPILE=0`
- `ADAM_FUSED=0`
- one SP1024 train shard and a 1M-token validation shard
- `NUM_LAYERS=4`
- `MODEL_DIM=256`
- `NUM_HEADS=4`
- `NUM_KV_HEADS=2`
- `MLP_MULT=2`
- `TRAIN_SEQ_LEN=512`
- `TRAIN_BATCH_TOKENS=16384`
- `ITERATIONS=120`

The metric is `final_int8_zlib_roundtrip_exact val_bpb`.

## Results

| seed | val_loss | val_bpb | artifact bytes |
| ---: | ---: | ---: | ---: |
| 42 | 4.61842299 | 2.76711303 | 2243058 |
| 0 | 4.61715984 | 2.76635623 | 2241742 |
| 314 | 4.61549568 | 2.76535915 | 2243871 |

Mean local MPS `val_bpb`: `2.76627614`.

Compared with the local AttnOutGate + QK5.0 + all-layer IHA base, this is a paired mean improvement of `-0.03708098` BPB. Compared with random-map adapters alone, the causal conv memory addition is a paired mean improvement of `-0.01635720` BPB.

## Reproduction Command

From this folder, with the same local data paths used for screening:

```bash
DEVICE=mps \
COMPUTE_DTYPE=float32 \
TORCH_COMPILE=0 \
ADAM_FUSED=0 \
DATA_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/datasets/fineweb10B_sp1024_screen \
TOKENIZER_PATH=/private/tmp/parameter-golf-worktrees/pg-strong-search/data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=4 \
MODEL_DIM=256 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
TRAIN_SEQ_LEN=512 \
TRAIN_BATCH_TOKENS=16384 \
VAL_BATCH_SIZE=65536 \
ITERATIONS=120 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=30 \
WARMUP_STEPS=1 \
MAX_WALLCLOCK_SECONDS=0 \
ATTN_OUT_GATE=1 \
ATTN_OUT_GATE_SRC=proj \
QK_GAIN_INIT=5.0 \
IHA_LITE=1 \
IHA_START_LAYER=0 \
RANDOM_MAP_ADAPTER=1 \
RANDOM_MAP_DIM=64 \
CAUSAL_CONV_MEMORY=1 \
CONV_KERNEL=5 \
CONV_DILATION=2 \
CONV_START_LAYER=0 \
SEED=42 \
RUN_ID=screen_randommap64_convmem_k5d2_best_seed42 \
python train_gpt.py
```

For official validation, this still needs a real 3-seed 8xH100 run. The PyTorch `train_gpt.py` supports CUDA/DDP, but the local screen is not enough to justify an upstream PR.

## Files

- `train_gpt.py`: PyTorch training and evaluation script with the random-map adapter and causal conv memory code paths.
- `train_seed42_local_mps.log`, `train_seed0_local_mps.log`, `train_seed314_local_mps.log`: local MPS logs from the three seeds above.
- `submission.json`: metadata for the local screen package.
