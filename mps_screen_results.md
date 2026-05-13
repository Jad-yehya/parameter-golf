# Local MPS Screen Results

These are local compact SP1024 screens only. They are not official Parameter Golf validation evidence.

Common setup:

- `DEVICE=mps`, `COMPUTE_DTYPE=float32`, `TORCH_COMPILE=0`, `ADAM_FUSED=0`
- `NUM_LAYERS=4`, `MODEL_DIM=256`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `MLP_MULT=2`
- `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=16384`, `ITERATIONS=120`
- one SP1024 train shard, 1M-token validation shard
- metric: `final_int8_zlib_roundtrip_exact val_bpb`

## Three-Seed Summary

| variant | seed 42 | seed 0 | seed 314 | mean |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.83884630 | 2.83116766 | 2.82523777 | 2.83175058 |
| AttnOutGate + QK 5.25 | 2.80973965 | 2.80653757 | 2.80589733 | 2.80739152 |
| AttnOutGate + QK 5.25 + midpoint IHA | 2.80768378 | 2.80495682 | 2.80443943 | 2.80569334 |
| AttnOutGate + QK 5.0 + all-layer IHA | 2.80552135 | 2.80257583 | 2.80197416 | 2.80335711 |

Best local setting:

```bash
ATTN_OUT_GATE=1 ATTN_OUT_GATE_SRC=proj QK_GAIN_INIT=5.0 IHA_LITE=1 IHA_START_LAYER=0
```

Paired mean deltas:

- best vs baseline: `-0.02839347` BPB
- best vs AttnOutGate + QK 5.25: `-0.00403441` BPB
- best vs midpoint IHA + QK 5.25: `-0.00233623` BPB

## Sparse-Gate Check

After adding official-style `SPARSE_ATTN_GATE` support to the local MPS harness:

| variant | seed 42 |
| --- | ---: |
| SparseAttnGate scale 0.5 + QK 5.0 | 2.84527787 |
| SparseAttnGate scale 0.5 + QK 5.0 + all-layer IHA | 2.84035791 |

In this compact harness, SparseAttnGate is much worse than AttnOutGate, so the forked candidate's optional validation wrapper uses AttnOutGate + all-layer IHA.
