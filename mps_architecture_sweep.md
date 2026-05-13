# MPS Architecture Sweep

Local compact SP1024 screens only. These are not official Parameter Golf validation results.

Common setup:

- `DEVICE=mps`, `COMPUTE_DTYPE=float32`, `TORCH_COMPILE=0`, `ADAM_FUSED=0`
- one SP1024 train shard, 1M-token validation shard
- `NUM_LAYERS=4`, `MODEL_DIM=256`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `MLP_MULT=2`
- `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=16384`, `ITERATIONS=120`
- base recipe: `ATTN_OUT_GATE=1`, `ATTN_OUT_GATE_SRC=proj`, `QK_GAIN_INIT=5.0`, `IHA_LITE=1`, `IHA_START_LAYER=0`
- metric: `final_int8_zlib_roundtrip_exact val_bpb`

## Results

| branch | README request | seed 42 | seed 0 | seed 314 | mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `pg-strong-search` | local attention/IHA base | 2.80552135 | 2.80257583 | 2.80197416 | 2.80335711 |
| `pg-exp-random-map-adapters` | learning adapters on random linear maps | 2.78375022 | 2.78200176 | 2.78214804 | 2.78263334 |
| `pg-exp-randommap-convmem` | random-map adapters + causal conv memory | 2.76711303 | 2.76635623 | 2.76535915 | 2.76627614 |

Single-seed side screens:

| branch | README request | seed 42 |
| --- | --- | ---: |
| `pg-exp-conv-memory` | state-space / long-context-style memory | 2.78918015 |
| `pg-exp-universal-recur` | universal transformer / shared recurrent blocks | 2.80006600 |

## Interpretation

The strongest local lead is the composition of fixed random sinusoidal adapters and a causal dilated depthwise memory filter. It improves the local attention/IHA base by a paired mean of `-0.03708098` BPB over three seeds and improves random-map-only by `-0.01635720` BPB.

The raw speed cost is visible in the compact screen: the combo runs around `490-512 ms/step` versus about `424 ms/step` for the attention/IHA base. The artifact remains small in this local harness (`~2.24 MB` int8+zlib including code).
