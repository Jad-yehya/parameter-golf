#!/bin/bash
# 3-seed runner for PR #2135 stack + MP3 marker-pair fusion + adaptive n-gram + gated XSA residual.
# Built on PR #2135's train_gpt.py (PR #1797 base + token-only n-gram tilt + AsymLogit Rescale
# + #2060 levers + NUM_PHASES=1 + GPTQ_CALIBRATION_BATCHES=32 + EVAL_SEQ_LEN=2560), with one
# orthogonal addition: MP3 marker-pair fusion. Three alias donor tokens (8/9/10) replace the
# 2-grams [SPACE,TITLE]/[SPACE,ALLCAPS]/[SPACE,CAPNEXT] in the already-tokenized CaseOps stream;
# tokenizer is unchanged. The model gets a small embedding warm-init in train_model() that
# initializes each donor row as w_s*E[SPACE] + w_t*E[MARKER] (norm-matched to E[MARKER]).
# This candidate also enables ADAPTIVE_NGRAM_GAMMA=1.0: the strict token-only n-gram hint
# remains prefix-only, but the boost is scaled by model disagreement `(1 - q_hint)`.
# It also enables GATED_XSA_RESIDUAL=1: each active XSA layer gets a per-head
# zero-init residual scale `1 + tanh(alpha)` on the self-value projection subtraction.
# Step 0 is therefore exactly the MP3/#2135 XSA path, while training can attenuate or amplify
# XSA per head if useful.
#
# Note: alias smear dampening (the second lever in our prior PR #2109) is DISABLED here —
# the forward-path patches exist in the code but `alias_dampening_active=False` is forced
# regardless of MARKER_PAIR_MODE, so SmearGate behaves identically to PR #2135 vanilla.
#
# Usage:
#   bash run_3seed.sh
# Env overrides:
#   SEEDS="42 0 314"     # default (matches PR #2135 author convention)
#   DATA_PATH=...        # default ./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3
#   TOKENIZER_PATH=...   # default ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
set -o pipefail

SEEDS="${SEEDS:-42 0 314}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

if [ ! -d "$DATA_PATH" ]; then
  echo "ERROR: dataset dir not found: $DATA_PATH"
  echo "Run download_docs.py, prepare_caseops_data.py, prepare_marker_pair_v3.py first."
  exit 1
fi
if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "ERROR: tokenizer model not found: $TOKENIZER_PATH"
  exit 1
fi
if [ ! -f "$DATA_PATH/alias_map.json" ]; then
  echo "ERROR: MP3 alias map not found: $DATA_PATH/alias_map.json"
  echo "Run prepare_marker_pair_v3.py, or point DATA_PATH at the marker-pair dataset."
  exit 1
fi

# Build native ngram-tilt helper if missing (matches PR #2135 setup).
if [ ! -f "libonline_ngram_state.so" ] && [ -f "online_ngram_state.c" ]; then
  echo "Building libonline_ngram_state.so ..."
  gcc -O3 -march=native -shared -fPIC online_ngram_state.c -o libonline_ngram_state.so
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === 3-seed run START ==="
echo "  SEEDS:          $SEEDS"
echo "  DATA_PATH:      $DATA_PATH"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"

FAILED_SEEDS=()

for SEED in $SEEDS; do
  LOG="train_seed${SEED}.log"
  echo ""
  echo "[$(ts)] >>> seed=$SEED -> $LOG >>>"
  env SEED="$SEED" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_NET=Socket \
    DATA_DIR=./data \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    CASEOPS_ENABLED=1 \
    VOCAB_SIZE=8192 \
    ASYM_LOGIT_RESCALE=1 \
    NGRAM_TILT_ENABLED=1 \
    NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 \
    TOKEN_ORDER=16 \
    TOKEN_THRESHOLD=0.800 \
    TOKEN_BOOST=2.625 \
    ADAPTIVE_NGRAM_GAMMA=1.0 \
    WITHIN_TAU=99.0 WITHIN_BOOST=0.0 \
    WORD_TAU=99.0 WORD_BOOST=0.0 \
    AGREE_ADD_BOOST=0.0 \
    SMEAR_GATE_BOS_FIX=0 \
    TTT_LORA_EMA_DECAY=0.0 \
    TTT_UPDATE_EVERY=1 \
    PHASED_TTT_PREFIX_DOCS=2500 \
    PHASED_TTT_NUM_PHASES=1 \
    EVAL_SEQ_LEN=2560 \
    TTT_EVAL_SEQ_LEN=2560 \
    COMPRESSOR=pergroup \
    MATRIX_CLIP_SIGMAS=12.85 \
    ATTN_CLIP_SIGMAS=13.0 \
    MLP_CLIP_SIGMAS=11.5 \
    EMBED_BITS=7 \
    EMBED_CLIP_SIGMAS=14.0 \
    MATRIX_LR=0.028 \
    MIN_LR=0.1 \
    WARMDOWN_FRAC=0.85 \
    BETA2=0.99 \
    FUSED_CE_ENABLED=1 \
    SPARSE_ATTN_GATE_ENABLED=1 \
    SPARSE_ATTN_GATE_SCALE=0.5 \
    SMEAR_GATE_ENABLED=1 \
    GATE_WINDOW=12 \
    LQER_ENABLED=1 \
    LQER_RANK=4 \
    LQER_TOP_K=3 \
    LQER_FACTOR_BITS=4 \
    LQER_ASYM_ENABLED=1 \
    LQER_ASYM_GROUP=32 \
    TTT_WARM_START_A=1 \
    TTT_LORA_RANK=80 \
    TTT_BETA2=0.99 \
    TTT_WEIGHT_DECAY=2.0 \
    TTT_LORA_LR=8e-5 \
    GPTQ_RESERVE_SECONDS=0.5 \
    GPTQ_CALIBRATION_BATCHES=32 \
    GATED_XSA_RESIDUAL=1 \
    GATED_XSA_RESIDUAL_SPAN=1.0 \
    TTT_K_LORA=0 \
    TTT_O_LORA=1 \
    TTT_MLP_LORA=1 \
    EMA_DECAY=0.9965 \
    MARKER_PAIR_MODE=1 \
    MARKER_PAIR_W_SPACE=0.4 \
    MARKER_PAIR_W_TITLE=0.6 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"
  rc="${PIPESTATUS[0]}"
  echo "[$(ts)] <<< seed=$SEED done (rc=$rc) <<<"
  if [ "$rc" -ne 0 ]; then
    FAILED_SEEDS+=("$SEED:rc=$rc")
  fi
done

echo ""
echo "[$(ts)] === 3-seed run DONE ==="
echo ""
echo "=== Summary ==="
printf "%-8s %-16s %-14s %-12s %s\n" "seed" "ttt_phased_bpb" "val_loss" "artifact_B" "eval_s"
echo "------------------------------------------------------------------------"
for SEED in $SEEDS; do
  LOG="train_seed${SEED}.log"
  sw=$(grep -E "quantized_ttt_phased" "$LOG" 2>/dev/null | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
  vl=$(grep -E "quantized_ttt_phased" "$LOG" 2>/dev/null | grep -oP "val_loss:\K[0-9.]+" | tail -1)
  sz=$(grep "Total submission size quantized+pergroup" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
  ev=$(grep -oP "total_eval_time:\K[0-9.]+" "$LOG" 2>/dev/null | tail -1)
  printf "%-8s %-16s %-14s %-12s %s\n" "$SEED" "${sw:-?}" "${vl:-?}" "${sz:-?}" "${ev:-?}"
  if [ -z "${sw:-}" ] || [ -z "${vl:-}" ] || [ -z "${sz:-}" ] || [ -z "${ev:-}" ]; then
    FAILED_SEEDS+=("$SEED:missing_summary_metric")
  fi
done

if [ "${#FAILED_SEEDS[@]}" -ne 0 ]; then
  echo ""
  echo "ERROR: incomplete or failed validation: ${FAILED_SEEDS[*]}"
  exit 1
fi
