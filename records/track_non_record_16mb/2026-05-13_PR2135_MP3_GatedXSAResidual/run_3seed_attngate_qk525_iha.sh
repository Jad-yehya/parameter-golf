#!/bin/bash
# Optional validation variant from local MPS screening:
# replace SparseAttnGate with AttnOutGate, raise QK_GAIN_INIT to 5.25, and
# enable IHA-lite from the midpoint layer onward. Logs are tagged so they do
# not overwrite the default MP3+sparse-gate candidate logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LOG_SUFFIX="${LOG_SUFFIX:-attngate_qk525_iha}"
export SPARSE_ATTN_GATE_ENABLED=0
export ATTN_OUT_GATE_ENABLED=1
export ATTN_OUT_GATE_SRC=proj
export QK_GAIN_INIT=5.25
export IHA_LITE=1
export IHA_START_LAYER="${IHA_START_LAYER:-5}"
export IHA_MIX_V=1

exec bash ./run_3seed.sh
