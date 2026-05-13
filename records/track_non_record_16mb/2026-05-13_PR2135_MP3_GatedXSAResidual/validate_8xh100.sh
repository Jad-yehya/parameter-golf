#!/bin/bash
# Prepare the canonical CaseOps+MP3 data locally in this records folder, then
# launch the validation-gated 3-seed candidate. Intended for an 8xH100
# Parameter Golf environment, run from this directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3}"
DOCS_PATH="${DOCS_PATH:-./data/datasets/docs_selected.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

if [[ ! -f "$DOCS_PATH" ]]; then
  python3 download_docs.py
fi

if [[ ! -d "./data/datasets/fineweb10B_sp8192_caseops" ]]; then
  python3 prepare_caseops_data.py \
    --docs "$DOCS_PATH" \
    --out ./data \
    --sp "$TOKENIZER_PATH"
fi

if [[ ! -d "$DATA_PATH" ]]; then
  python3 prepare_marker_pair_v3.py
fi

DATA_PATH="$DATA_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" bash run_3seed.sh
