# Candidate: PR #2135 + MP3 + Adaptive N-gram + Gated XSA Residual

This is a validation candidate, not a completed submission. Do not open a PR with this folder until it has fresh 3-seed logs from the target 8xH100 environment.

## Hypothesis

[PR #2158](https://github.com/openai/parameter-golf/pull/2158) showed that MP3 marker-pair fusion composes with the current PR #2135 frontier stack, reporting a post-deadline non-record 3-seed mean of 1.05559 BPB. This candidate keeps that MP3 mechanism and adds two small validation levers:

`ADAPTIVE_NGRAM_GAMMA=1.0` keeps the existing strict token-only n-gram hint but scales its boost by `(1 - q_hint)`, where `q_hint` is the model's prefix-conditioned probability assigned to the hinted token. This preserves the closed-form normalization and reduces over-tilting when the neural model already agrees with the n-gram expert.

`GATED_XSA_RESIDUAL=1` gives each XSA-enabled attention layer a per-head scalar `alpha` and scales the usual self-value projection subtraction by `1 + tanh(alpha)`. The parameter is zero-initialized, so the initial forward pass is exactly the PR #2135/MP3 XSA path. Training can then attenuate or amplify XSA per head if that helps.

## Delta vs PR #2158

- `train_gpt.py`: adds `ADAPTIVE_NGRAM_GAMMA`, `GATED_XSA_RESIDUAL`, and `GATED_XSA_RESIDUAL_SPAN`.
- `online_ngram_tilt.py`: optionally scales the per-position boost by model disagreement while defaulting to the original fixed-boost formula.
- `run_3seed.sh`: enables `ADAPTIVE_NGRAM_GAMMA=1.0` and `GATED_XSA_RESIDUAL=1` with span 1.0.
- No tokenizer-rule change, no eval-time cache, no target-conditioned gating, and no change to score-first TTT ordering.

## MP3 Base

The folder inherits PR #2158's marker-pair fusion pipeline:

```bash
python3 download_docs.py
python3 prepare_caseops_data.py \
  --docs ./data/datasets/docs_selected.jsonl \
  --out ./data \
  --sp ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
python3 prepare_marker_pair_v3.py
```

Then run the candidate:

```bash
bash validate_8xh100.sh
```

If data is already prepared, `validate_8xh100.sh` skips preparation and delegates to `run_3seed.sh`.
At the end of a complete run, `run_3seed.sh` calls `summarize_validation.py` and writes `validation_summary.json` with the 3-seed aggregate and budget checks.

Expected validation protocol before PR:

- seeds `42 0 314`
- full canonical CaseOps SP8192 validation
- `NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`
- train and eval both under 600 seconds
- max artifact under 16,000,000 bytes
- report paired comparison against PR #2158 and PR #2135

## Current Status

No candidate results are included yet. Prior MP3 logs were intentionally removed from this copied folder to avoid mixing baseline evidence with the new gated-XSA hypothesis.
