#!/usr/bin/env bash
# Track B — zero/few-shot SLM classification, 70/30 hold-out (canonical).
# REQUIRES a local LM Studio server with qwen/qwen2.5-vl-7b loaded.
# Deterministic (temperature 0). WARNING: very long (~17 h, ~12,000 LLM calls
# for the full 6,000-sample test partition). Use --max-rows N to try a subset.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
exec python src/slm_prompting/llm_email_classifier_holdout_v4.py \
    --input data/email_dataset.csv --output outputs --test-size 0.30 "$@"
