#!/usr/bin/env bash
# Track B — LoRA fine-tuning + evaluation, 70/30 hold-out (canonical result R15).
# REQUIRES an NVIDIA GPU and requirements-lora.txt (QLoRA 4-bit via bitsandbytes).
#
# NOTE ON THE SPLIT: config.yaml ships data.test_size: 0.20 (an 80/20 split, the
# superseded run). To reproduce the PUBLISHED 0.8542 macro-F1 (70/30), set
# data.test_size: 0.30 in src/lora/config.yaml BEFORE running this script.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../src/lora"
# Put the dataset where config.yaml expects it (csv_path: email_dataset.csv):
ln -sf ../../data/email_dataset.csv email_dataset.csv
python scripts/prepare_data.py      # -> outputs/train.csv, outputs/test.csv, outputs/labels.json
python scripts/train_final.py       # -> outputs/final_adapter/
python scripts/evaluate_test.py     # -> outputs/test_metrics.json, test_classification_report.json, ...
# Optional export for LM Studio/GGUF (~15 GB):  python scripts/merge_lora.py
