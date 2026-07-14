#!/usr/bin/env bash
# Track B — 35-class classical ML (TF-IDF/BoW/One-Hot/Word2Vec x LogReg/SVM/RF),
# 70/30 hold-out (canonical). Input: data/email_dataset.csv. CPU only.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
exec python src/classical_ml/multiclass_35/email_classifier_nuevo_dataset_holdout.py \
    --data data/email_dataset.csv --test_size 0.30 "$@"
