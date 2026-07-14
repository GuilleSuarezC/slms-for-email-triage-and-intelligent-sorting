#!/usr/bin/env bash
# Track A — binary spam/ham classification on the REAL Kaggle dataset.
# Requires Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json);
# the dataset (slug meruvulikith/190k-spam-ham-email-dataset-for-classification)
# is downloaded automatically. Seed 42, 70/30 split, Optuna HPO.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
exec python src/classical_ml/binary_spam/spam_classifier_pipeline.py "$@"
