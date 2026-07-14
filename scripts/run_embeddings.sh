#!/usr/bin/env bash
# Track B — embedding k-NN baseline (Sentence-BERT all-MiniLM-L6-v2 + ChromaDB),
# k=5. Input: data/email_dataset.csv. Downloads the embedding model on first run.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
exec python src/embeddings/task1_knn_chromadb.py \
    --csv_path data/email_dataset.csv --k 5 "$@"
