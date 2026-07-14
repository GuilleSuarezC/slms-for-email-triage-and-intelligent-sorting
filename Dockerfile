# ============================================================================
#  Minimal image for the CPU-reproducible parts of the pipeline:
#    - classical ML baselines (Tracks A and B)
#    - the Sentence-BERT + ChromaDB embedding baseline
#    - metrics / plots / smoke tests
#
#  NOT run inside this image (they need host resources — see README.md):
#    - zero/few-shot SLM  -> requires a LOCAL LM Studio server on the host
#    - LoRA fine-tuning   -> requires an NVIDIA GPU (requirements-lora.txt)
#
#  Build:  docker build -t slms-email-triage .
#  Run   :  docker run --rm slms-email-triage                 # runs smoke tests
#  Run a stage (mount the dataset):
#          docker run --rm -v "$(pwd)/data:/app/data" slms-email-triage \
#              python src/embeddings/task1_knn_chromadb.py --csv_path data/email_dataset.csv --k 5
# ============================================================================
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime needed by lightgbm / scikit-learn at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only what is needed to run the pipeline (data is mounted at runtime;
# see .dockerignore).
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY README.md .

# Default command: run the offline smoke tests so `docker run` verifies the
# environment without needing external data, GPU or LM Studio.
CMD ["python", "tests/run_smoke.py"]
