"""Smoke test: the LoRA config carries the reproducibility-critical values
stated in the thesis (seed 42, LoRA r=16/alpha=32, 3 epochs, greedy decoding).

Runnable directly (`python tests/test_config.py`) or via pytest.
Uses only PyYAML — no heavy ML dependency.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "src" / "lora" / "config.yaml"


def load_config():
    assert CONFIG_PATH.is_file(), f"Missing config file: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_reproducibility_values():
    cfg = load_config()

    # Global seed (thesis: "A global random seed (42) is fixed ...")
    assert cfg["seed"] == 42, f"seed must be 42, got {cfg['seed']}"

    # Data contract consumed by every track that reads the corpus
    data = cfg["data"]
    assert data["sep"] == ";", "corpus separator must be ';'"
    assert data["text_col"] == "email_corpus"
    assert data["label_col"] == "class_label"

    # LoRA hyper-parameters (thesis: r=16, alpha=32, dropout=0.05)
    lora = cfg["lora"]
    assert lora["r"] == 16, f"LoRA r must be 16, got {lora['r']}"
    assert lora["alpha"] == 32, f"LoRA alpha must be 32, got {lora['alpha']}"
    assert abs(float(lora["dropout"]) - 0.05) < 1e-9

    # Training (thesis: three epochs on the training partition)
    assert cfg["train"]["epochs"] == 3, "final training must use 3 epochs"

    # Deterministic inference (thesis: temperature 0 / greedy decoding)
    assert cfg["infer"]["greedy"] is True, "inference must be greedy (deterministic)"


if __name__ == "__main__":
    test_reproducibility_values()
    print("OK  test_config: reproducibility-critical config values match the thesis")
