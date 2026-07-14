"""Aggregate offline smoke tests. Exit code 0 = all passed, 1 = a failure.

This is the default command of the Docker image: it verifies the environment
(config values, example schema, source byte-compilation) without needing any
external data, GPU or LM Studio server.

    python tests/run_smoke.py
"""
import sys
import traceback
from pathlib import Path

# Ensure the sibling test modules are importable regardless of how this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_config
import test_example_format
import test_imports

CHECKS = [
    ("config values (seed/LoRA/greedy) match the thesis", test_config.test_reproducibility_values),
    ("example corpus excerpt schema", test_example_format.test_sample_schema),
    ("all src/ and scripts/ byte-compile", test_imports.test_all_sources_compile),
]


def main() -> int:
    print("=" * 70)
    print(" SLMs email-triage — offline smoke tests")
    print("=" * 70)
    failed = 0
    for label, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {label}")
        except Exception:  # noqa: BLE001 - report any failure
            failed += 1
            print(f"  FAIL  {label}")
            traceback.print_exc()
    print("-" * 70)
    if failed:
        print(f" {failed}/{len(CHECKS)} check(s) FAILED")
        return 1
    print(f" all {len(CHECKS)} checks passed")
    print("\nTo run an actual pipeline stage, mount the dataset and call a script,")
    print("e.g.:  python src/embeddings/task1_knn_chromadb.py "
          "--csv_path data/email_dataset.csv --k 5")
    print("See README.md for the full, ordered reproduction commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
