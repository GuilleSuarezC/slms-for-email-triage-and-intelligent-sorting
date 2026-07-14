"""Smoke test: the example corpus excerpt has the expected schema
(';'-separated, with the input column `email_corpus` and target `class_label`).

Runnable directly (`python tests/test_example_format.py`) or via pytest.
Uses only pandas.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "examples" / "sample_emails.csv"

REQUIRED_COLUMNS = {"email_id", "email_corpus", "class_label"}


def test_sample_schema():
    assert SAMPLE_PATH.is_file(), f"Missing example file: {SAMPLE_PATH}"
    df = pd.read_csv(SAMPLE_PATH, sep=";", dtype=str, encoding="utf-8")

    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"example is missing required columns: {sorted(missing)}"
    assert len(df) > 0, "example file has no rows"

    # The input text and the target label must be populated for every row.
    assert df["email_corpus"].notna().all(), "empty email_corpus value(s)"
    assert df["class_label"].notna().all(), "empty class_label value(s)"


if __name__ == "__main__":
    test_sample_schema()
    print("OK  test_example_format: example corpus excerpt has the expected schema")
