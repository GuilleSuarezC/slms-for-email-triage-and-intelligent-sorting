# Corpus schema — `email_dataset.csv`

`sample_emails.csv` in this folder is a 12-row excerpt of the synthetic corpus
`data/email_dataset.csv`, kept for quick inspection and for the format smoke
test (`tests/test_example_format.py`). The full corpus and its provenance are
documented in [`../data/DATA_CARD.md`](../data/DATA_CARD.md).

- **Separator:** `;`  ·  **Quote char:** `"`  ·  **Encoding:** UTF-8
- **Rows:** 20,000 (full corpus)  ·  **Columns:** 22

The two columns the classifiers consume are **`email_corpus`** (input text) and
**`class_label`** (target, one of 35 classes). All other columns are metadata.

| # | Column | Description |
|---|--------|-------------|
| 1 | `email_id` | Unique identifier of the message. |
| 2 | `subject` | Email subject line. |
| 3 | `email_corpus` | **Input text** — the email body used for classification. |
| 4 | `class_label` | **Target label** — one of the 35 categories (kept in Spanish). |
| 5 | `secondary_label` | Optional secondary category for multi-intent messages. |
| 6 | `is_multi_intent` | Whether the message mixes more than one intent. |
| 7 | `is_ambiguous` | Whether the message is deliberately ambiguous. |
| 8 | `is_spam` | Whether the message is spam/phishing-like. |
| 9 | `language` | Language of the message. |
| 10 | `tone` | Register/tone requested during generation. |
| 11 | `sender_profile` | Simulated sender profile. |
| 12 | `email_length` | Requested length band. |
| 13 | `noise_level` | Injected linguistic-noise level. |
| 14 | `channel` | Simulated origin channel. |
| 15 | `temperature` | Sampling temperature used to **generate** this email. |
| 16 | `top_p` | Nucleus-sampling `top_p` used to generate this email. |
| 17 | `top_k` | `top_k` used to generate this email. |
| 18 | `seed` | Per-email generation seed (reproducibility of generation). |
| 19 | `model_assigned` | Local model that generated this email (round-robin). |
| 20 | `generation_time_s` | Wall-clock generation time for this email. |
| 21 | `prompt_used` | The prompt sent to the generator model. |
| 22 | `timestamp` | Generation timestamp. |

> Columns 15–19 record the exact per-email generation parameters, which is what
> makes the *generation* stage reproducible even though it uses stochastic
> sampling (see `../data/DATA_CARD.md`). Classification/inference, by contrast,
> is deterministic (temperature 0 / greedy).
