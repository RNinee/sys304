# Milestone 1 — Problem Scoping & The Prototype

SYS-304 · due Aug 28, 2026

Binary **disaster vs not-disaster** tweet classification on
[Kaggle NLP Getting Started](https://www.kaggle.com/competitions/nlp-getting-started).

## Deliverables in this folder

| File | What it is |
| --- | --- |
| [eda_and_baseline.ipynb](eda_and_baseline.ipynb) | EDA, data pipeline, training, evaluation |
| [main.py](main.py) | Kaggle download into `data/` |
| [data/](data/) | `train.csv`, `test.csv`, `sample_submission.csv` |
| [models/qwen2.5-1.5b-disaster-lora/](models/qwen2.5-1.5b-disaster-lora/) | Saved Qwen 1.5B LoRA weights |

Validation: **85.2% accuracy**, F1 (disaster) **0.816**.

Course-level writeup: [../../README.md](../../README.md)
