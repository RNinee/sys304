# SYS-304: Scalable Algorithms and Infrastructure

**Milestone 1** lives here: [`milestone/1/`](milestone/1/)

| Path | Contents |
| --- | --- |
| [`milestone/1/README.md`](milestone/1/README.md) | Milestone 1 index |
| [`milestone/1/eda_and_baseline.ipynb`](milestone/1/eda_and_baseline.ipynb) | EDA, training, metrics |
| [`milestone/1/models/`](milestone/1/models/) | Saved baseline weights |
| [`milestone/1/data/`](milestone/1/data/) | Train/test CSVs |

# Milestone 1: Disaster Tweet Classification

**Course:** SYS-304 Scalable Algorithms and Infrastructure  
**Domain:** NLP / binary text classification  
**Dataset:** [Kaggle NLP Getting Started — Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

## Problem

Given a tweet (and optional keyword/location metadata), predict whether it describes a **real disaster** (`1`) or not (`0`). This is a noisy, short-text problem: the same words (“fire”, “flood”, “body bags”) appear in both news reports and jokes/metaphors. The competition metric is **F1**.

We use this as a well-scoped prototype before later milestones add scalable training and serving infrastructure.

## Dataset

| Split | Rows | Files |
| --- | ---: | --- |
| Train | 7,613 | `milestone/1/data/train.csv` (`id`, `keyword`, `location`, `text`, `target`) |
| Test | 3,263 | `milestone/1/data/test.csv` (no labels) |

Download (requires a Kaggle API token and accepted [competition rules](https://www.kaggle.com/competitions/nlp-getting-started/rules)):

```bash
source .venv/bin/activate
python milestone/1/main.py
```

## Prototype

Notebook: [`milestone/1/eda_and_baseline.ipynb`](milestone/1/eda_and_baseline.ipynb)

1. EDA: class balance, missing values, keyword rates, text length, duplicates.
2. Naive LM baseline: **Qwen2.5-1.5B-Instruct** (1.5B, Apache-2.0) with LoRA sequence classification. One epoch, no hyperparameter search.

Validation (1,523 tweets, 80/20 stratified split): **85.2% accuracy**, F1 (disaster) **0.816**.

## Saved weights

- LM adapters: `milestone/1/models/qwen2.5-1.5b-disaster-lora/` (`adapter_model.safetensors`)

The 1.5B base weights stay in the Hugging Face cache. Reload the LoRA adapter from the notebook’s last section (skip retraining for the demo video).

## Setup

```bash
# data download
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# EDA + training (Python 3.12, Apple MPS)
uv venv .venv-ml --python 3.12
uv pip install --python .venv-ml/bin/python -r requirements-ml.txt
```

Open the notebook with the `Python (sys304-ml)` kernel.
