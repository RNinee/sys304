#!/usr/bin/env python3
"""Run the DistilBERT student on Kaggle test.csv and write submission.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "python"))

from infer import StudentPredictor  # noqa: E402

TEST_CSV = ROOT.parent / "1" / "data" / "test.csv"
TEACHER_SUB = ROOT.parent / "1" / "data" / "submission.csv"
OUT_DIR = ROOT / "data"
STUDENT_DIR = ROOT / "models" / "distilbert-student"
BATCH = 32


def main() -> None:
    test = pd.read_csv(TEST_CSV)
    pred = StudentPredictor(STUDENT_DIR)
    items = [
        (str(text), None if pd.isna(kw) else str(kw))
        for text, kw in zip(test["text"], test["keyword"], strict=True)
    ]
    targets: list[int] = []
    for i in range(0, len(items), BATCH):
        chunk = items[i : i + BATCH]
        results = pred.predict_batch(chunk)
        targets.extend(int(r["target"]) for r in results)
        if i == 0 or (i // BATCH) % 20 == 0:
            print(f"  {min(i + BATCH, len(items))}/{len(items)}", flush=True)

    sub = pd.DataFrame({"id": test["id"], "target": targets})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)

    summary = {
        "n": int(len(sub)),
        "disaster_rate": float(sub["target"].mean()),
        "path": str(out_path),
        "model": pred.name,
    }
    if TEACHER_SUB.exists():
        teacher = pd.read_csv(TEACHER_SUB)
        merged = sub.merge(teacher, on="id", suffixes=("_student", "_qwen"))
        agree = float((merged["target_student"] == merged["target_qwen"]).mean())
        summary["agreement_with_qwen_submission"] = agree
        summary["qwen_disaster_rate"] = float(teacher["target"].mean())
    (OUT_DIR / "submission_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("upload", out_path, "to https://www.kaggle.com/competitions/nlp-getting-started")


if __name__ == "__main__":
    main()
