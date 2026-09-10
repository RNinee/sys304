#!/usr/bin/env python3
"""Export the DistilBERT student to ONNX for ONNX Runtime serving.

    python export_onnx.py
    python export_onnx.py --src ../models/distilbert-student --out ../models/distilbert-student-onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "milestone").is_dir():
        REPO = parent
        break
else:
    REPO = ROOT.parents[3]

DEFAULT_SRC = REPO / "milestone" / "3" / "models" / "distilbert-student"
DEFAULT_OUT = REPO / "milestone" / "3" / "models" / "distilbert-student-onnx"
MAX_LEN = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    if not (args.src / "config.json").exists():
        print(
            f"student checkpoint missing at {args.src}\n"
            "Train it first: python distill.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(args.src))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.src))
    model.eval()

    dummy = tokenizer(
        "Forest fire near La Ronge Sask. Canada",
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "model.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
    )
    tokenizer.save_pretrained(args.out)
    shutil.copy2(args.src / "config.json", args.out / "config.json")
    data_path = args.out / "model.onnx.data"
    total_bytes = onnx_path.stat().st_size + (data_path.stat().st_size if data_path.exists() else 0)
    meta = {
        "format": "onnx",
        "opset": args.opset,
        "bytes": total_bytes,
        "src": str(args.src),
    }
    (args.out / "export.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {onnx_path} ({meta['bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
