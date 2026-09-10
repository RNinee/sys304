#!/usr/bin/env python3
"""Train a DistilBERT student on disaster tweets (task + optional teacher distillation).

Task mode (default) trains on hard labels from train.csv — minutes on MPS/CPU.
Distill mode adds a KL term against the Phase 2 Qwen teacher (slower, closer mimic).

    python distill.py
    python distill.py --mode distill --teacher-max-samples 512
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "milestone" / "1" / "data" / "train.csv").exists():
        REPO = parent
        break
else:
    REPO = ROOT.parents[3]

TRAIN_CSV = REPO / "milestone" / "1" / "data" / "train.csv"
STUDENT_OUT = REPO / "milestone" / "3" / "models" / "distilbert-student"
TEACHER_DIR = REPO / "milestone" / "1" / "models" / "qwen2.5-1.5b-disaster-lora"
STUDENT_ID = "distilbert-base-uncased"
MAX_LEN = 128


class TweetDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        teacher_logits: torch.Tensor | None = None,
    ) -> None:
        self.enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[idx]
        return item


def format_row(text: str, keyword: object) -> str:
    body = str(text).strip()
    kw = "" if pd.isna(keyword) else str(keyword).strip()
    if not kw:
        return body
    return f"keyword: {kw}\n{body}"


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    preds, gold = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(labels.tolist())
    return {
        "accuracy": float(np.mean(np.array(preds) == np.array(gold))),
        "f1_disaster": float(f1_score(gold, preds, pos_label=1)),
        "f1_weighted": float(f1_score(gold, preds, average="weighted")),
    }


def teacher_logits(texts: list[str], device, max_samples: int) -> torch.Tensor | None:
    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoTokenizer as QwenTok

    if not (TEACHER_DIR / "adapter_model.safetensors").exists():
        print("teacher adapter missing; skipping soft labels", file=sys.stderr)
        return None
    print(
        f"loading teacher for distillation ({min(len(texts), max_samples)} tweets)",
        file=sys.stderr,
    )
    tok = QwenTok.from_pretrained(str(TEACHER_DIR), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        str(TEACHER_DIR),
        dtype=torch.float16,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    model.eval()
    out = []
    subset = texts[:max_samples]
    teacher_bs = 8
    with torch.inference_mode():
        for i in range(0, len(subset), teacher_bs):
            chunk = subset[i : i + teacher_bs]
            enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float().cpu()
            out.append(logits)
            if (i // teacher_bs) % 25 == 0:
                print(f"  teacher {i}/{len(subset)}", file=sys.stderr, flush=True)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return torch.cat(out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["task", "distill"], default="task")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.7, help="weight on hard CE vs KL")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--teacher-max-samples", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=STUDENT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_CSV)
    texts = [format_row(t, k) for t, k in zip(df["text"], df["keyword"], strict=True)]
    labels = df["target"].astype(int).tolist()
    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_ID, num_labels=2)
    model.to(device)

    teacher = None
    if args.mode == "distill":
        soft = teacher_logits(x_train, device, args.teacher_max_samples)
        if soft is not None:
            teacher = torch.full((len(x_train), 2), float("nan"))
            teacher[: soft.shape[0]] = soft

    train_ds = TweetDataset(x_train, y_train, tokenizer, teacher)
    val_ds = TweetDataset(x_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        for step, batch in enumerate(train_loader):
            labels_t = batch.pop("labels").to(device)
            teacher_t = batch.pop("teacher_logits", None)
            if teacher_t is not None:
                teacher_t = teacher_t.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = ce(logits, labels_t)
            if teacher_t is not None:
                mask = ~torch.isnan(teacher_t[:, 0])
                if mask.any():
                    log_s = torch.log_softmax(logits[mask] / args.temperature, dim=-1)
                    teacher_p = torch.softmax(teacher_t[mask] / args.temperature, dim=-1)
                    distill = (args.temperature**2) * kl(log_s, teacher_p)
                    loss = args.alpha * loss + (1.0 - args.alpha) * distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * labels_t.size(0)
            seen += labels_t.size(0)
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step} loss={loss.item():.4f}", flush=True)
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1} train_loss={running / seen:.4f} "
            f"val_acc={metrics['accuracy']:.3f} val_f1={metrics['f1_disaster']:.3f}",
            flush=True,
        )

    metrics = evaluate(model, val_loader, device)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    payload = json.dumps(
        {
            "mode": args.mode,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "teacher_max_samples": args.teacher_max_samples if args.mode == "distill" else 0,
            **metrics,
        },
        indent=2,
    ) + "\n"
    (args.out / "metrics.json").write_text(payload)
    print(f"saved student to {args.out}")
    print(json.dumps({"mode": args.mode, **metrics}, indent=2))
    print("teacher Phase 1 F1 (disaster) was 0.816 — student should stay in a similar band.")


if __name__ == "__main__":
    main()
