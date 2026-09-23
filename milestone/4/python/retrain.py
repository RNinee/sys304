#!/usr/bin/env python3
"""Retrain the DistilBERT student from labeled request logs.

The cycle runs when either the time since the last attempt exceeds
--interval-minutes, or mean logged confidence is below --confidence-floor.
--mean-confidence simulates that drop. A better validation F1 replaces the
live student and ONNX weights, then asks the API to restart the infer worker.

    python python/retrain.py --check
    python python/retrain.py --apply-train-labels
    python python/retrain.py --force --epochs 1 --max-samples 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from trigger import should_retrain

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "milestone" / "1" / "data" / "train.csv").exists():
        REPO = parent
        break
else:
    REPO = ROOT.parents[3]

MILESTONE = ROOT.parents[1]
TRAIN_CSV = REPO / "milestone" / "1" / "data" / "train.csv"
DEFAULT_DB = MILESTONE / "data" / "requests.sqlite"
STATE_PATH = MILESTONE / "data" / "retrain_state.json"
CANDIDATE = MILESTONE / "models" / "candidate"
LIVE_STUDENT = REPO / "milestone" / "3" / "models" / "distilbert-student"
LIVE_ONNX = REPO / "milestone" / "3" / "models" / "distilbert-student-onnx"
EXPORT_SCRIPT = REPO / "milestone" / "3" / "backend" / "python" / "export_onnx.py"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    return db


def read_state(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    last = payload.get("last_run")
    return float(last) if last is not None else None


def write_state(path: Path, when: float, reason: str, promoted: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"last_run": when, "reason": reason, "promoted": promoted},
            indent=2,
        )
        + "\n"
    )


def db_mean_confidence(db: sqlite3.Connection) -> float | None:
    row = db.execute(
        "SELECT avg(confidence) AS mean FROM requests WHERE confidence IS NOT NULL"
    ).fetchone()
    if row is None or row["mean"] is None:
        return None
    return float(row["mean"])


def apply_labels(db: sqlite3.Connection, pairs: list[tuple[str, int]]) -> int:
    updated = 0
    for text, target in pairs:
        cursor = db.execute(
            """UPDATE requests
               SET truth_target = ?
               WHERE text = ? AND truth_target IS NULL""",
            (target, text),
        )
        updated += cursor.rowcount
    db.commit()
    return updated


def labels_from_csv(path: Path) -> list[tuple[str, int]]:
    with path.open(newline="") as handle:
        return [
            (row["text"].strip(), int(row["target"]))
            for row in csv.DictReader(handle)
            if row.get("text") and row.get("target") not in (None, "")
        ]


def labeled_rows(db: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return db.execute(
        """SELECT text, keyword, truth_target FROM requests
           WHERE truth_target IS NOT NULL AND text != ''
           ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()


def reload_worker(url: str, token: str | None) -> None:
    request = urllib.request.Request(f"{url.rstrip('/')}/admin/reload", method="POST", data=b"")
    if token:
        request.add_header("X-Admin-Token", token)
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status >= 300:
            raise RuntimeError(f"reload failed: HTTP {response.status}")


def replace_tree(src: Path, dst: Path) -> Path | None:
    backup = None
    if dst.exists():
        backup = dst.with_name(f"{dst.name}.bak")
        if backup.exists():
            shutil.rmtree(backup)
        dst.rename(backup)
    try:
        shutil.copytree(src, dst)
    except Exception:
        restore_tree(dst, backup)
        raise
    return backup


def restore_tree(dst: Path, backup: Path | None) -> None:
    if backup is None or not backup.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    backup.rename(dst)


def train_student(
    texts: list[str],
    keywords: list[str],
    labels: list[int],
    epochs: int,
    batch_size: int,
    out: Path,
) -> tuple[float, float]:
    import torch
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    sys.path.insert(0, str(REPO / "milestone" / "3" / "backend" / "python"))
    from distill import TweetDataset, evaluate, format_row

    formatted = [
        format_row(text, keyword or None)
        for text, keyword in zip(texts, keywords, strict=True)
    ]
    stratify = labels if len(set(labels)) > 1 else None
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            formatted,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            formatted,
            labels,
            test_size=0.2,
            random_state=42,
        )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    source = str(LIVE_STUDENT) if (LIVE_STUDENT / "config.json").exists() else "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2)
    model.to(device)

    old_f1 = -1.0
    if (LIVE_STUDENT / "config.json").exists():
        old = AutoModelForSequenceClassification.from_pretrained(str(LIVE_STUDENT), num_labels=2)
        old.to(device)
        old_loader = DataLoader(TweetDataset(x_val, y_val, tokenizer), batch_size=batch_size)
        old_f1 = float(evaluate(old, old_loader, device)["f1_disaster"])
        del old

    train_loader = DataLoader(
        TweetDataset(x_train, y_train, tokenizer),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TweetDataset(x_val, y_val, tokenizer), batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            labels_t = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items() if key != "teacher_logits"}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1} val_f1={metrics['f1_disaster']:.3f} val_acc={metrics['accuracy']:.3f}",
            flush=True,
        )
    new_f1 = float(evaluate(model, val_loader, device)["f1_disaster"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "metrics.json").write_text(
        json.dumps({"f1_disaster": new_f1, "old_f1_disaster": old_f1}, indent=2) + "\n"
    )
    return old_f1, new_f1


def promote(candidate: Path, api_url: str, token: str | None) -> None:
    student_backup = replace_tree(candidate, LIVE_STUDENT)
    onnx_tmp = MILESTONE / "models" / "candidate-onnx"
    onnx_backup = None
    if onnx_tmp.exists():
        shutil.rmtree(onnx_tmp)
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--src",
                str(LIVE_STUDENT),
                "--out",
                str(onnx_tmp),
            ]
        )
        onnx_backup = replace_tree(onnx_tmp, LIVE_ONNX)
        reload_worker(api_url, token)
    except Exception:
        restore_tree(LIVE_STUDENT, student_backup)
        restore_tree(LIVE_ONNX, onnx_backup)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB))
    parser.add_argument("--interval-minutes", type=float, default=24 * 60)
    parser.add_argument("--confidence-floor", type=float, default=0.55)
    parser.add_argument("--mean-confidence", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply-train-labels", action="store_true")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--min-rows", type=int, default=16)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    db = connect(args.db)
    if args.labels:
        print(f"applied {apply_labels(db, labels_from_csv(args.labels))} labels from {args.labels}")
    if args.apply_train_labels:
        print(f"applied {apply_labels(db, labels_from_csv(TRAIN_CSV))} labels from train.csv")

    measured = args.mean_confidence
    if measured is None:
        measured = db_mean_confidence(db)
    decision = should_retrain(
        now=time.time(),
        last_run=read_state(STATE_PATH),
        interval_seconds=args.interval_minutes * 60,
        mean_confidence=measured,
        confidence_floor=args.confidence_floor,
        force=args.force,
    )
    labeled = labeled_rows(db, args.max_samples)
    print(
        json.dumps(
            {
                "run": decision.run,
                "reason": decision.reason,
                "mean_confidence": measured,
                "labeled_rows": len(labeled),
            }
        )
    )
    if args.check or not decision.run:
        return
    if len(labeled) < args.min_rows:
        print(
            f"need at least {args.min_rows} labeled rows, found {len(labeled)}. "
            "Add labels with --labels or --apply-train-labels.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    texts = [str(row["text"]) for row in labeled]
    keywords = [str(row["keyword"] or "") for row in labeled]
    labels = [int(row["truth_target"]) for row in labeled]
    if CANDIDATE.exists():
        shutil.rmtree(CANDIDATE)
    old_f1, new_f1 = train_student(
        texts, keywords, labels, args.epochs, args.batch_size, CANDIDATE
    )
    promoted = new_f1 > old_f1
    print(json.dumps({"old_f1": old_f1, "new_f1": new_f1, "promoted": promoted}))
    if promoted:
        token = os.environ.get("ADMIN_TOKEN")
        promote(CANDIDATE, args.api, token)
        print(f"promoted candidate and reloaded {args.api}")
    else:
        print("candidate did not beat the live model; weights left in place")
    write_state(STATE_PATH, time.time(), decision.reason, promoted)


if __name__ == "__main__":
    main()
