#!/usr/bin/env python3
"""Load the saved Qwen2.5-1.5B LoRA classifier and serve POST /predict."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for path in [start, *start.parents]:
        if (path / "milestone" / "1" / "models").is_dir():
            return path
    return start.parents[4]


REPO_ROOT = _repo_root()
DEFAULT_QWEN = REPO_ROOT / "milestone" / "1" / "models" / "qwen2.5-1.5b-disaster-lora"
MAX_LEN = 128
LABELS = {0: "not disaster", 1: "disaster"}
MODEL_NAME = "qwen2.5-1.5b-lora"


def build_input(text: str, keyword: str | None = None) -> str:
    """Match Milestone 1 training format: optional `keyword: …` prefix."""
    body = (text or "").strip()
    kw = (keyword or "").strip()
    if not kw:
        return body
    if body.lower().startswith("keyword:"):
        return body
    return f"keyword: {kw}\n{body}"


class QwenPredictor:
    name = MODEL_NAME

    def __init__(self, path: Path) -> None:
        import torch
        from peft import AutoPeftModelForSequenceClassification
        from transformers import AutoTokenizer

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoPeftModelForSequenceClassification.from_pretrained(
            str(path),
            dtype="float32",
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        model.eval()

        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.path = str(path)

    def predict(self, text: str, keyword: str | None = None) -> dict[str, Any]:
        import torch

        formatted = build_input(text, keyword)
        enc = self.tokenizer(
            [formatted],
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc, return_dict=True).logits
            proba = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            target = int(logits.argmax(dim=-1).item())
        return _result(target, proba, self.name, formatted)


def _result(target: int, proba, model: str, formatted: str) -> dict[str, Any]:
    p_not, p_yes = float(proba[0]), float(proba[1])
    return {
        "target": target,
        "label": LABELS[target],
        "confidence": float(proba[target]),
        "probabilities": {
            "not_disaster": p_not,
            "disaster": p_yes,
        },
        "model": model,
        "input": formatted,
    }


def load_predictor() -> QwenPredictor:
    qwen_path = Path(os.environ.get("QWEN_ADAPTER_DIR", DEFAULT_QWEN))
    adapter = qwen_path / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError(f"Qwen adapter not found: {adapter}")
    print(f"loading qwen adapter from {qwen_path}", file=sys.stderr, flush=True)
    return QwenPredictor(qwen_path)


PREDICTOR: QwenPredictor | None = None


def get_predictor() -> QwenPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = load_predictor()
        print(
            f"ready backend={PREDICTOR.name} device={PREDICTOR.device}",
            file=sys.stderr,
            flush=True,
        )
    return PREDICTOR


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/"):
            pred = get_predictor()
            self._send(
                200,
                {
                    "status": "ok",
                    "model": pred.name,
                    "role": "infer-worker",
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/predict":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send(400, {"error": "text is required"})
            return
        keyword = data.get("keyword")
        if keyword is not None and not isinstance(keyword, str):
            self._send(400, {"error": "keyword must be a string"})
            return
        try:
            result = get_predictor().predict(text, keyword)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc)})
            return
        self._send(200, result)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        payload = json.loads(sys.stdin.read() or "{}")
        print(json.dumps(get_predictor().predict(payload.get("text", ""), payload.get("keyword"))))
        return

    host = os.environ.get("INFER_HOST", "127.0.0.1")
    port = int(os.environ.get("INFER_PORT", "9377"))
    get_predictor()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"infer worker listening on http://{host}:{port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
