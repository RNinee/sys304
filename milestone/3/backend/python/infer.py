#!/usr/bin/env python3
"""Optimized infer worker: quantized Qwen, optional DistilBERT/ONNX, batched predict."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

MAX_LEN = 128
LABELS = {0: "not disaster", 1: "disaster"}


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for path in start.parents:
        if (path / "milestone" / "1" / "models").is_dir():
            return path
    return start.parent


REPO_ROOT = _repo_root()
DEFAULT_QWEN = REPO_ROOT / "milestone" / "1" / "models" / "qwen2.5-1.5b-disaster-lora"
DEFAULT_STUDENT = REPO_ROOT / "milestone" / "3" / "models" / "distilbert-student"
DEFAULT_ONNX = REPO_ROOT / "milestone" / "3" / "models" / "distilbert-student-onnx"


def build_input(text: str, keyword: str | None = None) -> str:
    """Match Milestone 1 training format: optional `keyword: …` prefix."""
    body = (text or "").strip()
    kw = (keyword or "").strip()
    if not kw:
        return body
    if body.lower().startswith("keyword:"):
        return body
    return f"keyword: {kw}\n{body}"


def _result(target: int, proba: np.ndarray, model: str, formatted: str) -> dict[str, Any]:
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


def _pick_device():
    import torch

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_backend() -> str:
    env = os.environ.get("INFER_BACKEND", "auto").strip().lower()
    if env and env != "auto":
        return env
    import torch

    student = Path(os.environ.get("STUDENT_DIR", DEFAULT_STUDENT))
    onnx_dir = Path(os.environ.get("ONNX_DIR", DEFAULT_ONNX))
    if (onnx_dir / "model.onnx").exists():
        return "onnx"
    if (student / "config.json").exists():
        return "student"
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        return "fp16"
    return "int8"


class QwenPredictor:
    """Phase 2 naive FP32 LoRA, or merged FP16 / dynamic INT8."""

    def __init__(self, path: Path, backend: str) -> None:
        import torch
        from peft import AutoPeftModelForSequenceClassification
        from transformers import AutoTokenizer

        self.backend = backend
        self.device = _pick_device()
        if backend == "int8":
            self.device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        load_dtype = torch.float32 if backend in {"naive", "int8"} else torch.float16
        print(
            f"loading qwen adapter from {path} backend={backend} dtype={load_dtype}",
            file=sys.stderr,
            flush=True,
        )
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            str(path),
            dtype=load_dtype,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        if backend != "naive":
            model = model.merge_and_unload()
        if backend == "fp16":
            model = model.half()
        if backend == "int8":
            model = torch.ao.quantization.quantize_dynamic(
                model.cpu(),
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

        model.to(self.device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.name = {
            "naive": "qwen2.5-1.5b-lora",
            "fp16": "qwen2.5-1.5b-lora-fp16",
            "int8": "qwen2.5-1.5b-lora-int8",
        }.get(backend, f"qwen2.5-1.5b-lora-{backend}")

    def predict_batch(self, items: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
        import torch

        formatted = [build_input(text, keyword) for text, keyword in items]
        enc = self.tokenizer(
            formatted,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc, return_dict=True).logits
            proba = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
            targets = logits.argmax(dim=-1).detach().cpu().tolist()
        return [
            _result(int(target), row, self.name, text)
            for target, row, text in zip(targets, proba, formatted, strict=True)
        ]


class StudentPredictor:
    """Distilled DistilBERT student (PyTorch)."""

    def __init__(self, path: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = _pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.model.to(self.device)
        self.model.eval()
        self.backend = "student"
        self.name = "distilbert-student"
        print(f"loading student from {path} device={self.device}", file=sys.stderr, flush=True)

    def predict_batch(self, items: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
        import torch

        formatted = [build_input(text, keyword) for text, keyword in items]
        enc = self.tokenizer(
            formatted,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
            proba = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            targets = logits.argmax(dim=-1).detach().cpu().tolist()
        return [
            _result(int(target), row, self.name, text)
            for target, row, text in zip(targets, proba, formatted, strict=True)
        ]


class OnnxPredictor:
    """ONNX Runtime student (format optimization)."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = path / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        # CoreML EP cannot take this DistilBERT graph whole (attention MatMul,
        # LayerNorm, GELU-as-Erf, mask Where). ORT then splits ~50 partitions and
        # copies tensors CPU-CoreML, which is slower than running the full graph
        # on the CPU EP. MLProgram fails to initialize on this export.
        # Force the old path with ORT_COREML=1 if you want to compare.
        providers: list[str] = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        use_coreml = os.environ.get("ORT_COREML", "").strip() in {"1", "true", "yes"}
        if use_coreml and "CoreMLExecutionProvider" in available:
            providers.insert(0, "CoreMLExecutionProvider")
        elif "CoreMLExecutionProvider" in available:
            print(
                "skipping CoreML EP (graph would split); set ORT_COREML=1 to force",
                file=sys.stderr,
                flush=True,
            )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.backend = "onnx"
        self.name = "distilbert-student-onnx"
        self.device = str(self.session.get_providers()[0])
        print(
            f"loading onnx from {model_path} providers={self.session.get_providers()}",
            file=sys.stderr,
            flush=True,
        )

    def predict_batch(self, items: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
        formatted = [build_input(text, keyword) for text, keyword in items]
        enc = self.tokenizer(
            formatted,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="np",
        )
        logits = self.session.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        proba = _softmax(logits)
        targets = np.argmax(proba, axis=-1).tolist()
        return [
            _result(int(target), row, self.name, text)
            for target, row, text in zip(targets, proba, formatted, strict=True)
        ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_predictor():
    backend = resolve_backend()
    qwen_path = Path(os.environ.get("QWEN_ADAPTER_DIR", DEFAULT_QWEN))
    student_path = Path(os.environ.get("STUDENT_DIR", DEFAULT_STUDENT))
    onnx_path = Path(os.environ.get("ONNX_DIR", DEFAULT_ONNX))

    if backend in {"naive", "fp16", "int8"}:
        adapter = qwen_path / "adapter_model.safetensors"
        if not adapter.exists():
            raise FileNotFoundError(f"Qwen adapter not found: {adapter}")
        return QwenPredictor(qwen_path, backend)
    if backend == "student":
        return StudentPredictor(student_path)
    if backend == "onnx":
        return OnnxPredictor(onnx_path)
    raise ValueError(f"unknown INFER_BACKEND={backend}")


PREDICTOR = None


def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = load_predictor()
        device = getattr(PREDICTOR, "device", "unknown")
        print(
            f"ready backend={PREDICTOR.backend} model={PREDICTOR.name} device={device}",
            file=sys.stderr,
            flush=True,
        )
    return PREDICTOR


def _optimizations(pred) -> list[str]:
    mapping = {
        "naive": [],
        "fp16": ["fp16-quantization", "merged-lora"],
        "int8": ["int8-dynamic-quantization", "merged-lora"],
        "student": ["distillation"],
        "onnx": ["distillation", "onnx-runtime"],
    }
    opts = list(mapping.get(pred.backend, [pred.backend]))
    opts.append("batched-inference")
    return opts


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
                    "backend": pred.backend,
                    "device": str(getattr(pred, "device", "unknown")),
                    "optimizations": _optimizations(pred),
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return

        if path == "/predict":
            items, error = _parse_items([data] if isinstance(data, dict) else [])
            if error:
                self._send(400, {"error": error})
                return
            try:
                result = get_predictor().predict_batch(items)[0]
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": str(exc)})
                return
            self._send(200, result)
            return

        if path == "/predict_batch":
            raw_items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(raw_items, list) or not raw_items:
                self._send(400, {"error": "items array required"})
                return
            items, error = _parse_items(raw_items)
            if error:
                self._send(400, {"error": error})
                return
            try:
                results = get_predictor().predict_batch(items)
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": str(exc)})
                return
            self._send(200, {"results": results})
            return

        self._send(404, {"error": "not found"})


def _parse_items(raw_items: list[Any]) -> tuple[list[tuple[str, str | None]], str | None]:
    items: list[tuple[str, str | None]] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            return [], "each item must be an object"
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            return [], "text is required"
        keyword = entry.get("keyword")
        if keyword is not None and not isinstance(keyword, str):
            return [], "keyword must be a string"
        items.append((text, keyword))
    return items, None


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        payload = json.loads(sys.stdin.read() or "{}")
        print(
            json.dumps(
                get_predictor().predict_batch(
                    [(payload.get("text", ""), payload.get("keyword"))]
                )[0]
            )
        )
        return

    host = os.environ.get("INFER_HOST", "127.0.0.1")
    port = int(os.environ.get("INFER_PORT", "9377"))
    get_predictor()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"infer worker listening on http://{host}:{port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
