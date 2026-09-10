# Milestone 3 — Architectural Scaling & Optimization

SYS-304 · due Sep 25, 2026

Phase 2 served the **Qwen2.5-1.5B LoRA** classifier as a single FP32 worker. Phase 3 keeps that contract (`POST /predict`) and adds **model-level** and **infrastructure-level** optimizations so the same UI can take concurrent traffic without blocking on one slow forward pass.

## Run

```bash
chmod +x milestone/3/deploy.sh
./milestone/3/deploy.sh local     # UI :3000  API :8000  Redis :6379
# or
./milestone/3/deploy.sh           # docker compose --build
```

`INFER_BACKEND=auto` (default on the host) picks **ONNX student** if you exported it, else the **DistilBERT student**, else **FP16** on MPS/CUDA or **INT8** on CPU.

## What we implemented

### Week 5 — model-level (all three)

| Technique | Where | What it does |
| --- | --- | --- |
| **Quantization** | `backend/python/infer.py` | Merge LoRA, run FP16 on GPU/MPS or dynamic INT8 (`torch.ao.quantization.quantize_dynamic`) on CPU |
| **Distillation** | `backend/python/distill.py` | Train `distilbert-base-uncased` (~66M) with CE on `train.csv` plus KL against Qwen 1.5B LoRA teacher logits |
| **ONNX Runtime** | `backend/python/export_onnx.py` | Export the student to `models/distilbert-student-onnx/model.onnx` and serve with ONNX Runtime (CPU EP; CoreML is opt-in via `ORT_COREML=1` because it splits this graph) |

### Week 6 — system-level (all three)

| Technique | Where | Bottleneck it targets |
| --- | --- | --- |
| **Async API + infer workers** | `backend/src/index.ts` | Elysia stays non-blocking; `INFER_WORKERS` can spawn extra Python replicas (use with the small student, not two 1.5B copies) |
| **Dynamic batching** | `backend/src/batcher.ts` + `POST /predict_batch` | Concurrent tweets share one padded forward pass (15 ms window, max 8) |
| **Redis exact-match cache** | `backend/src/cache.ts` + compose `redis` | Repeat tweets skip the model (`GET /metrics` shows hits) |

```
browser ──POST /predict──► Elysia :8000 ──► Redis GET
                              │ miss
                              ▼
                        dynamic batcher (≤15 ms, ≤8)
                              │
                              ▼
                     Python infer worker(s) :9377
                     FP16 / INT8 Qwen  or  DistilBERT / ONNX
```

## Train the student + export ONNX (optional, recommended)

```bash
cd milestone/3/backend/python
# mimic Qwen: hard CE + KL on teacher logits (all train tweets)
python distill.py --mode distill --teacher-max-samples 10000
python export_onnx.py
```

Then `INFER_BACKEND=onnx ./milestone/3/deploy.sh local`.

## Benchmarks

**Model-level** (naive FP32 Qwen vs quantized / student / ONNX). Loads one backend at a time:

```bash
cd milestone/3
python benchmarks/bench.py --backends naive,fp16
python benchmarks/bench.py --backends student,onnx
```

Writes `benchmarks/results/latest.md`. Paste that table into the PDF report.

**System-level** (needs the API up):

```bash
bun run benchmarks/bench_api.ts --url http://127.0.0.1:8000 --concurrency 8
```

Compares sequential unique tweets, concurrent traffic (batching), and repeated tweets (cache).

## API

Same as Milestone 2, plus:

| Endpoint | Notes |
| --- | --- |
| `POST /predict` | Adds `cached` and `batch_size` |
| `POST /predict_batch` | Worker-only; `{ "items": [ { "text", "keyword?" } ] }` |
| `GET /metrics` | Cache hits/misses, batch counts |
| `GET /health` | `cache`, `optimizations`, `infer` |

The Milestone 2 assistant-ui frontend is reused (`milestone/2/frontend`) so the chat contract does not change.

## Tests

```bash
cd milestone/3/backend && bun test && bun run lint
cd python && ruff check infer.py distill.py export_onnx.py test_infer.py && pytest -q
```

## Report

Write-up + architecture diagram: [`report/REPORT.md`](report/REPORT.md) (PDF: [`report/REPORT.pdf`](report/REPORT.pdf)).
