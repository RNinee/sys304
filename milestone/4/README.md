# Milestone 4 — Observability and retraining

SYS-304. v4 keeps the v3 classifier and adds a request log, Prometheus, Grafana, and a DistilBERT retraining script.

| Version | Adds |
| --- | --- |
| v1 | Qwen2.5-1.5B LoRA classifier in `milestone/1` |
| v2 | Bun/Elysia API and the chat UI |
| v3 | Quantization, distillation, ONNX, batching, Redis |
| v4 | SQLite request log, Prometheus + Grafana, retraining |

## Run

```bash
chmod +x milestone/4/deploy.sh
./milestone/4/deploy.sh local
```

| Surface | URL |
| --- | --- |
| Chat UI | http://localhost:3000 |
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001/d/sys304-monitor |

Grafana signs in anonymously as a viewer. The admin user is `admin` / `admin`. Prometheus scrapes `GET /prometheus` on the host API.

## Week 7 — logging, metrics, drift

Every `/predict` and `/chat` call is written to `milestone/4/data/requests.sqlite` (text, keyword, latency, status, label, confidence). Ground truth stays empty until a later labeling step fills `truth_target`.

`GET /prometheus` turns that log into counters, a latency histogram, mean confidence, and drift gauges. Drift compares the last 200 successful requests with `milestone/1/data/train.csv`: text length, word count, keyword mix (total variation distance), and the live predicted disaster rate against the training label rate.

To open Grafana without loading the model:

```bash
cd milestone/4
bun run scripts/preview.ts
```

Generate traffic:

```bash
cd milestone/4
bun run workload -- --url http://127.0.0.1:8000 --count 40 --drift 0.5 --errors 2
```

Half of the tweets are real training rows. Half are short, long, or use keywords that never appear in training, so the drift panels move. `--errors` sends broken JSON so the error ratio is not stuck at zero.

`scripts/fake_traffic.ts` posts made-up user tweets and writes `data/fake_labels.csv` with a `text,target` label for each one. The API still stores those rows with an empty `truth_target`. Attach the file before a retrain check:

```bash
cd milestone/4
bun run fake-traffic -- --url http://127.0.0.1:8000 --count 30 --errors 2
python python/retrain.py --labels data/fake_labels.csv --check
```

## Week 8 — retraining

`python/retrain.py` fine-tunes the DistilBERT student already served in v3. It starts a cycle when **either**:

- the last cycle is older than `--interval-minutes` (default 24 hours), or
- mean logged confidence is below `--confidence-floor` (default 0.55).

`--mean-confidence 0.4` simulates the confidence drop. `--force` skips both checks.

Labels are not invented by the model. Attach them from a CSV (`text,target`) or, for a demo, from exact matches against `train.csv`:

```bash
cd milestone/4
python python/retrain.py --apply-train-labels --check
python python/retrain.py --force --epochs 1 --max-samples 64
```

The script holds out 20% of the labeled log, scores the live student and the new one on that split, and promotes only when the new disaster F1 is higher. Promotion copies the checkpoint over `milestone/3/models/distilbert-student`, exports ONNX beside it, and `POST /admin/reload` restarts the Python infer worker. A failed export restores the previous weights.

## Tests

```bash
cd milestone/4 && bun test && python -m unittest discover -s python -p 'test_*.py'
cd milestone/3/backend && bun test
```

Design write-up: [`report/REPORT.md`](report/REPORT.md).
