# Milestone 2 — Deploying the model service

SYS-304 · due Sep 11, 2026

Wrap the Phase 1 **Qwen2.5-1.5B LoRA** disaster-tweet classifier in a **Bun/Elysia** API, a **Next.js + [assistant-ui](https://www.assistant-ui.com/)** chat UI, Docker, and GitHub Actions CI.

## Run the stack

From the repo root (one command):

```bash
chmod +x deploy.sh
./deploy.sh local     # host: UI http://localhost:3000  API http://localhost:8000
# or
./deploy.sh           # docker compose up --build
```

`./deploy.sh local` is the one to use for the Canvas screen recording: the same terminal prints `[predict] …` lines as the UI sends tweets. The first start loads the 1.5B base model from the Hugging Face cache (can take a minute).

Uses `.venv-ml` when present (`PYTHON=.venv-ml/bin/python` otherwise). Needs the Milestone 1 adapter at `milestone/1/models/qwen2.5-1.5b-disaster-lora/`.

## Architecture

```
browser  --POST /predict-->  Elysia :8000  -->  Python infer worker :9377
         (assistant-ui Thread)               (Qwen2.5-1.5B LoRA adapter)
```

| Piece | Path | Role |
| --- | --- | --- |
| API | `backend/` | Elysia + Bun. `POST /predict`, `POST /chat`, `GET /health` |
| Infer worker | `backend/python/infer.py` | Loads the saved Qwen 1.5B LoRA sequence classifier |
| UI | `frontend/` | Next.js App Router + assistant-ui `Thread` + `useLocalRuntime` |
| CI | `.github/workflows/ci.yml` | ruff, biome, eslint, unit tests, API integration tests, Playwright |

The worker loads `milestone/1/models/qwen2.5-1.5b-disaster-lora/` (the same adapters as Milestone 1). The 1.5B base weights stay in the Hugging Face cache (`Qwen/Qwen2.5-1.5B-Instruct`). Docker uses CPU torch and a named volume for that cache so the download happens once.

## API

`POST /predict`

```json
{ "text": "Forest fire near La Ronge Sask. Canada", "keyword": "forest fire" }
```

```json
{
  "target": 1,
  "label": "disaster",
  "confidence": 0.91,
  "probabilities": { "not_disaster": 0.09, "disaster": 0.91 },
  "model": "qwen2.5-1.5b-lora",
  "input": "keyword: forest fire\nForest fire near La Ronge Sask. Canada"
}
```

`keyword` is optional. When present, the worker prefixes the tweet the same way Milestone 1 training did (`keyword: …\\n` + body).

## Tests

```bash
# backend (unit + /predict integration against a mock infer worker)
cd milestone/2/backend && bun test && bun run lint
cd python && ruff check infer.py test_infer.py && pytest -q

# frontend
cd milestone/2/frontend && bun test src && bun run lint
bunx playwright install chromium
bun run test:e2e
```

CI does not download the 1.5B weights. API integration tests talk to a stub worker with the same `/predict` contract; the live Qwen model is what `./deploy.sh local` serves.

## Docker

`backend/Dockerfile` and `frontend/Dockerfile` are composed by `docker-compose.yml`. Browser calls `http://localhost:8000/predict` (CORS enabled) so API logs stay visible next to the UI.
