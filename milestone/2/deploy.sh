#!/usr/bin/env bash
# Launch the Milestone 2 stack.
#   ./deploy.sh          docker compose (frontend :3000, API :8000)
#   ./deploy.sh local    bun + python on the host (better for the demo recording)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
MODE="${1:-docker}"

if [[ "$MODE" == "local" ]]; then
  PYTHON="${PYTHON:-}"
  if [[ -z "$PYTHON" && -x "$REPO/.venv-ml/bin/python" ]]; then
    PYTHON="$REPO/.venv-ml/bin/python"
  elif [[ -z "$PYTHON" ]]; then
    PYTHON="python3"
  fi

  if ! "$PYTHON" -c "import torch, transformers, peft" >/dev/null 2>&1; then
    echo "Installing Python inference deps with $PYTHON ..."
    "$PYTHON" -m pip install torch
    "$PYTHON" -m pip install -r "$ROOT/backend/python/requirements.txt"
  fi

  if [[ ! -f "$ROOT/backend/node_modules/elysia/package.json" ]]; then
    (cd "$ROOT/backend" && bun install)
  fi
  if [[ ! -f "$ROOT/frontend/node_modules/next/package.json" ]]; then
    (cd "$ROOT/frontend" && bun install)
  fi

  cleanup() {
    trap - INT TERM EXIT
    [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
    wait 2>/dev/null || true
  }
  trap cleanup INT TERM EXIT

  echo "API  http://localhost:8000"
  echo "UI   http://localhost:3000"
  echo "POST /predict — Qwen2.5-1.5B LoRA (first load can take a minute)"
  echo

  (
    cd "$ROOT/backend"
    PYTHON="$PYTHON" bun run src/index.ts
  ) &
  API_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "API did not become healthy on :8000" >&2
    exit 1
  fi

  cd "$ROOT/frontend"
  bun run dev
  exit 0
fi

if [[ "$MODE" != "docker" ]]; then
  echo "Usage: $0 [docker|local]" >&2
  exit 1
fi

cd "$ROOT"
docker compose up --build
