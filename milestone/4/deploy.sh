#!/usr/bin/env bash
# Milestone 4: v3 API plus SQLite logging, Prometheus, and Grafana.
#   ./deploy.sh local
#   ./deploy.sh
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

  if [[ ! -f "$ROOT/../3/backend/node_modules/elysia/package.json" ]]; then
    (cd "$ROOT/../3/backend" && bun install)
  fi
  if [[ ! -f "$ROOT/node_modules/bun-types/package.json" ]]; then
    (cd "$ROOT" && bun install)
  fi
  if [[ ! -f "$ROOT/../3/frontend/node_modules/next/package.json" ]]; then
    (cd "$ROOT/../3/frontend" && bun install)
  fi

  (cd "$ROOT" && docker compose up -d)
  export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"

  cleanup() {
    trap - INT TERM EXIT
    [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
    wait 2>/dev/null || true
  }
  trap cleanup INT TERM EXIT

  echo "API        http://localhost:8000"
  echo "Prometheus http://localhost:9090"
  echo "Grafana    http://localhost:3001/d/sys304-monitor"
  echo "UI         http://localhost:3000"
  echo

  (
    cd "$ROOT"
    PYTHON="$PYTHON" \
      INFER_BACKEND="${INFER_BACKEND:-auto}" \
      bun run src/index.ts
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

  cd "$ROOT/../3/frontend"
  bun run dev
  exit 0
fi

if [[ "$MODE" != "docker" ]]; then
  echo "Usage: $0 [docker|local]" >&2
  exit 1
fi

cd "$ROOT"
docker compose up
