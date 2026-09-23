import { type Subprocess, spawn } from "bun";
import { DynamicBatcher } from "../../3/backend/src/batcher";
import { createCache } from "../../3/backend/src/cache";
import {
  inferHealth,
  inferPredictBatch,
  inferUrl,
} from "../../3/backend/src/infer";
import { createMonitorApp } from "./app";

const PORT = Number(process.env.PORT ?? 8000);
const HOST = process.env.HOST ?? "0.0.0.0";
const MANAGE_WORKER = process.env.MANAGE_INFER_WORKER !== "0";
const INFER_WORKERS = Math.max(1, Number(process.env.INFER_WORKERS ?? 1));

function pythonBin(): string {
  return process.env.PYTHON ?? process.env.INFER_PYTHON ?? "python3";
}

function inferScript(): string {
  return (
    process.env.INFER_SCRIPT ??
    `${import.meta.dir}/../../3/backend/python/infer.py`
  );
}

async function waitForWorker(url: string, timeoutMs = 180_000): Promise<void> {
  const start = Date.now();
  let lastError = "timeout";
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(`${url}/health`);
      if (response.ok) return;
      lastError = `health ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await Bun.sleep(200);
  }
  throw new Error(`infer worker at ${url} did not become ready: ${lastError}`);
}

const workers: Subprocess[] = [];

function stopWorkers(): void {
  for (const worker of workers) worker.kill();
  workers.length = 0;
}

async function startWorkers(): Promise<void> {
  const script = inferScript();
  const python = pythonBin();
  const basePort = Number(process.env.INFER_PORT ?? "9377");
  const host = process.env.INFER_HOST ?? "127.0.0.1";

  for (let i = 0; i < INFER_WORKERS; i++) {
    const port = basePort + i;
    const url = `http://${host}:${port}`;
    try {
      const response = await fetch(`${url}/health`);
      if (response.ok) {
        console.log(`[api] infer worker already running at ${url}`);
        continue;
      }
    } catch {
      // spawn below
    }
    console.log(
      `[api] starting infer worker ${i + 1}/${INFER_WORKERS}: ${python} ${script} :${port}`,
    );
    workers.push(
      spawn({
        cmd: [python, script],
        stdout: "inherit",
        stderr: "inherit",
        env: { ...process.env, INFER_HOST: host, INFER_PORT: String(port) },
      }),
    );
    await waitForWorker(url);
  }
}

async function reloadWorkers(): Promise<void> {
  stopWorkers();
  await Bun.sleep(300);
  await startWorkers();
}

const batcher = new DynamicBatcher(inferPredictBatch);
const app = createMonitorApp({
  infer: (body, options) => batcher.enqueue(body, options?.signal),
  cache: createCache(),
  onReload: MANAGE_WORKER ? reloadWorkers : undefined,
});

if (import.meta.main) {
  if (MANAGE_WORKER) await startWorkers();
  else await waitForWorker(inferUrl());

  const workerInfo = await inferHealth().catch(() => null);
  const server = app.listen({ port: PORT, hostname: HOST });
  console.log(
    `[api] listening on http://${HOST}:${PORT} model=${workerInfo?.model ?? "unknown"}`,
  );
  console.log(`[api] prometheus http://${HOST}:${PORT}/prometheus`);
  console.log("[api] grafana http://127.0.0.1:3001");

  const shutdown = () => {
    console.log("[api] shutting down");
    stopWorkers();
    server.stop(true);
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

export { app };
