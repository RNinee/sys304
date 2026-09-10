import { type Subprocess, spawn } from "bun";
import { createApp } from "./app";
import { DynamicBatcher } from "./batcher";
import { createCache } from "./cache";
import { inferHealth, inferPredictBatch, inferUrl } from "./infer";

const PORT = Number(process.env.PORT ?? 8000);
const HOST = process.env.HOST ?? "0.0.0.0";
const MANAGE_WORKER = process.env.MANAGE_INFER_WORKER !== "0";
const INFER_WORKERS = Math.max(1, Number(process.env.INFER_WORKERS ?? 1));

function pythonBin(): string {
  return process.env.PYTHON ?? process.env.INFER_PYTHON ?? "python3";
}

function inferScript(): string {
  return process.env.INFER_SCRIPT ?? `${import.meta.dir}/../python/infer.py`;
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
      // spawn
    }
    console.log(
      `[api] starting infer worker ${i + 1}/${INFER_WORKERS}: ${python} ${script} :${port}`,
    );
    workers.push(
      spawn({
        cmd: [python, script],
        stdout: "inherit",
        stderr: "inherit",
        env: {
          ...process.env,
          INFER_HOST: host,
          INFER_PORT: String(port),
        },
      }),
    );
    await waitForWorker(url);
  }
}

const batcher = new DynamicBatcher(inferPredictBatch);
const cache = createCache();
const app = createApp({
  infer: (body, options) => batcher.enqueue(body, options?.signal),
  cache,
});

if (import.meta.main) {
  if (MANAGE_WORKER) {
    await startWorkers();
  } else {
    await waitForWorker(inferUrl());
  }

  const workerInfo = await inferHealth().catch(() => null);
  const server = app.listen({ port: PORT, hostname: HOST });
  console.log(
    `[api] listening on http://${HOST}:${PORT} cache=${cache.status} workers=${INFER_WORKERS} model=${workerInfo?.model ?? "unknown"}`,
  );

  const shutdown = () => {
    console.log("[api] shutting down");
    for (const worker of workers) worker.kill();
    server.stop(true);
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

export { app };
