import { type Subprocess, spawn } from "bun";
import { app } from "./app";
import { inferHealth, inferUrl } from "./infer";

const PORT = Number(process.env.PORT ?? 8000);
const HOST = process.env.HOST ?? "0.0.0.0";
const MANAGE_WORKER = process.env.MANAGE_INFER_WORKER !== "0";

function pythonBin(): string {
  return process.env.PYTHON ?? process.env.INFER_PYTHON ?? "python3";
}

function inferScript(): string {
  return process.env.INFER_SCRIPT ?? `${import.meta.dir}/../python/infer.py`;
}

async function waitForWorker(timeoutMs = 180_000): Promise<void> {
  const start = Date.now();
  let lastError = "timeout";
  while (Date.now() - start < timeoutMs) {
    try {
      await inferHealth();
      return;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      await Bun.sleep(200);
    }
  }
  throw new Error(
    `infer worker at ${inferUrl()} did not become ready: ${lastError}`,
  );
}

let worker: Subprocess | null = null;

async function startWorker(): Promise<void> {
  try {
    await inferHealth();
    console.log(`[api] infer worker already running at ${inferUrl()}`);
    return;
  } catch {
    // spawn our own
  }
  const python = pythonBin();
  const script = inferScript();
  console.log(`[api] starting infer worker: ${python} ${script}`);
  worker = spawn({
    cmd: [python, script],
    stdout: "inherit",
    stderr: "inherit",
    env: {
      ...process.env,
      INFER_HOST: process.env.INFER_HOST ?? "127.0.0.1",
      INFER_PORT: process.env.INFER_PORT ?? "9377",
    },
  });
  await waitForWorker();
}

if (import.meta.main) {
  if (MANAGE_WORKER) {
    await startWorker();
  } else {
    await waitForWorker();
  }

  const server = app.listen({ port: PORT, hostname: HOST });
  console.log(`[api] listening on http://${HOST}:${PORT}`);

  const shutdown = () => {
    console.log("[api] shutting down");
    worker?.kill();
    server.stop(true);
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

export { app };
