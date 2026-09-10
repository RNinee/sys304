/**
 * System-level HTTP benchmark: sequential vs concurrent, unique vs repeated tweets.
 *
 *   bun run benchmarks/bench_api.ts
 *   bun run benchmarks/bench_api.ts --url http://127.0.0.1:8000 --concurrency 8
 */
import tweetsJson from "./tweets.json";

type Tweet = { text: string; keyword?: string };
type PredictResponse = {
  label: string;
  cached?: boolean;
  batch_size?: number;
};

const tweets = tweetsJson as Tweet[];

function arg(name: string, fallback: string): string {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return process.argv[idx + 1];
  return fallback;
}

const URL_BASE = arg("url", "http://127.0.0.1:8000").replace(/\/$/, "");
const CONCURRENCY = Number(arg("concurrency", "8"));
const REPEATS = Number(arg("repeats", "2"));

async function predict(body: Tweet): Promise<{
  ms: number;
  cached: boolean;
  batch: number;
}> {
  const start = performance.now();
  const res = await fetch(`${URL_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = (await res.json()) as PredictResponse & { error?: string };
  if (!res.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
  return {
    ms: performance.now() - start,
    cached: Boolean(json.cached),
    batch: json.batch_size ?? 1,
  };
}

async function runSequential(items: Tweet[]) {
  const times: number[] = [];
  let hits = 0;
  const t0 = performance.now();
  for (const item of items) {
    const r = await predict(item);
    times.push(r.ms);
    if (r.cached) hits += 1;
  }
  return summarize("sequential-unique", times, hits, performance.now() - t0);
}

async function runConcurrent(items: Tweet[], concurrency: number) {
  const times: number[] = [];
  let hits = 0;
  let batches = 0;
  let i = 0;
  const t0 = performance.now();
  async function worker() {
    while (i < items.length) {
      const idx = i;
      i += 1;
      const r = await predict(items[idx]);
      times.push(r.ms);
      if (r.cached) hits += 1;
      batches += r.batch;
    }
  }
  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return {
    ...summarize(`concurrent-${concurrency}`, times, hits, performance.now() - t0),
    mean_reported_batch: Number(
      (batches / Math.max(1, times.length)).toFixed(2),
    ),
  };
}

function summarize(
  name: string,
  times: number[],
  hits: number,
  wallMs: number,
) {
  const ordered = [...times].sort((a, b) => a - b);
  const sum = times.reduce((a, b) => a + b, 0);
  const p = (q: number) =>
    ordered[Math.min(ordered.length - 1, Math.floor((q / 100) * ordered.length))];
  return {
    name,
    n: times.length,
    cache_hits: hits,
    mean_ms: Number((sum / times.length).toFixed(2)),
    p50_ms: Number(p(50).toFixed(2)),
    p95_ms: Number(p(95).toFixed(2)),
    wall_ms: Number(wallMs.toFixed(2)),
    qps: Number((times.length / (wallMs / 1000)).toFixed(3)),
  };
}

const health = await fetch(`${URL_BASE}/health`);
if (!health.ok) {
  console.error(`API not healthy at ${URL_BASE}/health`);
  process.exit(1);
}
const healthJson = await health.json();
console.log("health", healthJson);

const runId = Date.now();
function tag(items: Tweet[], suffix: string): Tweet[] {
  return items.map((item) => ({
    ...item,
    text: `${item.text} [${runId}${suffix}]`,
  }));
}

const seqItems = tag(tweets, "-seq");
const concItems = tag(tweets, "-conc");
const cacheItems = Array.from(
  { length: tweets.length * REPEATS },
  (_, i) => seqItems[i % seqItems.length],
);

const seq = await runSequential(seqItems);
const conc = await runConcurrent(concItems, CONCURRENCY);
const cached = await runSequential(cacheItems);
const metrics = await fetch(`${URL_BASE}/metrics`).then((r) => r.json());

const report = { health: healthJson, seq, conc, cached, metrics };
console.log(JSON.stringify(report, null, 2));

const outDir = new URL("./results/", import.meta.url);
await Bun.write(new URL("api-latest.json", outDir), JSON.stringify(report, null, 2));
console.log("wrote benchmarks/results/api-latest.json");
