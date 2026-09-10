import type { PredictRequest, PredictResponse } from "./types";

const DEFAULT_INFER_URL = "http://127.0.0.1:9377";

export function inferUrl(): string {
  return inferUrls()[0];
}

export function inferUrls(): string[] {
  const many = process.env.INFER_URLS;
  if (many) {
    return many
      .split(",")
      .map((part) => part.trim().replace(/\/$/, ""))
      .filter(Boolean);
  }
  const workers = Math.max(1, Number(process.env.INFER_WORKERS ?? 1));
  const baseUrl = (process.env.INFER_URL ?? DEFAULT_INFER_URL).replace(
    /\/$/,
    "",
  );
  if (workers <= 1) return [baseUrl];
  const parsed = new URL(baseUrl);
  const basePort = Number(parsed.port || "9377");
  return Array.from({ length: workers }, (_, i) => {
    parsed.port = String(basePort + i);
    return parsed.origin;
  });
}

let nextWorker = 0;

function nextInferUrl(): string {
  const urls = inferUrls();
  const url = urls[nextWorker % urls.length];
  nextWorker += 1;
  return url;
}

export async function inferPredict(
  body: PredictRequest,
  options: { signal?: AbortSignal } = {},
): Promise<PredictResponse> {
  const results = await inferPredictBatch([body], options);
  return results[0];
}

export async function inferPredictBatch(
  items: PredictRequest[],
  options: { signal?: AbortSignal } = {},
): Promise<PredictResponse[]> {
  if (items.length === 0) return [];
  if (items.length === 1) {
    return [await inferOne(nextInferUrl(), items[0], options)];
  }
  const url = nextInferUrl();
  const response = await fetch(`${url}/predict_batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
    signal: options.signal,
  });
  const payload = (await response.json()) as {
    results?: PredictResponse[];
    error?: string;
  };
  if (!response.ok || !payload.results) {
    throw new Error(
      payload.error ?? `infer worker returned ${response.status}`,
    );
  }
  return payload.results;
}

async function inferOne(
  url: string,
  body: PredictRequest,
  options: { signal?: AbortSignal },
): Promise<PredictResponse> {
  const response = await fetch(`${url}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal,
  });
  const payload = (await response.json()) as PredictResponse & {
    error?: string;
  };
  if (!response.ok) {
    throw new Error(
      payload.error ?? `infer worker returned ${response.status}`,
    );
  }
  return payload;
}

export async function inferHealth(): Promise<{
  status: string;
  model: string;
  backend?: string;
  device?: string;
  optimizations?: string[];
}> {
  const response = await fetch(`${inferUrl()}/health`);
  if (!response.ok) {
    throw new Error(`infer worker health ${response.status}`);
  }
  return (await response.json()) as {
    status: string;
    model: string;
    backend?: string;
    device?: string;
    optimizations?: string[];
  };
}
