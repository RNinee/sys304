import type { PredictRequest, PredictResponse } from "./types";

const DEFAULT_INFER_URL = "http://127.0.0.1:9377";

export function inferUrl(): string {
  return (process.env.INFER_URL ?? DEFAULT_INFER_URL).replace(/\/$/, "");
}

export async function inferPredict(
  body: PredictRequest,
  options: { signal?: AbortSignal } = {},
): Promise<PredictResponse> {
  const url = `${inferUrl()}/predict`;
  const response = await fetch(url, {
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
}> {
  const response = await fetch(`${inferUrl()}/health`);
  if (!response.ok) {
    throw new Error(`infer worker health ${response.status}`);
  }
  return (await response.json()) as { status: string; model: string };
}
