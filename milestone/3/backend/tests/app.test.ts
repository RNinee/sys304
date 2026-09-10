import { describe, expect, test } from "bun:test";
import { createApp } from "../src/app";
import { memoryCache } from "../src/cache";
import { resetStats, stats } from "../src/stats";
import type { PredictRequest, PredictResponse } from "../src/types";

const fakeResult: PredictResponse = {
  target: 1,
  label: "disaster",
  confidence: 0.88,
  probabilities: { not_disaster: 0.12, disaster: 0.88 },
  model: "mock",
  input: "Forest fire near La Ronge Sask. Canada",
};

function mockInfer(body: PredictRequest): Promise<PredictResponse> {
  return Promise.resolve({ ...fakeResult, input: body.text });
}

describe("API unit", () => {
  const app = createApp(mockInfer);

  test("GET /health returns ok", async () => {
    const res = await app.handle(new Request("http://localhost/health"));
    expect(res.status).toBe(200);
    const json = (await res.json()) as { status: string; role: string };
    expect(json.status).toBe("ok");
    expect(json.role).toBe("api");
  });

  test("GET /metrics returns counters", async () => {
    const res = await app.handle(new Request("http://localhost/metrics"));
    expect(res.status).toBe(200);
    const json = (await res.json()) as { cacheHits: number };
    expect(typeof json.cacheHits).toBe("number");
  });

  test("POST /predict returns 400 on empty text", async () => {
    const res = await app.handle(
      new Request("http://localhost/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "" }),
      }),
    );
    expect(res.status).toBe(400);
    const json = (await res.json()) as { error: string };
    expect(json.error).toContain("text");
  });

  test("POST /predict returns 400 on invalid JSON", async () => {
    const res = await app.handle(
      new Request("http://localhost/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not-json",
      }),
    );
    expect(res.status).toBe(400);
  });
});

describe("API cache", () => {
  test("second identical request is served from cache", async () => {
    resetStats();
    let calls = 0;
    const infer = (body: PredictRequest) => {
      calls += 1;
      return mockInfer(body);
    };
    const app = createApp({ infer, cache: memoryCache() });
    const request = () =>
      app.handle(
        new Request("http://localhost/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: "Forest fire near La Ronge Sask. Canada",
          }),
        }),
      );

    const first = await request();
    const second = await request();
    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    const a = (await first.json()) as PredictResponse;
    const b = (await second.json()) as PredictResponse;
    expect(a.cached).toBe(false);
    expect(b.cached).toBe(true);
    expect(b.label).toBe(a.label);
    expect(calls).toBe(1);
    expect(stats.cacheHits).toBe(1);
    expect(stats.cacheMisses).toBe(1);
  });
});
