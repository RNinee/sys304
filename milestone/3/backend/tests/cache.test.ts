import { describe, expect, test } from "bun:test";
import { cacheKey, memoryCache } from "../src/cache";
import type { PredictResponse } from "../src/types";

const sample: PredictResponse = {
  target: 0,
  label: "not disaster",
  confidence: 0.7,
  probabilities: { not_disaster: 0.7, disaster: 0.3 },
  model: "mock",
  input: "Love skiing",
};

describe("cacheKey", () => {
  test("is stable for the same tweet", () => {
    expect(cacheKey({ text: "hello" })).toBe(cacheKey({ text: "hello" }));
  });

  test("changes when keyword changes", () => {
    expect(cacheKey({ text: "hello", keyword: "fire" })).not.toBe(
      cacheKey({ text: "hello" }),
    );
  });
});

describe("memoryCache", () => {
  test("round-trips a prediction and strips runtime fields", async () => {
    const cache = memoryCache();
    await cache.set(
      { text: "Love skiing" },
      { ...sample, cached: false, batch_size: 4 },
    );
    const hit = await cache.get({ text: "Love skiing" });
    expect(hit?.label).toBe("not disaster");
    expect(hit?.cached).toBeUndefined();
    expect(hit?.batch_size).toBeUndefined();
  });

  test("returns null on a miss", async () => {
    const cache = memoryCache();
    expect(await cache.get({ text: "missing" })).toBeNull();
  });
});
