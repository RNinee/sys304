import { describe, expect, test } from "bun:test";
import { DynamicBatcher } from "../src/batcher";
import { resetStats, stats } from "../src/stats";
import type { PredictRequest, PredictResponse } from "../src/types";

function result(text: string): PredictResponse {
  return {
    target: 1,
    label: "disaster",
    confidence: 0.9,
    probabilities: { not_disaster: 0.1, disaster: 0.9 },
    model: "mock",
    input: text,
  };
}

describe("DynamicBatcher", () => {
  test("groups concurrent requests into one infer batch", async () => {
    resetStats();
    const seen: number[] = [];
    const batcher = new DynamicBatcher(
      async (items: PredictRequest[]) => {
        seen.push(items.length);
        await Bun.sleep(5);
        return items.map((item) => result(item.text));
      },
      8,
      20,
    );

    const texts = ["a", "b", "c", "d"];
    const results = await Promise.all(
      texts.map((text) => batcher.enqueue({ text })),
    );

    expect(results.map((r) => r.input)).toEqual(texts);
    expect(results.every((r) => r.batch_size === 4)).toBe(true);
    expect(seen).toEqual([4]);
    expect(stats.batches).toBe(1);
    expect(stats.batchedItems).toBe(4);
  });

  test("flushes immediately once maxSize is reached", async () => {
    resetStats();
    const batcher = new DynamicBatcher(
      async (items: PredictRequest[]) => {
        return items.map((item) => result(item.text));
      },
      2,
      10_000,
    );

    const first = batcher.enqueue({ text: "one" });
    const second = batcher.enqueue({ text: "two" });
    const results = await Promise.all([first, second]);
    expect(results).toHaveLength(2);
    expect(stats.batches).toBe(1);
    expect(stats.batchedItems).toBe(2);
  });
});
