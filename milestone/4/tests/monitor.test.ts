import { describe, expect, test } from "bun:test";
import type {
  PredictRequest,
  PredictResponse,
} from "../../3/backend/src/types";
import { fakeTweets, labelsCsv } from "../scripts/fake_traffic";
import { driftedTweets, loadTrainTweets } from "../scripts/workload";
import { createMonitorApp } from "../src/app";
import { baselineFromCsv } from "../src/baseline";
import { totalVariation } from "../src/drift";
import { RequestLog } from "../src/log";
import { renderPrometheus } from "../src/prometheus";

const baseline = baselineFromCsv(
  [
    "text,keyword,target",
    "Forest fire near town,fire,1",
    "I love pizza,,0",
    "Flood warning tonight,flood,1",
    "The movie was funny,,0",
  ].join("\n"),
);

const prediction: PredictResponse = {
  target: 1,
  label: "disaster",
  confidence: 0.91,
  probabilities: { not_disaster: 0.09, disaster: 0.91 },
  model: "mock",
  input: "Forest fire near town",
};

function infer(body: PredictRequest): Promise<PredictResponse> {
  return Promise.resolve({ ...prediction, input: body.text });
}

describe("request log", () => {
  test("stores the prediction and leaves truth empty", () => {
    const log = new RequestLog(":memory:");
    const app = createMonitorApp({ infer, log, baseline });
    return app
      .handle(
        new Request("http://localhost/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: "Forest fire near town",
            keyword: "fire",
          }),
        }),
      )
      .then(async (response) => {
        expect(response.status).toBe(200);
        const rows = log.recent(5);
        expect(rows).toHaveLength(1);
        expect(rows[0]?.label).toBe("disaster");
        expect(rows[0]?.confidence).toBe(0.91);
        expect(rows[0]?.truth_target).toBeNull();
        expect(rows[0]?.keyword).toBe("fire");
        log.close();
      });
  });

  test("logs rejected requests so the error rate is visible", async () => {
    const log = new RequestLog(":memory:");
    const app = createMonitorApp({ infer, log, baseline });
    const response = await app.handle(
      new Request("http://localhost/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{",
      }),
    );
    expect(response.status).toBe(400);
    expect(log.recent(1)[0]?.status).toBe(400);
    log.close();
  });
});

describe("prometheus", () => {
  test("exposes counters, a latency histogram, and drift gauges", async () => {
    const log = new RequestLog(":memory:");
    let reloads = 0;
    const app = createMonitorApp({
      infer,
      log,
      baseline,
      onReload: async () => {
        reloads += 1;
      },
    });
    for (const item of driftedTweets(20)) {
      await app.handle(
        new Request("http://localhost/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item),
        }),
      );
    }
    const body = await (
      await app.handle(new Request("http://localhost/prometheus"))
    ).text();
    expect(body).toContain(
      'api_requests_total{route="/predict",code="200"} 20',
    );
    expect(body).toContain("api_request_duration_seconds_bucket");
    expect(body).toContain("drift_keyword_tvd");
    expect(body).toContain("drift_text_length_z");
    expect(body).toContain("drift_live_samples 20");
    const reload = await app.handle(
      new Request("http://localhost/admin/reload", { method: "POST" }),
    );
    expect(reload.status).toBe(200);
    expect(reloads).toBe(1);
    log.close();
  });

  test("renderPrometheus stays finite on an empty log", () => {
    const text = renderPrometheus([], baseline);
    expect(text).toContain("drift_train_disaster_rate 0.5");
    expect(text).not.toContain("NaN");
  });
});

describe("drift and workload helpers", () => {
  test("identical keyword shares have zero total variation", () => {
    const shares = { fire: 0.5, "(none)": 0.5 };
    expect(totalVariation(shares, shares)).toBe(0);
  });

  test("workload can read a tiny training csv", () => {
    const tweets = loadTrainTweets(
      "text,keyword,target\nhello there,quake,1\n",
    );
    expect(tweets).toEqual([{ text: "hello there", keyword: "quake" }]);
    expect(driftedTweets(3)).toHaveLength(3);
  });

  test("fake traffic carries a label for every posted text", () => {
    const items = fakeTweets(10);
    expect(items).toHaveLength(10);
    const csv = labelsCsv(items);
    expect(csv.startsWith("text,target\n")).toBe(true);
    expect(csv.trim().split("\n")).toHaveLength(11);
    expect(items.filter((item) => item.target === 1).length).toBeGreaterThan(0);
    expect(items.filter((item) => item.target === 0).length).toBeGreaterThan(0);
  });
});
