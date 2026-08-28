import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createApp } from "../src/app";
import { inferPredict } from "../src/infer";

const INFER_PORT = 19377;
const MODEL = "qwen2.5-1.5b-lora";

function classify(text: string, keyword?: string) {
  const hay = `${keyword ?? ""} ${text}`.toLowerCase();
  const disaster =
    /fire|earthquake|crash|flood|collapse|wildfire|hurricane/.test(hay);
  const p = disaster ? 0.91 : 0.12;
  return {
    target: disaster ? 1 : 0,
    label: disaster ? "disaster" : "not disaster",
    confidence: disaster ? p : 1 - p,
    probabilities: { not_disaster: 1 - p, disaster: p },
    model: MODEL,
    input: text,
  };
}

describe("API integration", () => {
  let server: ReturnType<typeof Bun.serve> | null = null;

  beforeAll(() => {
    process.env.INFER_URL = `http://127.0.0.1:${INFER_PORT}`;
    server = Bun.serve({
      hostname: "127.0.0.1",
      port: INFER_PORT,
      async fetch(request) {
        const path = new URL(request.url).pathname;
        if (request.method === "GET" && (path === "/health" || path === "/")) {
          return Response.json({
            status: "ok",
            model: MODEL,
            role: "infer-worker",
          });
        }
        if (request.method === "POST" && path === "/predict") {
          const body = (await request.json()) as {
            text?: string;
            keyword?: string;
          };
          return Response.json(classify(body.text ?? "", body.keyword));
        }
        return Response.json({ error: "not found" }, { status: 404 });
      },
    });
  });

  afterAll(() => {
    server?.stop(true);
  });

  const app = createApp(inferPredict);

  test("POST /predict returns 200 and a disaster label on valid input", async () => {
    const res = await app.handle(
      new Request("http://localhost/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "Forest fire near La Ronge Sask. Canada",
          keyword: "forest fire",
        }),
      }),
    );
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      target: number;
      label: string;
      confidence: number;
      model: string;
    };
    expect(json.label).toBe("disaster");
    expect(json.target).toBe(1);
    expect(json.confidence).toBeGreaterThan(0.5);
    expect(json.model).toBe(MODEL);
  });

  test("POST /predict returns 200 for a clearly non-disaster tweet", async () => {
    const res = await app.handle(
      new Request("http://localhost/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "I love this movie so much, what a blast at the premiere",
        }),
      }),
    );
    expect(res.status).toBe(200);
    const json = (await res.json()) as { label: string; target: number };
    expect(json.label).toBe("not disaster");
    expect(json.target).toBe(0);
  });

  test("POST /chat returns markdown wrapping a prediction", async () => {
    const res = await app.handle(
      new Request("http://localhost/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "Just happened a terrible car crash" },
              ],
            },
          ],
        }),
      }),
    );
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      text: string;
      prediction: { label: string };
    };
    expect(json.text).toContain("disaster");
    expect(json.prediction.label).toBeDefined();
  });
});
