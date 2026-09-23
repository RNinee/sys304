/**
 * Serve the monitor API with a fake model so Grafana can be opened
 * without loading Qwen or DistilBERT.
 *
 *   bun run scripts/preview.ts
 */
import { createMonitorApp } from "../src/app";
import { RequestLog } from "../src/log";
import { driftedTweets } from "./workload";

const prediction = {
  target: 1 as const,
  label: "disaster" as const,
  confidence: 0.42,
  probabilities: { not_disaster: 0.58, disaster: 0.42 },
  model: "preview",
  input: "",
};

const app = createMonitorApp({
  infer: async (body) => ({ ...prediction, input: body.text }),
  log: new RequestLog(":memory:"),
});

const port = Number(process.env.PORT ?? 8000);
for (const item of driftedTweets(24)) {
  await app.handle(
    new Request("http://localhost/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(item),
    }),
  );
}
await app.handle(
  new Request("http://localhost/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{",
  }),
);

app.listen({ port, hostname: "0.0.0.0" });
console.log(`preview API http://127.0.0.1:${port}`);
console.log(`prometheus http://127.0.0.1:${port}/prometheus`);
