/**
 * Submit a mix of in-distribution tweets and drifted inputs to the API.
 *
 *   bun run scripts/workload.ts
 *   bun run scripts/workload.ts --url http://127.0.0.1:8000 --count 40 --drift 0.5 --errors 2
 */
import { readFileSync } from "node:fs";
import { parseCsv } from "../src/baseline";

type Tweet = { text: string; keyword?: string };

function arg(name: string, fallback: string): string {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1])
    return process.argv[idx + 1] ?? fallback;
  return fallback;
}

export function loadTrainTweets(csv: string): Tweet[] {
  const rows = parseCsv(csv);
  const header = rows[0] ?? [];
  const textAt = header.indexOf("text");
  const keywordAt = header.indexOf("keyword");
  return rows.slice(1).flatMap((cells) => {
    const text = (cells[textAt] ?? "").trim();
    if (!text) return [];
    const keyword = keywordAt >= 0 ? (cells[keywordAt] ?? "").trim() : "";
    return keyword ? [{ text, keyword }] : [{ text }];
  });
}

export function driftedTweets(count: number): Tweet[] {
  const items: Tweet[] = [];
  for (let i = 0; i < count; i++) {
    const kind = i % 3;
    if (kind === 0) {
      items.push({
        text: "ok ".repeat(80).trim(),
        keyword: "not-a-training-keyword",
      });
    } else if (kind === 1) {
      items.push({ text: "x", keyword: `synthetic-${i}` });
    } else {
      items.push({ text: `${"drift ".repeat(40)} ${i}` });
    }
  }
  return items;
}

function sample<T>(items: T[], count: number): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const left = copy[i];
    const right = copy[j];
    if (left === undefined || right === undefined) continue;
    copy[i] = right;
    copy[j] = left;
  }
  return copy.slice(0, count);
}

async function post(url: string, body: string, bad = false): Promise<number> {
  const response = await fetch(`${url}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: bad ? "{" : body,
  });
  return response.status;
}

async function main() {
  const url = arg("url", "http://127.0.0.1:8000").replace(/\/$/, "");
  const count = Number(arg("count", "40"));
  const driftShare = Number(arg("drift", "0.5"));
  const errors = Number(arg("errors", "2"));
  const trainPath =
    process.env.TRAIN_CSV ?? `${import.meta.dir}/../../1/data/train.csv`;
  const train = loadTrainTweets(readFileSync(trainPath, "utf8"));
  const drifted = Math.round(count * driftShare);
  const live = Math.max(0, count - drifted);
  const batch = [...sample(train, live), ...driftedTweets(drifted)];

  let ok = 0;
  let failed = 0;
  for (const item of batch) {
    const status = await post(url, JSON.stringify(item));
    if (status >= 200 && status < 300) ok += 1;
    else failed += 1;
  }
  for (let i = 0; i < errors; i++) {
    const status = await post(url, "", true);
    if (status >= 400) failed += 1;
    else ok += 1;
  }
  console.log(
    JSON.stringify({ url, sent: batch.length + errors, ok, failed }, null, 2),
  );
}

if (import.meta.main) {
  await main();
}
