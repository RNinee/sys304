/**
 * Post fake user tweets, then write the ground-truth file those rows need.
 *
 * The API stores the text and the model's guess. It does not fill truth_target.
 * This script writes data/fake_labels.csv so retrain.py can attach labels by
 * exact text match.
 *
 *   bun run scripts/fake_traffic.ts
 *   bun run scripts/fake_traffic.ts -- --url http://127.0.0.1:8000 --count 30 --errors 2
 *
 * Then:
 *   python python/retrain.py --labels data/fake_labels.csv --check
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

export type FakeTweet = {
  text: string;
  keyword?: string;
  target: 0 | 1;
};

const DISASTER = [
  {
    keyword: "earthquake",
    text: "A 6.4 earthquake just hit the coast. Buildings are down and people are trapped.",
  },
  {
    keyword: "flood",
    text: "The river burst its banks overnight. Whole streets are under flood water.",
  },
  {
    keyword: "wildfire",
    text: "Wildfire jumped the highway. Residents are evacuating with the smoke already thick.",
  },
  {
    keyword: "crash",
    text: "Passenger train crash outside the station. Ambulances are still arriving.",
  },
  {
    keyword: "hurricane",
    text: "Hurricane winds tore the roof off the school. Power is out across the town.",
  },
];

const NOT_DISASTER = [
  {
    keyword: "lunch",
    text: "Grabbing lunch downtown. The new noodle place is packed but worth the wait.",
  },
  {
    keyword: "movie",
    text: "Just left the movie. The ending was loud, not an actual disaster.",
  },
  {
    keyword: "weather",
    text: "Beautiful afternoon. Might bike to the park if the clouds stay away.",
  },
  {
    keyword: "concert",
    text: "Concert tonight sold out. The crowd is huge and nobody is in danger.",
  },
  {
    keyword: "game",
    text: "Our team won in overtime. The stadium shook from cheering, not from an earthquake.",
  },
];

export function fakeTweets(count: number): FakeTweet[] {
  const items: FakeTweet[] = [];
  for (let i = 0; i < count; i++) {
    const source = i % 2 === 0 ? DISASTER : NOT_DISASTER;
    const base = source[i % source.length];
    if (!base) continue;
    const kind = i % 5;
    const target: 0 | 1 = i % 2 === 0 ? 1 : 0;
    if (kind === 3) {
      items.push({
        text: `x ${i}`,
        keyword: `synthetic-${i}`,
        target,
      });
    } else if (kind === 4) {
      items.push({
        text: `${"unseen traffic ".repeat(30)}${i}`.trim(),
        keyword: "not-a-training-keyword",
        target,
      });
    } else {
      items.push({
        text: `${base.text} #${i}`,
        keyword: base.keyword,
        target,
      });
    }
  }
  return items;
}

export function labelsCsv(items: FakeTweet[]): string {
  const lines = ["text,target"];
  for (const item of items) {
    lines.push(`${csvField(item.text)},${item.target}`);
  }
  return `${lines.join("\n")}\n`;
}

function csvField(value: string): string {
  if (/[",\n\r]/.test(value)) return `"${value.replaceAll('"', '""')}"`;
  return value;
}

function arg(name: string, fallback: string): string {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1])
    return process.argv[idx + 1] ?? fallback;
  return fallback;
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
  const count = Number(arg("count", "30"));
  const errors = Number(arg("errors", "2"));
  const labelsPath = resolve(
    arg("labels", `${import.meta.dir}/../data/fake_labels.csv`),
  );
  const items = fakeTweets(count);

  let ok = 0;
  let failed = 0;
  for (const item of items) {
    const body: { text: string; keyword?: string } = { text: item.text };
    if (item.keyword) body.keyword = item.keyword;
    const status = await post(url, JSON.stringify(body));
    if (status >= 200 && status < 300) ok += 1;
    else failed += 1;
  }
  for (let i = 0; i < errors; i++) {
    const status = await post(url, "", true);
    if (status >= 400) failed += 1;
    else ok += 1;
  }

  mkdirSync(dirname(labelsPath), { recursive: true });
  writeFileSync(labelsPath, labelsCsv(items));
  console.log(
    JSON.stringify(
      {
        url,
        sent: items.length + errors,
        ok,
        failed,
        labels: labelsPath,
        labeled: items.length,
      },
      null,
      2,
    ),
  );
}

if (import.meta.main) {
  await main();
}
