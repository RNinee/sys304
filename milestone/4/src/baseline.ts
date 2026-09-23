import { readFileSync } from "node:fs";
import {
  countsToShare,
  keywordKey,
  type NumericSummary,
  summarize,
} from "./drift";

export type TrainBaseline = {
  source: string;
  n: number;
  text_len: NumericSummary;
  word_count: NumericSummary;
  keyword_present_rate: number;
  disaster_label_rate: number;
  keywords: Record<string, number>;
};

export function defaultTrainCsv(): string {
  return process.env.TRAIN_CSV ?? `${import.meta.dir}/../../1/data/train.csv`;
}

export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (inQuotes) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
      continue;
    }
    if (char === '"') inQuotes = true;
    else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") field += char;
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((cells) => cells.some((cell) => cell.length > 0));
}

export function baselineFromCsv(
  csv: string,
  source = "train.csv",
): TrainBaseline {
  const rows = parseCsv(csv);
  const header = rows[0] ?? [];
  const textAt = header.indexOf("text");
  const keywordAt = header.indexOf("keyword");
  const targetAt = header.indexOf("target");
  if (textAt < 0 || targetAt < 0) {
    throw new Error("train.csv must include text and target columns");
  }
  const lengths: number[] = [];
  const words: number[] = [];
  const keywords = new Map<string, number>();
  let withKeyword = 0;
  let disasters = 0;
  for (const cells of rows.slice(1)) {
    const text = cells[textAt] ?? "";
    const keyword = keywordAt >= 0 ? (cells[keywordAt] ?? "") : "";
    const target = Number(cells[targetAt] ?? "0");
    lengths.push(text.length);
    const trimmed = text.trim();
    words.push(trimmed ? trimmed.split(/\s+/).length : 0);
    const key = keywordKey(keyword);
    keywords.set(key, (keywords.get(key) ?? 0) + 1);
    if (key !== "(none)") withKeyword += 1;
    if (target === 1) disasters += 1;
  }
  const n = lengths.length;
  return {
    source,
    n,
    text_len: summarize(lengths),
    word_count: summarize(words),
    keyword_present_rate: n === 0 ? 0 : withKeyword / n,
    disaster_label_rate: n === 0 ? 0 : disasters / n,
    keywords: countsToShare(keywords),
  };
}

export function loadTrainBaseline(path = defaultTrainCsv()): TrainBaseline {
  return baselineFromCsv(readFileSync(path, "utf8"), path);
}
