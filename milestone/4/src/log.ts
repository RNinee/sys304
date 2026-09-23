import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { LoggedRequest } from "../../3/backend/src/app";

export type RequestRow = {
  id: number;
  ts: string;
  route: string;
  status: number;
  latency_ms: number;
  text: string;
  keyword: string;
  text_len: number;
  word_count: number;
  label: string | null;
  target: number | null;
  confidence: number | null;
  model: string | null;
  cached: number;
  error: string | null;
  truth_target: number | null;
};

export type MetricRow = {
  route: string;
  status: number;
  latency_ms: number;
  text_len: number;
  word_count: number;
  keyword: string;
  target: number | null;
  confidence: number | null;
  ts: string;
};

export function wordCount(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

export class RequestLog {
  private db: Database;
  private insertStmt: {
    run: (...params: (string | number | null)[]) => unknown;
  };

  constructor(path = defaultLogPath()) {
    if (path !== ":memory:") {
      mkdirSync(dirname(path), { recursive: true });
    }
    this.db = new Database(path, { create: true });
    if (path !== ":memory:") this.db.exec("PRAGMA journal_mode = WAL;");
    this.db.exec(`CREATE TABLE IF NOT EXISTS requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      route TEXT NOT NULL,
      status INTEGER NOT NULL,
      latency_ms REAL NOT NULL,
      text TEXT NOT NULL,
      keyword TEXT NOT NULL DEFAULT '',
      text_len INTEGER NOT NULL,
      word_count INTEGER NOT NULL,
      label TEXT,
      target INTEGER,
      confidence REAL,
      model TEXT,
      cached INTEGER NOT NULL DEFAULT 0,
      error TEXT,
      truth_target INTEGER
    )`);
    const columns = this.db.prepare("PRAGMA table_info(requests)").all() as {
      name: string;
    }[];
    if (!columns.some((column) => column.name === "truth_target")) {
      this.db.exec("ALTER TABLE requests ADD COLUMN truth_target INTEGER");
    }
    this.db.exec("PRAGMA busy_timeout = 5000;");
    this.db.exec("CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts)");
    this.insertStmt = this.db.prepare(`INSERT INTO requests
      (ts, route, status, latency_ms, text, keyword, text_len, word_count, label, target, confidence, model, cached, error)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
  }

  insert(event: LoggedRequest): void {
    const text = event.text ?? "";
    this.insertStmt.run(
      new Date().toISOString(),
      event.route,
      event.status,
      event.latencyMs,
      text,
      event.keyword ?? "",
      text.length,
      wordCount(text),
      event.result?.label ?? null,
      event.result?.target ?? null,
      event.result?.confidence ?? null,
      event.result?.model ?? null,
      event.result?.cached ? 1 : 0,
      event.error ?? null,
    );
  }

  metricsRows(): MetricRow[] {
    return this.db
      .prepare(
        `SELECT route, status, latency_ms, text_len, word_count, keyword, target, confidence, ts
         FROM requests ORDER BY id ASC`,
      )
      .all() as MetricRow[];
  }

  meanConfidence(): number | null {
    const row = this.db
      .prepare(
        "SELECT avg(confidence) AS mean FROM requests WHERE confidence IS NOT NULL",
      )
      .get() as { mean: number | null };
    return row.mean;
  }

  recent(limit: number): RequestRow[] {
    return this.db
      .prepare("SELECT * FROM requests ORDER BY id DESC LIMIT ?")
      .all(limit) as RequestRow[];
  }

  since(iso: string): RequestRow[] {
    return this.db
      .prepare("SELECT * FROM requests WHERE ts >= ? ORDER BY id ASC")
      .all(iso) as RequestRow[];
  }

  all(): RequestRow[] {
    return this.db
      .prepare("SELECT * FROM requests ORDER BY id ASC")
      .all() as RequestRow[];
  }

  close(): void {
    this.db.close();
  }
}

export function defaultLogPath(): string {
  return (
    process.env.REQUEST_LOG_PATH ?? `${import.meta.dir}/../data/requests.sqlite`
  );
}
