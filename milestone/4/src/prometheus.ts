import type { TrainBaseline } from "./baseline";
import {
  alignLiveKeywords,
  keywordKey,
  summarize,
  totalVariation,
  zScore,
} from "./drift";
import type { MetricRow } from "./log";

const BUCKETS_SECONDS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5];
const LIVE_WINDOW = 200;

function num(value: number): string {
  if (!Number.isFinite(value)) return "+Inf";
  return String(Number(value.toFixed(6)));
}

function liveRows(rows: MetricRow[]): MetricRow[] {
  const ok = rows.filter((row) => row.status === 200 && row.text_len > 0);
  return ok.slice(-LIVE_WINDOW);
}

export function renderPrometheus(
  rows: MetricRow[],
  baseline: TrainBaseline,
): string {
  const lines: string[] = [];
  const metric = (name: string, help: string, type: string) => {
    lines.push(`# HELP ${name} ${help}`);
    lines.push(`# TYPE ${name} ${type}`);
  };

  const counts = new Map<string, number>();
  for (const row of rows) {
    const key = `${row.route}\t${row.status}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  metric("api_requests_total", "Logged API requests.", "counter");
  for (const [key, count] of counts) {
    const [route, status] = key.split("\t");
    lines.push(
      `api_requests_total{route="${route}",code="${status}"} ${count}`,
    );
  }
  if (counts.size === 0)
    lines.push('api_requests_total{route="/predict",code="200"} 0');

  const latencies = rows
    .map((row) => row.latency_ms / 1000)
    .sort((a, b) => a - b);
  metric(
    "api_request_duration_seconds",
    "Logged request latency in seconds.",
    "histogram",
  );
  let covered = 0;
  for (const bucket of BUCKETS_SECONDS) {
    while (covered < latencies.length && (latencies[covered] ?? 0) <= bucket) {
      covered += 1;
    }
    lines.push(
      `api_request_duration_seconds_bucket{le="${bucket}"} ${covered}`,
    );
  }
  lines.push(
    `api_request_duration_seconds_bucket{le="+Inf"} ${latencies.length}`,
  );
  const latencySum = latencies.reduce((sum, value) => sum + value, 0);
  lines.push(`api_request_duration_seconds_sum ${num(latencySum)}`);
  lines.push(`api_request_duration_seconds_count ${latencies.length}`);

  const confidences = rows
    .map((row) => row.confidence)
    .filter((value): value is number => value !== null);
  const confidenceAvg =
    confidences.length === 0
      ? 0
      : confidences.reduce((sum, value) => sum + value, 0) / confidences.length;
  metric(
    "prediction_confidence_avg",
    "Mean confidence of logged predictions.",
    "gauge",
  );
  lines.push(`prediction_confidence_avg ${num(confidenceAvg)}`);

  const live = liveRows(rows);
  const liveN = live.length;
  const lengthZ =
    liveN === 0
      ? 0
      : zScore(
          summarize(live.map((row) => row.text_len)).mean,
          baseline.text_len,
        );
  const wordZ =
    liveN === 0
      ? 0
      : zScore(
          summarize(live.map((row) => row.word_count)).mean,
          baseline.word_count,
        );
  const liveKeywordCounts = new Map<string, number>();
  let withKeyword = 0;
  let disasters = 0;
  let labeled = 0;
  for (const row of live) {
    const key = keywordKey(row.keyword);
    liveKeywordCounts.set(key, (liveKeywordCounts.get(key) ?? 0) + 1);
    if (key !== "(none)") withKeyword += 1;
    if (row.target !== null) {
      labeled += 1;
      if (row.target === 1) disasters += 1;
    }
  }
  const liveShares = alignLiveKeywords(baseline.keywords, liveKeywordCounts);
  const tvd = liveN === 0 ? 0 : totalVariation(baseline.keywords, liveShares);
  const keywordRate = liveN === 0 ? 0 : withKeyword / liveN;
  const predictedRate = labeled === 0 ? 0 : disasters / labeled;

  const gauge = (name: string, help: string, value: number) => {
    metric(name, help, "gauge");
    lines.push(`${name} ${num(value)}`);
  };
  gauge(
    "drift_live_samples",
    "Successful requests in the drift window.",
    liveN,
  );
  gauge(
    "drift_text_length_z",
    "Live mean text length versus the training distribution, in training standard deviations.",
    lengthZ,
  );
  gauge(
    "drift_word_count_z",
    "Live mean word count versus the training distribution, in training standard deviations.",
    wordZ,
  );
  gauge(
    "drift_keyword_tvd",
    "Total variation distance between live and training keyword shares.",
    tvd,
  );
  gauge(
    "drift_keyword_present_delta",
    "Live keyword-present rate minus the training rate.",
    keywordRate - baseline.keyword_present_rate,
  );
  gauge(
    "drift_predicted_disaster_rate",
    "Share of live predictions labeled disaster.",
    predictedRate,
  );
  gauge(
    "drift_train_disaster_rate",
    "Share of training rows with target 1.",
    baseline.disaster_label_rate,
  );

  return `${lines.join("\n")}\n`;
}
