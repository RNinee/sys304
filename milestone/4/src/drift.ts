export type NumericSummary = {
  n: number;
  mean: number;
  std: number;
  p50: number;
  p95: number;
};

export type DriftStatus = "ok" | "watch" | "drift" | "warming";

export const MIN_LIVE_SAMPLES = 20;

export function summarize(values: number[]): NumericSummary {
  if (values.length === 0) {
    return { n: 0, mean: 0, std: 0, p50: 0, p95: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((sum, value) => sum + value, 0) / n;
  const variance =
    n < 2
      ? 0
      : sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1);
  return {
    n,
    mean,
    std: Math.sqrt(variance),
    p50: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
  };
}

export function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(q * sorted.length) - 1),
  );
  return sorted[index] ?? 0;
}

export function totalVariation(
  train: Record<string, number>,
  live: Record<string, number>,
): number {
  const keys = new Set([...Object.keys(train), ...Object.keys(live)]);
  let l1 = 0;
  for (const key of keys) {
    l1 += Math.abs((train[key] ?? 0) - (live[key] ?? 0));
  }
  return 0.5 * l1;
}

export function countsToShare(
  counts: Map<string, number>,
): Record<string, number> {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  const shares: Record<string, number> = {};
  if (total === 0) return shares;
  for (const [key, count] of counts) shares[key] = count / total;
  return shares;
}

export function zScore(liveMean: number, train: NumericSummary): number {
  if (train.std === 0)
    return liveMean === train.mean ? 0 : Number.POSITIVE_INFINITY;
  return (liveMean - train.mean) / train.std;
}

export function numericStatus(z: number, liveN: number): DriftStatus {
  if (liveN < MIN_LIVE_SAMPLES) return "warming";
  const magnitude = Math.abs(z);
  if (magnitude >= 2) return "drift";
  if (magnitude >= 1) return "watch";
  return "ok";
}

export function deltaStatus(
  delta: number,
  liveN: number,
  watchAt: number,
  driftAt: number,
): DriftStatus {
  if (liveN < MIN_LIVE_SAMPLES) return "warming";
  const magnitude = Math.abs(delta);
  if (magnitude >= driftAt) return "drift";
  if (magnitude >= watchAt) return "watch";
  return "ok";
}

const NONE = "(none)";
const OTHER = "(other)";

export function keywordKey(keyword: string | null | undefined): string {
  const trimmed = keyword?.trim().toLowerCase() ?? "";
  return trimmed || NONE;
}

export function alignLiveKeywords(
  trainShares: Record<string, number>,
  liveCounts: Map<string, number>,
): Record<string, number> {
  const aligned = new Map<string, number>();
  for (const key of Object.keys(trainShares)) aligned.set(key, 0);
  aligned.set(OTHER, 0);
  for (const [key, count] of liveCounts) {
    if (key in trainShares) aligned.set(key, (aligned.get(key) ?? 0) + count);
    else aligned.set(OTHER, (aligned.get(OTHER) ?? 0) + count);
  }
  return countsToShare(aligned);
}
