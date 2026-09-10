export type ServiceStats = {
  cacheHits: number;
  cacheMisses: number;
  batches: number;
  batchedItems: number;
};

export const stats: ServiceStats = {
  cacheHits: 0,
  cacheMisses: 0,
  batches: 0,
  batchedItems: 0,
};

export function resetStats(): void {
  stats.cacheHits = 0;
  stats.cacheMisses = 0;
  stats.batches = 0;
  stats.batchedItems = 0;
}

export function snapshot(): ServiceStats & { avgBatchSize: number } {
  return {
    ...stats,
    avgBatchSize: stats.batches === 0 ? 0 : stats.batchedItems / stats.batches,
  };
}
