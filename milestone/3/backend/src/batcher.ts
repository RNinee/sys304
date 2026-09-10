import { stats } from "./stats";
import type { PredictRequest, PredictResponse } from "./types";

type Pending = {
  body: PredictRequest;
  resolve: (value: PredictResponse) => void;
  reject: (error: unknown) => void;
};

export type InferBatchFn = (
  items: PredictRequest[],
) => Promise<PredictResponse[]>;

export class DynamicBatcher {
  private queue: Pending[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly inferBatch: InferBatchFn,
    readonly maxSize = Number(process.env.BATCH_MAX_SIZE ?? 8),
    readonly maxWaitMs = Number(process.env.BATCH_MAX_WAIT_MS ?? 15),
  ) {}

  enqueue(
    body: PredictRequest,
    signal?: AbortSignal,
  ): Promise<PredictResponse> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(new Error("aborted"));
        return;
      }
      const pending: Pending = { body, resolve, reject };
      const onAbort = () => {
        const idx = this.queue.indexOf(pending);
        if (idx >= 0) {
          this.queue.splice(idx, 1);
          reject(new Error("aborted"));
        }
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      this.queue.push(pending);
      if (this.queue.length >= this.maxSize) {
        void this.flush();
      } else if (!this.timer) {
        this.timer = setTimeout(() => void this.flush(), this.maxWaitMs);
      }
    });
  }

  async flush(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.queue.length === 0) return;
    const batch = this.queue.splice(0, this.maxSize);
    stats.batches += 1;
    stats.batchedItems += batch.length;
    try {
      const results = await this.inferBatch(batch.map((item) => item.body));
      if (results.length !== batch.length) {
        throw new Error(
          `batch size mismatch: sent ${batch.length}, got ${results.length}`,
        );
      }
      for (let i = 0; i < batch.length; i++) {
        batch[i].resolve({ ...results[i], batch_size: batch.length });
      }
    } catch (error) {
      for (const item of batch) item.reject(error);
    }
    if (this.queue.length > 0) {
      await this.flush();
    }
  }
}
