import { createHash } from "node:crypto";
import { RedisClient } from "bun";
import type { PredictRequest, PredictResponse } from "./types";

const PREFIX = "sys304:predict:v1:";

export function cacheKey(body: PredictRequest): string {
  const raw = `${body.keyword ?? ""}\n${body.text}`;
  return PREFIX + createHash("sha256").update(raw).digest("hex");
}

export type PredictCache = {
  status: "up" | "down" | "off";
  get: (body: PredictRequest) => Promise<PredictResponse | null>;
  set: (body: PredictRequest, value: PredictResponse) => Promise<void>;
};

export const noopCache: PredictCache = {
  status: "off",
  async get() {
    return null;
  },
  async set() {},
};

function stripRuntimeFields(value: PredictResponse): PredictResponse {
  return {
    target: value.target,
    label: value.label,
    confidence: value.confidence,
    probabilities: value.probabilities,
    model: value.model,
    input: value.input,
  };
}

export function memoryCache(): PredictCache {
  const store = new Map<string, string>();
  return {
    status: "up",
    async get(body) {
      const raw = store.get(cacheKey(body));
      if (!raw) return null;
      return JSON.parse(raw) as PredictResponse;
    },
    async set(body, value) {
      store.set(cacheKey(body), JSON.stringify(stripRuntimeFields(value)));
    },
  };
}

export function redisCache(
  url: string,
  ttlSeconds = Number(process.env.CACHE_TTL_SECONDS ?? 3600),
): PredictCache {
  const client = new RedisClient(url);
  const cache: PredictCache = {
    status: "up",
    async get(body) {
      try {
        const raw = await client.get(cacheKey(body));
        if (!raw) return null;
        return JSON.parse(raw) as PredictResponse;
      } catch (error) {
        cache.status = "down";
        console.error("[cache] get failed:", error);
        return null;
      }
    },
    async set(body, value) {
      try {
        const payload = JSON.stringify(stripRuntimeFields(value));
        const key = cacheKey(body);
        await client.set(key, payload);
        await client.expire(key, ttlSeconds);
        cache.status = "up";
      } catch (error) {
        cache.status = "down";
        console.error("[cache] set failed:", error);
      }
    },
  };
  return cache;
}

export function createCache(): PredictCache {
  const url = process.env.REDIS_URL;
  if (!url) return noopCache;
  try {
    return redisCache(url);
  } catch (error) {
    console.error("[cache] redis disabled:", error);
    return { ...noopCache, status: "down" };
  }
}
