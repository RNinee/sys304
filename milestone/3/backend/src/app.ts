import { cors } from "@elysiajs/cors";
import { Elysia } from "elysia";
import { noopCache, type PredictCache } from "./cache";
import { formatAssistantReply, validatePredictBody } from "./format";
import { inferHealth } from "./infer";
import { snapshot, stats } from "./stats";
import type { PredictRequest, PredictResponse } from "./types";

export type InferFn = (
  body: PredictRequest,
  options?: { signal?: AbortSignal },
) => Promise<PredictResponse>;

export type AppOptions = {
  infer?: InferFn;
  cache?: PredictCache;
};

export function createApp(inferOrOptions?: InferFn | AppOptions) {
  const options: AppOptions =
    typeof inferOrOptions === "function"
      ? { infer: inferOrOptions }
      : (inferOrOptions ?? {});
  const infer = options.infer;
  const cache = options.cache ?? noopCache;

  return new Elysia()
    .use(
      cors({
        origin: true,
        methods: ["GET", "POST", "OPTIONS"],
      }),
    )
    .onError(({ code, error, set }) => {
      const message = error instanceof Error ? error.message : String(error);
      if (code === "NOT_FOUND") {
        set.status = 404;
        return { error: "not found" };
      }
      set.status = 500;
      return { error: message };
    })
    .get("/health", async () => {
      const worker = await inferHealth().catch(() => null);
      return {
        status: "ok" as const,
        role: "api" as const,
        model: worker?.model ?? "unknown",
        infer: worker ? ("up" as const) : ("down" as const),
        cache: cache.status,
        optimizations: worker?.optimizations ?? [
          "dynamic-batching",
          ...(cache.status === "off" ? [] : ["redis-cache"]),
        ],
      };
    })
    .get("/metrics", () => ({
      ...snapshot(),
      cache: cache.status,
    }))
    .post("/predict", async ({ request, set }) => {
      let raw: unknown;
      try {
        raw = await request.json();
      } catch {
        set.status = 400;
        return { error: "invalid JSON" };
      }
      const parsed = validatePredictBody(raw);
      if (!parsed.ok) {
        set.status = 400;
        return { error: parsed.error };
      }
      const body: PredictRequest = parsed.keyword
        ? { text: parsed.text, keyword: parsed.keyword }
        : { text: parsed.text };
      try {
        const hit = await cache.get(body);
        if (hit) {
          stats.cacheHits += 1;
          const result = { ...hit, cached: true };
          console.log(
            `[predict] cache hit ${JSON.stringify(body.text.slice(0, 80))} -> ${result.label}`,
          );
          return result;
        }
        stats.cacheMisses += 1;
        if (!infer) {
          throw new Error("infer function not configured");
        }
        const result = await infer(body, { signal: request.signal });
        await cache.set(body, result);
        console.log(
          `[predict] ${JSON.stringify(body.text.slice(0, 80))} -> ${result.label} (${result.confidence.toFixed(3)}) [${result.model}] batch=${result.batch_size ?? 1}`,
        );
        return { ...result, cached: false };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error("[predict] failed:", message);
        set.status = 503;
        return { error: message };
      }
    })
    .post("/chat", async ({ request, set }) => {
      let raw: unknown;
      try {
        raw = await request.json();
      } catch {
        set.status = 400;
        return { error: "invalid JSON" };
      }
      const messages = (
        raw as { messages?: Array<{ role?: string; content?: unknown }> }
      ).messages;
      if (!Array.isArray(messages) || messages.length === 0) {
        set.status = 400;
        return { error: "messages array required" };
      }
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      const text = extractText(lastUser?.content);
      const parsed = validatePredictBody({ text });
      if (!parsed.ok) {
        set.status = 400;
        return { error: parsed.error };
      }
      const body: PredictRequest = parsed.keyword
        ? { text: parsed.text, keyword: parsed.keyword }
        : { text: parsed.text };
      try {
        const hit = await cache.get(body);
        let result: PredictResponse;
        if (hit) {
          stats.cacheHits += 1;
          result = { ...hit, cached: true };
        } else {
          stats.cacheMisses += 1;
          if (!infer) {
            throw new Error("infer function not configured");
          }
          result = {
            ...(await infer(body, { signal: request.signal })),
            cached: false,
          };
          await cache.set(body, result);
        }
        console.log(
          `[chat] ${JSON.stringify(parsed.text.slice(0, 80))} -> ${result.label} (${result.confidence.toFixed(3)})${result.cached ? " cached" : ""}`,
        );
        return {
          text: formatAssistantReply(result),
          prediction: result,
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error("[chat] failed:", message);
        set.status = 503;
        return { error: message };
      }
    });
}

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object" && "text" in part) {
        return String((part as { text: unknown }).text ?? "");
      }
      return "";
    })
    .join("\n")
    .trim();
}
