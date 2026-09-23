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

export type LoggedRequest = {
  route: "/predict" | "/chat";
  status: number;
  latencyMs: number;
  text: string;
  keyword?: string;
  result?: PredictResponse;
  error?: string;
};

export type AppOptions = {
  infer?: InferFn;
  cache?: PredictCache;
  onLogged?: (event: LoggedRequest) => void;
};

export function createApp(inferOrOptions?: InferFn | AppOptions) {
  const options: AppOptions =
    typeof inferOrOptions === "function"
      ? { infer: inferOrOptions }
      : (inferOrOptions ?? {});
  const infer = options.infer;
  const cache = options.cache ?? noopCache;

  const emit = (
    route: LoggedRequest["route"],
    started: number,
    status: number,
    fields: {
      text?: string;
      keyword?: string;
      result?: PredictResponse;
      error?: string;
    },
  ) => {
    if (!options.onLogged) return;
    try {
      const event: LoggedRequest = {
        route,
        status,
        latencyMs: performance.now() - started,
        text: fields.text ?? "",
      };
      if (fields.keyword) event.keyword = fields.keyword;
      if (fields.result) event.result = fields.result;
      if (fields.error) event.error = fields.error;
      options.onLogged(event);
    } catch (error) {
      console.error("[log] failed:", error);
    }
  };

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
      const started = performance.now();
      let raw: unknown;
      try {
        raw = await request.json();
      } catch {
        set.status = 400;
        emit("/predict", started, 400, { error: "invalid JSON" });
        return { error: "invalid JSON" };
      }
      const parsed = validatePredictBody(raw);
      if (!parsed.ok) {
        set.status = 400;
        const text =
          raw && typeof raw === "object" && "text" in raw
            ? String((raw as { text?: unknown }).text ?? "")
            : "";
        emit("/predict", started, 400, { text, error: parsed.error });
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
          emit("/predict", started, 200, {
            text: body.text,
            keyword: body.keyword,
            result,
          });
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
        const response = { ...result, cached: false };
        emit("/predict", started, 200, {
          text: body.text,
          keyword: body.keyword,
          result: response,
        });
        return response;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error("[predict] failed:", message);
        set.status = 503;
        emit("/predict", started, 503, {
          text: body.text,
          keyword: body.keyword,
          error: message,
        });
        return { error: message };
      }
    })
    .post("/chat", async ({ request, set }) => {
      const started = performance.now();
      let raw: unknown;
      try {
        raw = await request.json();
      } catch {
        set.status = 400;
        emit("/chat", started, 400, { error: "invalid JSON" });
        return { error: "invalid JSON" };
      }
      const messages = (
        raw as { messages?: Array<{ role?: string; content?: unknown }> }
      ).messages;
      if (!Array.isArray(messages) || messages.length === 0) {
        set.status = 400;
        emit("/chat", started, 400, { error: "messages array required" });
        return { error: "messages array required" };
      }
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      const text = extractText(lastUser?.content);
      const parsed = validatePredictBody({ text });
      if (!parsed.ok) {
        set.status = 400;
        emit("/chat", started, 400, { text, error: parsed.error });
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
        emit("/chat", started, 200, {
          text: body.text,
          keyword: body.keyword,
          result,
        });
        return {
          text: formatAssistantReply(result),
          prediction: result,
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error("[chat] failed:", message);
        set.status = 503;
        emit("/chat", started, 503, {
          text: body.text,
          keyword: body.keyword,
          error: message,
        });
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
