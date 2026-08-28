import { cors } from "@elysiajs/cors";
import { Elysia } from "elysia";
import { formatAssistantReply, validatePredictBody } from "./format";
import { inferHealth, inferPredict } from "./infer";
import type { PredictRequest, PredictResponse } from "./types";

export type InferFn = (
  body: PredictRequest,
  options?: { signal?: AbortSignal },
) => Promise<PredictResponse>;

export function createApp(infer: InferFn = inferPredict) {
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
        infer: worker ? "up" : "down",
      };
    })
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
        const result = await infer(body, { signal: request.signal });
        console.log(
          `[predict] ${JSON.stringify(body.text.slice(0, 80))} -> ${result.label} (${result.confidence.toFixed(3)}) [${result.model}]`,
        );
        return result;
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
      try {
        const result = await infer(
          parsed.keyword
            ? { text: parsed.text, keyword: parsed.keyword }
            : { text: parsed.text },
          { signal: request.signal },
        );
        console.log(
          `[chat] ${JSON.stringify(parsed.text.slice(0, 80))} -> ${result.label} (${result.confidence.toFixed(3)})`,
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

export const app = createApp();
