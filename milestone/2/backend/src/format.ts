const MAX_TWEET_CHARS = 560;

export function buildInput(text: string, keyword?: string): string {
  const body = text.trim();
  const kw = keyword?.trim() ?? "";
  if (!kw) return body;
  if (body.toLowerCase().startsWith("keyword:")) return body;
  return `keyword: ${kw}\n${body}`;
}

export function validatePredictBody(body: unknown):
  | {
      ok: true;
      text: string;
      keyword?: string;
    }
  | {
      ok: false;
      error: string;
    } {
  if (body === null || typeof body !== "object") {
    return { ok: false, error: "JSON object required" };
  }
  const record = body as Record<string, unknown>;
  if (typeof record.text !== "string") {
    return { ok: false, error: "text is required" };
  }
  const text = record.text.trim();
  if (!text) {
    return { ok: false, error: "text is required" };
  }
  if (text.length > MAX_TWEET_CHARS) {
    return {
      ok: false,
      error: `text must be at most ${MAX_TWEET_CHARS} characters`,
    };
  }
  if (record.keyword !== undefined && typeof record.keyword !== "string") {
    return { ok: false, error: "keyword must be a string" };
  }
  const keyword =
    typeof record.keyword === "string" ? record.keyword.trim() : "";
  return keyword ? { ok: true, text, keyword } : { ok: true, text };
}

export function formatAssistantReply(result: {
  label: string;
  confidence: number;
  probabilities: { disaster: number; not_disaster: number };
  model: string;
}): string {
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
  const headline =
    result.label === "disaster"
      ? "This looks like a **real disaster** tweet."
      : "This looks like **not a disaster** (joke, metaphor, or unrelated).";
  return [
    headline,
    "",
    `| Class | Probability |`,
    `| --- | ---: |`,
    `| disaster | ${pct(result.probabilities.disaster)} |`,
    `| not disaster | ${pct(result.probabilities.not_disaster)} |`,
    "",
    `Confidence: **${pct(result.confidence)}** · model: \`${result.model}\``,
  ].join("\n");
}

export { MAX_TWEET_CHARS };
