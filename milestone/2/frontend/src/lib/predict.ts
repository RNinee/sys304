export type PredictRequest = {
  text: string;
  keyword?: string;
};

export type PredictResponse = {
  target: 0 | 1;
  label: "disaster" | "not disaster";
  confidence: number;
  probabilities: {
    not_disaster: number;
    disaster: number;
  };
  model: string;
  input: string;
};

export function apiUrl(): string {
  return (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
    /\/$/,
    "",
  );
}

export function extractUserText(content: unknown): string {
  if (typeof content === "string") return content.trim();
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

export function formatAssistantReply(result: PredictResponse): string {
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

export async function predictTweet(
  body: PredictRequest,
  options: { signal?: AbortSignal } = {},
): Promise<PredictResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiUrl()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options.signal,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new Error(
      `Cannot reach the classifier API at ${apiUrl()}. Start it with ./deploy.sh local`,
    );
  }
  const payload = (await response.json()) as PredictResponse & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error ?? `API returned ${response.status}`);
  }
  return payload;
}
