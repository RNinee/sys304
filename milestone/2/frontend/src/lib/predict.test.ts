import { describe, expect, test } from "bun:test";
import {
  apiUrl,
  extractUserText,
  formatAssistantReply,
  type PredictResponse,
} from "./predict";

const disaster: PredictResponse = {
  target: 1,
  label: "disaster",
  confidence: 0.94,
  probabilities: { disaster: 0.94, not_disaster: 0.06 },
  model: "qwen2.5-1.5b-lora",
  input: "Forest fire near La Ronge Sask. Canada",
};

describe("extractUserText", () => {
  test("reads assistant-ui text parts", () => {
    expect(
      extractUserText([{ type: "text", text: "Just happened a terrible car crash" }]),
    ).toBe("Just happened a terrible car crash");
  });

  test("trims a plain string", () => {
    expect(extractUserText("  hello  ")).toBe("hello");
  });
});

describe("formatAssistantReply", () => {
  test("labels a disaster prediction", () => {
    const text = formatAssistantReply(disaster);
    expect(text).toContain("real disaster");
    expect(text).toContain("94.0%");
    expect(text).toContain("qwen2.5-1.5b-lora");
  });
});

describe("apiUrl", () => {
  test("defaults to local Elysia", () => {
    expect(apiUrl()).toBe("http://localhost:8000");
  });
});
