import { describe, expect, test } from "bun:test";
import {
  buildInput,
  formatAssistantReply,
  MAX_TWEET_CHARS,
  validatePredictBody,
} from "../src/format";

describe("buildInput", () => {
  test("returns trimmed text when keyword is empty", () => {
    expect(buildInput("  Forest fire near La Ronge  ")).toBe(
      "Forest fire near La Ronge",
    );
  });

  test("prefixes keyword in the Milestone 1 training format", () => {
    expect(buildInput("Our Deeds are the Reason", "earthquake")).toBe(
      "keyword: earthquake\nOur Deeds are the Reason",
    );
  });

  test("does not double-prefix if the tweet already includes keyword:", () => {
    expect(buildInput("keyword: flood\nwater in the streets", "flood")).toBe(
      "keyword: flood\nwater in the streets",
    );
  });
});

describe("validatePredictBody", () => {
  test("accepts a tweet", () => {
    expect(
      validatePredictBody({ text: "Just happened a terrible car crash" }),
    ).toEqual({
      ok: true,
      text: "Just happened a terrible car crash",
    });
  });

  test("rejects missing text", () => {
    expect(validatePredictBody({})).toEqual({
      ok: false,
      error: "text is required",
    });
  });

  test("rejects blank text", () => {
    expect(validatePredictBody({ text: "   " })).toEqual({
      ok: false,
      error: "text is required",
    });
  });

  test("rejects oversized tweets", () => {
    const text = "a".repeat(MAX_TWEET_CHARS + 1);
    expect(validatePredictBody({ text }).ok).toBe(false);
  });

  test("rejects a non-string text field", () => {
    expect(validatePredictBody({ text: 1 })).toEqual({
      ok: false,
      error: "text is required",
    });
  });

  test("rejects a non-string keyword", () => {
    expect(validatePredictBody({ text: "hi", keyword: 1 })).toEqual({
      ok: false,
      error: "keyword must be a string",
    });
  });
});

describe("formatAssistantReply", () => {
  test("mentions disaster when the model predicts class 1", () => {
    const text = formatAssistantReply({
      label: "disaster",
      confidence: 0.91,
      probabilities: { disaster: 0.91, not_disaster: 0.09 },
      model: "qwen2.5-1.5b-lora",
    });
    expect(text).toContain("real disaster");
    expect(text).toContain("91.0%");
    expect(text).toContain("qwen2.5-1.5b-lora");
  });

  test("mentions not a disaster for class 0", () => {
    const text = formatAssistantReply({
      label: "not disaster",
      confidence: 0.8,
      probabilities: { disaster: 0.2, not_disaster: 0.8 },
      model: "qwen2.5-1.5b-lora",
    });
    expect(text).toContain("not a disaster");
  });
});
