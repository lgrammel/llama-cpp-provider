import { describe, it, expect } from "vitest";
import {
  convertFinishReason,
  convertUsage,
  resolveReasoningConfig,
  splitReasoningContent,
} from "../../src/llama-cpp-language-model.js";
import { gemma4Reasoning } from "../../src/llama-cpp-provider-config.js";

describe("convertFinishReason", () => {
  describe("known finish reasons", () => {
    it('converts "stop" to unified "stop"', () => {
      const result = convertFinishReason("stop");

      expect(result.unified).toBe("stop");
      expect(result.raw).toBe("stop");
    });

    it('converts "length" to unified "length"', () => {
      const result = convertFinishReason("length");

      expect(result.unified).toBe("length");
      expect(result.raw).toBe("length");
    });
  });

  describe("unknown finish reasons", () => {
    it('converts "error" to unified "other"', () => {
      const result = convertFinishReason("error");

      expect(result.unified).toBe("other");
      expect(result.raw).toBe("error");
    });

    it('converts unknown reason to unified "other"', () => {
      const result = convertFinishReason("some_unknown_reason");

      expect(result.unified).toBe("other");
      expect(result.raw).toBe("some_unknown_reason");
    });

    it('converts empty string to unified "other"', () => {
      const result = convertFinishReason("");

      expect(result.unified).toBe("other");
      expect(result.raw).toBe("");
    });
  });

  describe("return value structure", () => {
    it("returns object with unified and raw properties", () => {
      const result = convertFinishReason("stop");

      expect(result).toHaveProperty("unified");
      expect(result).toHaveProperty("raw");
      expect(Object.keys(result)).toHaveLength(2);
    });
  });
});

describe("convertUsage", () => {
  describe("token counting", () => {
    it("correctly maps prompt tokens to inputTokens.total", () => {
      const result = convertUsage(100, 50);

      expect(result.inputTokens.total).toBe(100);
    });

    it("correctly maps completion tokens to outputTokens.total", () => {
      const result = convertUsage(100, 50);

      expect(result.outputTokens.total).toBe(50);
    });

    it("sets outputTokens.text to completion tokens", () => {
      const result = convertUsage(100, 50);

      expect(result.outputTokens.text).toBe(50);
    });
  });

  describe("undefined cache fields", () => {
    it("sets inputTokens cache fields to undefined", () => {
      const result = convertUsage(100, 50);

      expect(result.inputTokens.noCache).toBeUndefined();
      expect(result.inputTokens.cacheRead).toBeUndefined();
      expect(result.inputTokens.cacheWrite).toBeUndefined();
    });

    it("sets outputTokens.reasoning to undefined", () => {
      const result = convertUsage(100, 50);

      expect(result.outputTokens.reasoning).toBeUndefined();
    });
  });

  describe("edge cases", () => {
    it("handles zero tokens", () => {
      const result = convertUsage(0, 0);

      expect(result.inputTokens.total).toBe(0);
      expect(result.outputTokens.total).toBe(0);
      expect(result.outputTokens.text).toBe(0);
    });

    it("handles large token counts", () => {
      const result = convertUsage(100000, 50000);

      expect(result.inputTokens.total).toBe(100000);
      expect(result.outputTokens.total).toBe(50000);
    });
  });

  describe("return value structure", () => {
    it("returns object with inputTokens and outputTokens", () => {
      const result = convertUsage(10, 5);

      expect(result).toHaveProperty("inputTokens");
      expect(result).toHaveProperty("outputTokens");
    });

    it("inputTokens has correct structure", () => {
      const result = convertUsage(10, 5);

      expect(result.inputTokens).toHaveProperty("total");
      expect(result.inputTokens).toHaveProperty("noCache");
      expect(result.inputTokens).toHaveProperty("cacheRead");
      expect(result.inputTokens).toHaveProperty("cacheWrite");
    });

    it("outputTokens has correct structure", () => {
      const result = convertUsage(10, 5);

      expect(result.outputTokens).toHaveProperty("total");
      expect(result.outputTokens).toHaveProperty("text");
      expect(result.outputTokens).toHaveProperty("reasoning");
    });
  });
});

describe("resolveReasoningConfig", () => {
  it("defaults to think tag markers", () => {
    const result = resolveReasoningConfig({});

    expect(result).toEqual({
      opening: "<think>",
      closing: "</think>",
      promptPrefix: undefined,
    });
  });

  it("supports Gemma 4 thinking markers", () => {
    const result = resolveReasoningConfig(gemma4Reasoning);

    expect(result).toEqual({
      opening: "<|channel>thought\n",
      closing: "<channel|>",
      promptPrefix: "<|think|>\n",
    });
  });

  it("supports custom markers and prompt prefix", () => {
    const result = resolveReasoningConfig({
      openingMarker: "[reasoning]",
      closingMarker: "[/reasoning]",
      promptPrefix: "think first\n",
    });

    expect(result).toEqual({
      opening: "[reasoning]",
      closing: "[/reasoning]",
      promptPrefix: "think first\n",
    });
  });
});

describe("splitReasoningContent", () => {
  const gemma4Markers = {
    opening: "<|channel>thought\n",
    closing: "<channel|>",
  };

  it("extracts Gemma 4 reasoning and final text", () => {
    const result = splitReasoningContent(
      "<|channel>thought\nNeed to calculate.<channel|>The answer is 42.",
      gemma4Markers
    );

    expect(result).toEqual([
      { type: "reasoning", text: "Need to calculate." },
      { type: "text", text: "The answer is 42." },
    ]);
  });

  it("handles empty Gemma 4 thinking blocks", () => {
    const result = splitReasoningContent(
      "<|channel>thought\n<channel|>Thinking disabled.",
      gemma4Markers
    );

    expect(result).toEqual([
      { type: "reasoning", text: "" },
      { type: "text", text: "Thinking disabled." },
    ]);
  });

  it("returns text when no reasoning markers are present", () => {
    const result = splitReasoningContent("Plain answer.", gemma4Markers);

    expect(result).toEqual([{ type: "text", text: "Plain answer." }]);
  });
});
