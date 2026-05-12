import { describe, expect, it } from "vitest";
import {
  estimateMemoryUsage,
  checkMemorySafety,
  gemma4_26b_a4b,
  gemma4_31b_it,
  qwen3_6_dense,
  qwen3_6_moe,
} from "../../src/index.js";

describe("estimateMemoryUsage", () => {
  it("estimates KV cache and total memory from model metadata", () => {
    const estimate = estimateMemoryUsage({
      model: gemma4_31b_it.memory!,
      contextSize: 1024,
      modelFileSizeBytes: 20,
      mmprojFileSizeBytes: 10,
      computeOverheadBytes: 5,
    });

    expect(estimate.bytesPerToken).toBe(901120);
    expect(estimate.kvCacheBytes).toBe(901120 * 1024);
    expect(estimate.totalBytes).toBe(901120 * 1024 + 35);
    expect(estimate.maxContextSize).toBe(262144);
  });

  it("supports Qwen dense and MoE presets", () => {
    expect(
      estimateMemoryUsage({
        model: qwen3_6_dense.memory!,
        contextSize: 1,
        computeOverheadBytes: 0,
      }).bytesPerToken
    ).toBeGreaterThan(0);

    expect(
      estimateMemoryUsage({
        model: qwen3_6_moe.memory!,
        contextSize: 1,
        computeOverheadBytes: 0,
      }).bytesPerToken
    ).toBeGreaterThan(0);
  });

  it("exports the Gemma 26B A4B preset with memory metadata", () => {
    expect(gemma4_26b_a4b).toBeDefined();
    expect(gemma4_26b_a4b.memory).toBeDefined();
  });

  it("validates context size", () => {
    expect(() =>
      estimateMemoryUsage({
        model: gemma4_31b_it.memory!,
        contextSize: 0,
      })
    ).toThrow("contextSize must be a positive integer");
  });
});

describe("checkMemorySafety", () => {
  it("returns the requested context size when no model metadata is provided", () => {
    expect(checkMemorySafety({ contextSize: 100000 })).toEqual({
      contextSize: 100000,
    });
  });

  it("throws when requested context exceeds model maximum", () => {
    expect(() =>
      checkMemorySafety({
        model: gemma4_31b_it.memory,
        contextSize: 262145,
        memorySafety: { maxMemoryBytes: Number.MAX_SAFE_INTEGER },
      })
    ).toThrow("exceeds the model maximum context size");
  });

  it("throws when estimated memory exceeds the safety budget", () => {
    expect(() =>
      checkMemorySafety({
        model: gemma4_31b_it.memory,
        contextSize: 4096,
        memorySafety: { maxMemoryBytes: 1024 },
      })
    ).toThrow("is estimated to require");
  });

  it("clamps context size when configured", () => {
    const result = checkMemorySafety({
      model: gemma4_31b_it.memory,
      contextSize: 4096,
      memorySafety: {
        mode: "clamp",
        maxMemoryBytes: 901120 * 128,
        computeOverheadBytes: 0,
      },
    });

    expect(result.contextSize).toBe(128);
  });
});
