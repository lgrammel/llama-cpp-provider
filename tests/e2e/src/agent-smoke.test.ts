import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { generateText, Output, streamText, tool } from "ai";
import { z } from "zod/v4";
import {
  llamaCpp,
  type LlamaCppLanguageModel,
} from "@lgrammel/llama-cpp-provider";
import {
  formatModelInfo,
  languageModelConfig,
  modelRuntimeConfig,
} from "./e2e-config.js";

const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
const describeAgentE2E = TEST_MODEL_PATH ? describe : describe.skip;

describeAgentE2E("Agent E2E smoke tests", () => {
  let model: LlamaCppLanguageModel;

  beforeAll(() => {
    if (!TEST_MODEL_PATH) {
      throw new Error("TEST_MODEL_PATH environment variable not set");
    }

    console.log(
      `Running agent E2E smoke tests with model: ${formatModelInfo(TEST_MODEL_PATH)}`
    );

    model = llamaCpp(
      languageModelConfig({
        modelPath: TEST_MODEL_PATH,
        ...modelRuntimeConfig({ contextSize: 1024, threads: 2 }),
      })
    );
  });

  afterAll(async () => {
    await model?.dispose();
  });

  it("generates text with usage and finish metadata", async () => {
    const result = await generateText({
      model,
      prompt: "Reply with a short greeting.",
      maxOutputTokens: 16,
      temperature: 0,
    });

    expect(result.text.trim().length).toBeGreaterThan(0);
    expect(result.usage.inputTokens).toBeGreaterThan(0);
    expect(result.usage.outputTokens).toBeGreaterThan(0);
    expect(["stop", "length", "other"]).toContain(result.finishReason);
  });

  it("streams text chunks and resolves usage", async () => {
    const result = streamText({
      model,
      prompt: "Count 1, 2, 3.",
      maxOutputTokens: 20,
      temperature: 0,
    });

    const chunks: string[] = [];
    for await (const chunk of result.textStream) {
      chunks.push(chunk);
    }

    const usage = await result.usage;
    const finishReason = await result.finishReason;

    expect(chunks.join("").trim().length).toBeGreaterThan(0);
    expect(usage.inputTokens).toBeGreaterThan(0);
    expect(usage.outputTokens).toBeGreaterThan(0);
    expect(["stop", "length", "other"]).toContain(finishReason);
  });

  it("constrains structured output with an enum schema", async () => {
    const schema = z.object({
      status: z.enum(["ok"]),
    });

    const { output } = await generateText({
      model,
      output: Output.object({ schema }),
      prompt: 'Generate JSON with status set to "ok".',
      maxOutputTokens: 32,
      temperature: 0,
    });

    expect(output).toEqual({ status: "ok" });
  });

  it("honors toolChoice none without invoking tools", async () => {
    let called = false;

    const result = await generateText({
      model,
      prompt: "Say hello without using tools.",
      maxOutputTokens: 32,
      temperature: 0,
      toolChoice: "none",
      tools: {
        get_weather: tool({
          description: "Get the weather for a city",
          inputSchema: z.object({ city: z.string() }),
          execute: async () => {
            called = true;
            return { temperature: 20 };
          },
        }),
      },
    });

    expect(result.text.trim().length).toBeGreaterThan(0);
    expect(called).toBe(false);
    expect(
      result.steps.some((step) => step.toolCalls && step.toolCalls.length > 0)
    ).toBe(false);
  });
});
