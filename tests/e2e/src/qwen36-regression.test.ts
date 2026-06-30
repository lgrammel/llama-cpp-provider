import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod/v4";
import {
  llamaCpp,
  qwen3_6_dense,
  type LlamaCppLanguageModel,
} from "@lgrammel/llama-cpp-provider";
import { formatModelInfo, modelRuntimeConfig } from "./e2e-config.js";

const TEST_QWEN36_MODEL_PATH =
  process.env.TEST_QWEN36_MODEL_PATH ??
  (process.env.TEST_MODEL_PATH?.includes("Qwen3.6")
    ? process.env.TEST_MODEL_PATH
    : undefined);
const describeQwen36E2E = TEST_QWEN36_MODEL_PATH ? describe : describe.skip;

describeQwen36E2E("Qwen 3.6 E2E regressions", () => {
  let model: LlamaCppLanguageModel;

  beforeAll(() => {
    if (!TEST_QWEN36_MODEL_PATH) {
      throw new Error("TEST_QWEN36_MODEL_PATH environment variable not set");
    }

    console.log(
      `Running Qwen 3.6 regression tests with model: ${formatModelInfo(TEST_QWEN36_MODEL_PATH)}`
    );

    model = llamaCpp({
      modelPath: TEST_QWEN36_MODEL_PATH,
      memorySafety: { mode: "off" },
      cache: { mode: "prefix" },
      model: qwen3_6_dense,
      ...modelRuntimeConfig({ contextSize: 4096, threads: 4 }),
    });
  });

  afterAll(async () => {
    await model?.dispose();
  });

  it(
    "uses the sandbox shell tool without leaking chat-template or thinking markup",
    { timeout: 180000 },
    async () => {
      const result = await generateText({
        model,
        prompt:
          "Use the sandboxshell tool to check whether ffmpeg is installed. " +
          "Call it with cmd exactly: which ffmpeg && ffmpeg -version",
        maxOutputTokens: 192,
        temperature: 0,
        stopWhen: stepCountIs(1),
        tools: {
          sandboxshell: tool({
            description: "Run a shell command in the sandbox.",
            inputSchema: z.object({ cmd: z.string() }),
          }),
        },
      });

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0]).toMatchObject({
        toolName: "sandboxshell",
        input: { cmd: "which ffmpeg && ffmpeg -version" },
      });

      const visibleText = result.steps.map((step) => step.text).join("\n");
      expect(visibleText).not.toContain("<|im_start|>");
      expect(visibleText).not.toContain("<think>");
      expect(visibleText).not.toContain("</think>");
      expect(visibleText).not.toContain("<sandboxshell:cmd>");
    }
  );
});
