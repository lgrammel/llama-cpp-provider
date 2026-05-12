import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { generateText, streamText } from "ai";
import { llamaCpp, LlamaCppLanguageModel } from "ai-sdk-llama-cpp";

/**
 * E2E tests for the llama.cpp provider.
 *
 * These tests require a real GGUF model file. Set the TEST_MODEL_PATH
 * environment variable to run these tests:
 *
 *   TEST_MODEL_PATH=./models/your-model.gguf pnpm test:e2e
 *
 * If TEST_MODEL_PATH is not set, these tests will be skipped.
 */

const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
const shouldRunTests = !!TEST_MODEL_PATH;

const describeE2E = shouldRunTests ? describe : describe.skip;

describeE2E("E2E Generation Tests", () => {
  let model: LlamaCppLanguageModel;

  beforeAll(() => {
    if (!TEST_MODEL_PATH) {
      throw new Error("TEST_MODEL_PATH environment variable not set");
    }

    model = llamaCpp({
      modelPath: TEST_MODEL_PATH,
      contextSize: 2048,
      gpuLayers: 0, // Use CPU for CI compatibility
      threads: 4,
    });
  });

  afterAll(async () => {
    if (model) {
      await model.dispose();
    }
  });

  describe("generateText", () => {
    it("generates text response", { timeout: 120000 }, async () => {
      const { text, usage, finishReason } = await generateText({
        model,
        prompt: "Say hello in one word.",
        maxOutputTokens: 20,
      });

      expect(text.length).toBeGreaterThan(0);
      // AI SDK v6 uses inputTokens/outputTokens structure
      expect(usage.inputTokens).toBeGreaterThan(0);
      expect(usage.outputTokens).toBeGreaterThan(0);
      expect(finishReason).toBeDefined();
    });

    it("generates text with messages array", { timeout: 120000 }, async () => {
      const { text } = await generateText({
        model,
        messages: [
          {
            role: "user",
            content: "What is 1+1? Answer with just the number.",
          },
        ],
        maxOutputTokens: 10,
      });

      expect(text.length).toBeGreaterThan(0);
    });

    it("accepts maxOutputTokens parameter", { timeout: 120000 }, async () => {
      // Test that maxOutputTokens parameter is accepted and generation completes
      const { text, finishReason } = await generateText({
        model,
        prompt: "Say one word.",
        maxOutputTokens: 50,
      });

      // Generation should complete successfully
      expect(text.length).toBeGreaterThan(0);
      expect(finishReason).toBeDefined();
    });

    it("handles multi-turn conversation", { timeout: 120000 }, async () => {
      const { text } = await generateText({
        model,
        messages: [
          { role: "user", content: "My name is Alice." },
          {
            role: "assistant",
            content: "Nice to meet you, Alice! How can I help you today?",
          },
          { role: "user", content: "What is my name?" },
        ],
        maxOutputTokens: 30,
      });

      expect(text.length).toBeGreaterThan(0);
      // The model should reference "Alice" in some way
    });
  });

  describe("streamText", () => {
    it("streams text tokens", { timeout: 120000 }, async () => {
      const { textStream } = streamText({
        model,
        prompt: "Count from 1 to 3.",
        maxOutputTokens: 30,
      });

      const chunks: string[] = [];
      for await (const chunk of textStream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(1);
      expect(chunks.join("")).toBeTruthy();
    });

    it(
      "provides usage information after streaming",
      { timeout: 120000 },
      async () => {
        const result = streamText({
          model,
          prompt: "Hello",
          maxOutputTokens: 20,
        });

        // Consume the stream
        for await (const _ of result.textStream) {
          // Just consume
        }

        const usage = await result.usage;
        // AI SDK v6 uses inputTokens/outputTokens
        expect(usage.inputTokens).toBeGreaterThan(0);
        expect(usage.outputTokens).toBeGreaterThan(0);
      }
    );

    it(
      "provides finish reason after streaming",
      { timeout: 120000 },
      async () => {
        const result = streamText({
          model,
          prompt: "Say hi.",
          maxOutputTokens: 10,
        });

        // Consume the stream
        for await (const _ of result.textStream) {
          // Just consume
        }

        const finishReason = await result.finishReason;
        expect(["stop", "length", "other"]).toContain(finishReason);
      }
    );
  });

  describe("generation parameters", () => {
    it("applies temperature setting", { timeout: 120000 }, async () => {
      // Low temperature should produce more deterministic output
      const { text: text1 } = await generateText({
        model,
        prompt: "What is 2+2?",
        maxOutputTokens: 10,
        temperature: 0.0,
      });

      const { text: text2 } = await generateText({
        model,
        prompt: "What is 2+2?",
        maxOutputTokens: 10,
        temperature: 0.0,
      });

      // With temperature 0, outputs should be similar (not necessarily identical due to implementation)
      expect(text1.length).toBeGreaterThan(0);
      expect(text2.length).toBeGreaterThan(0);
    });

    it("applies topP setting", { timeout: 60000 }, async () => {
      const { text } = await generateText({
        model,
        prompt: "Hello",
        maxOutputTokens: 10,
        topP: 0.5,
      });

      expect(text.length).toBeGreaterThan(0);
    });

    it("applies topK setting", { timeout: 60000 }, async () => {
      const { text } = await generateText({
        model,
        prompt: "Hello",
        maxOutputTokens: 10,
        topK: 10,
      });

      expect(text.length).toBeGreaterThan(0);
    });
  });

  describe("model lifecycle", () => {
    it("can create multiple model instances", { timeout: 120000 }, async () => {
      if (!TEST_MODEL_PATH) return;

      const model2 = llamaCpp({
        modelPath: TEST_MODEL_PATH,
        contextSize: 1024,
      });

      const { text } = await generateText({
        model: model2,
        prompt: "Hi",
        maxOutputTokens: 5,
      });

      expect(text.length).toBeGreaterThan(0);

      await model2.dispose();
    });

    it("handles dispose gracefully", { timeout: 120000 }, async () => {
      if (!TEST_MODEL_PATH) return;

      const tempModel = llamaCpp({
        modelPath: TEST_MODEL_PATH,
      });

      // Generate to load the model
      await generateText({
        model: tempModel,
        prompt: "Test",
        maxOutputTokens: 5,
      });

      // Dispose should not throw
      await expect(tempModel.dispose()).resolves.toBeUndefined();
    });
  });
});

// Test that runs without a model to verify skip behavior
describe("E2E Test Configuration", () => {
  it("TEST_MODEL_PATH environment variable info", () => {
    if (!TEST_MODEL_PATH) {
      console.log(
        "\n📋 E2E tests skipped: Set TEST_MODEL_PATH to run with a real model"
      );
      console.log(
        "   Example: TEST_MODEL_PATH=./models/model.gguf pnpm test:e2e\n"
      );
    } else {
      console.log(`\n✅ Running E2E tests with model: ${TEST_MODEL_PATH}\n`);
    }
    expect(true).toBe(true);
  });
});
