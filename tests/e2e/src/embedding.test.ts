import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { embed, embedMany } from "ai";
import { llamaCpp, type LlamaCppEmbeddingModel } from "ai-sdk-llama-cpp";

/**
 * E2E tests for the llama.cpp embedding provider.
 *
 * These tests require a real GGUF model file. Set the TEST_EMBEDDING_PATH
 * environment variable to run these tests:
 *
 *   TEST_EMBEDDING_PATH=./models/your-model.gguf pnpm test:e2e
 *
 * If TEST_EMBEDDING_PATH is not set, these tests will be skipped.
 */

const TEST_EMBEDDING_PATH = process.env.TEST_EMBEDDING_PATH;
const shouldRunTests = !!TEST_EMBEDDING_PATH;

const describeE2E = shouldRunTests ? describe : describe.skip;

describeE2E("E2E Embedding Tests", () => {
  let model: LlamaCppEmbeddingModel;

  beforeAll(() => {
    if (!TEST_EMBEDDING_PATH) {
      throw new Error("TEST_EMBEDDING_PATH environment variable not set");
    }

    model = llamaCpp.embedding({
      modelPath: TEST_EMBEDDING_PATH,
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

  describe("embed", () => {
    it("embeds text", { timeout: 120000 }, async () => {
      const { embedding, usage } = await embed({
        model,
        value: "Hello, world!",
      });

      expect(embedding.length).toBeGreaterThan(0);
      expect(usage.tokens).toBeGreaterThan(0);
    });

    it("embeds multiple texts", { timeout: 120000 }, async () => {
      const { embeddings, usage } = await embedMany({
        model,
        values: ["Hello, world!", "Hello, universe!"],
      });

      expect(embeddings.length).toBe(2);
      expect(usage.tokens).toBeGreaterThan(0);
    });
  });

  describe("model lifecycle", () => {
    it("can create multiple model instances", { timeout: 120000 }, async () => {
      if (!TEST_EMBEDDING_PATH) return;

      const model2 = llamaCpp.embedding({
        modelPath: TEST_EMBEDDING_PATH,
        contextSize: 1024,
      });

      const { embedding } = await embed({
        model: model2,
        value: "Hello, world!",
      });

      expect(embedding.length).toBeGreaterThan(0);

      await model2.dispose();
    });

    it("handles dispose gracefully", { timeout: 120000 }, async () => {
      if (!TEST_EMBEDDING_PATH) return;

      const tempModel = llamaCpp.embedding({
        modelPath: TEST_EMBEDDING_PATH,
      });

      // Embed to load the model
      await embed({
        model: tempModel,
        value: "Hello, world!",
      });

      // Dispose should not throw
      await expect(tempModel.dispose()).resolves.toBeUndefined();
    });
  });
});

// Test that runs without a model to verify skip behavior
describe("E2E Test Configuration", () => {
  it("TEST_EMBEDDING_PATH environment variable info", () => {
    if (!TEST_EMBEDDING_PATH) {
      console.log(
        "\n📋 E2E tests skipped: Set TEST_EMBEDDING_PATH to run with a real model"
      );
      console.log(
        "   Example: TEST_EMBEDDING_PATH=./models/model.gguf pnpm test:e2e\n"
      );
    } else {
      console.log(
        `\n✅ Running E2E tests with model: ${TEST_EMBEDDING_PATH}\n`
      );
    }
    expect(true).toBe(true);
  });
});
