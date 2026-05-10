import { describe, it, expect } from "vitest";
import { llamaCpp, LlamaCppLanguageModel } from "../../src/index.js";

describe("llamaCpp", () => {
  describe("return value", () => {
    it("returns an instance of LlamaCppLanguageModel", () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(model).toBeInstanceOf(LlamaCppLanguageModel);
    });
  });

  describe("config propagation", () => {
    it("sets modelId to the model path", () => {
      const model = llamaCpp({
        modelPath: "/path/to/my-model.gguf",
      });

      expect(model.modelId).toBe("/path/to/my-model.gguf");
    });

    it("passes all config options through", () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
        contextSize: 4096,
        gpuLayers: 32,
        threads: 8,
        debug: true,
      });

      // We can verify the model was created (config is private, but modelId confirms path)
      expect(model.modelId).toBe("/path/to/model.gguf");
    });

    it("handles minimal config with only modelPath", () => {
      const model = llamaCpp({
        modelPath: "./models/test.gguf",
      });

      expect(model).toBeInstanceOf(LlamaCppLanguageModel);
      expect(model.modelId).toBe("./models/test.gguf");
    });
  });

  describe("LanguageModelV4 interface", () => {
    it('has specificationVersion "v4"', () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(model.specificationVersion).toBe("v4");
    });

    it('has provider "llama.cpp"', () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(model.provider).toBe("llama.cpp");
    });

    it("has empty supportedUrls (local files only)", () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(model.supportedUrls).toEqual({});
    });

    it("has doGenerate method", () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(typeof model.doGenerate).toBe("function");
    });

    it("has doStream method", () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(typeof model.doStream).toBe("function");
    });

    it("has dispose method", () => {
      const model = llamaCpp({
        modelPath: "/path/to/model.gguf",
      });

      expect(typeof model.dispose).toBe("function");
    });
  });

  describe("EmbeddingModelV4 interface", () => {
    it('has specificationVersion "v4"', () => {
      const model = llamaCpp.embedding({
        modelPath: "/path/to/model.gguf",
      });

      expect(model.specificationVersion).toBe("v4");
    });

    it('has provider "llama.cpp"', () => {
      const model = llamaCpp.embedding({
        modelPath: "/path/to/model.gguf",
      });

      expect(model.provider).toBe("llama.cpp");
    });

    it("has doEmbed method", () => {
      const model = llamaCpp.embedding({
        modelPath: "/path/to/model.gguf",
      });

      expect(typeof model.doEmbed).toBe("function");
    });

    it("has dispose method", () => {
      const model = llamaCpp.embedding({
        modelPath: "/path/to/model.gguf",
      });

      expect(typeof model.dispose).toBe("function");
    });
  });

  describe("multiple instances", () => {
    it("creates independent instances", () => {
      const model1 = llamaCpp({
        modelPath: "/path/to/model1.gguf",
      });
      const model2 = llamaCpp({
        modelPath: "/path/to/model2.gguf",
      });

      expect(model1).not.toBe(model2);
      expect(model1.modelId).toBe("/path/to/model1.gguf");
      expect(model2.modelId).toBe("/path/to/model2.gguf");
    });
  });
});

describe("LlamaCppLanguageModel constructor", () => {
  it("can be instantiated directly with config", () => {
    const model = new LlamaCppLanguageModel({
      modelPath: "/direct/path.gguf",
      contextSize: 1024,
    });

    expect(model.modelId).toBe("/direct/path.gguf");
  });
});
