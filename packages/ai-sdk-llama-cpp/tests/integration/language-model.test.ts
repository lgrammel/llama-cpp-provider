import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { LanguageModelV4Message } from "@ai-sdk/provider";

// Mock the native binding module before importing the language model
vi.mock("../../src/native-binding.js", () => ({
  loadModel: vi.fn().mockResolvedValue(1),
  unloadModel: vi.fn().mockReturnValue(true),
  generate: vi.fn().mockResolvedValue({
    text: "Mock response text",
    promptTokens: 50,
    completionTokens: 10,
    finishReason: "stop",
  }),
  generateStream: vi.fn((handle, opts, onToken) => {
    // Simulate streaming tokens
    onToken("Hello");
    onToken(" ");
    onToken("world");
    onToken("!");
    return Promise.resolve({
      text: "Hello world!",
      promptTokens: 30,
      completionTokens: 4,
      finishReason: "stop",
    });
  }),
  isModelLoaded: vi.fn().mockReturnValue(true),
}));

// Import after mocking
import { LlamaCppLanguageModel } from "../../src/llama-cpp-language-model.js";
import * as nativeBinding from "../../src/native-binding.js";
import { gemma4Reasoning } from "../../src/gemma4.js";

describe("LlamaCppLanguageModel Integration", () => {
  let model: LlamaCppLanguageModel;

  beforeEach(() => {
    vi.clearAllMocks();
    model = new LlamaCppLanguageModel({
      modelPath: "/test/model.gguf",
      contextSize: 2048,
      gpuLayers: 99,
      threads: 4,
    });
  });

  afterEach(async () => {
    await model.dispose();
  });

  describe("doGenerate", () => {
    const testMessages: LanguageModelV4Message[] = [
      {
        role: "user",
        content: [{ type: "text", text: "Hello, how are you?" }],
      },
    ];

    it("returns valid LanguageModelV4GenerateResult structure", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 100,
      });

      expect(result).toHaveProperty("content");
      expect(result).toHaveProperty("finishReason");
      expect(result).toHaveProperty("usage");
      expect(result).toHaveProperty("warnings");
      expect(result).toHaveProperty("request");
    });

    it("returns text content from generation", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe("text");
      expect(result.content[0]).toHaveProperty("text", "Mock response text");
    });

    it("returns correct finish reason", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.finishReason.unified).toBe("stop");
      expect(result.finishReason.raw).toBe("stop");
    });

    it("returns correct usage statistics", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.usage.inputTokens.total).toBe(50);
      expect(result.usage.outputTokens.total).toBe(10);
    });

    it("returns empty warnings array", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.warnings).toEqual([]);
    });

    it("includes request body in result", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 256,
      });

      expect(result.request).toHaveProperty("body");
      expect(result.request!.body).toHaveProperty("messages");
      expect(result.request!.body).toHaveProperty("maxTokens", 256);
    });

    it("applies default maxTokens when not specified", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.request!.body).toHaveProperty("maxTokens", 2048);
    });

    it("applies default temperature when not specified", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.request!.body).toHaveProperty("temperature", 0.7);
    });

    it("passes custom generation options", async () => {
      await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 500,
        temperature: 0.5,
        topP: 0.8,
        topK: 30,
        stopSequences: ["END"],
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          maxTokens: 500,
          temperature: 0.5,
          topP: 0.8,
          topK: 30,
          stopSequences: ["END"],
        })
      );
    });

    it("passes messages correctly to native binding", async () => {
      await model.doGenerate({
        prompt: testMessages,
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          messages: [{ role: "user", content: "Hello, how are you?" }],
        })
      );
    });

    it("passes image inputs to native binding", async () => {
      const imageData = new Uint8Array([1, 2, 3]);
      await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              { type: "text", text: "Describe this image." },
              {
                type: "file",
                data: { type: "data", data: imageData },
                mediaType: "image/png",
              },
            ],
          },
        ],
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          messages: [
            {
              role: "user",
              content: "Describe this image.\n<__media__>",
              images: [{ data: imageData, mediaType: "image/png" }],
            },
          ],
        })
      );
    });

    it("extracts default think-tag reasoning when reasoning is omitted", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "<think>Think first.</think>Final answer.",
        promptTokens: 50,
        completionTokens: 12,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.content).toEqual([
        {
          type: "reasoning",
          text: "Think first.",
          providerMetadata: undefined,
        },
        {
          type: "text",
          text: "Final answer.",
          providerMetadata: undefined,
        },
      ]);
    });

    it('disables reasoning extraction when reasoning is "none"', async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "<think>Hidden thinking.</think>Visible answer.",
        promptTokens: 50,
        completionTokens: 12,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        reasoning: "none",
      });

      expect(result.content).toEqual([
        {
          type: "text",
          text: "<think>Hidden thinking.</think>Visible answer.",
          providerMetadata: undefined,
        },
      ]);
    });

    it('enables reasoning extraction for non-"none" reasoning values', async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "<think>Use more effort.</think>Better answer.",
        promptTokens: 50,
        completionTokens: 12,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        reasoning: "high",
      });

      expect(result.content).toEqual([
        {
          type: "reasoning",
          text: "Use more effort.",
          providerMetadata: undefined,
        },
        {
          type: "text",
          text: "Better answer.",
          providerMetadata: undefined,
        },
      ]);
    });

    it("returns Gemma 4 thinking as reasoning content", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "<|channel>thought\nI should answer briefly.<channel|>Hello!",
        promptTokens: 50,
        completionTokens: 12,
        finishReason: "stop",
      });

      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: gemma4Reasoning,
      });

      const result = await reasoningModel.doGenerate({
        prompt: testMessages,
      });

      expect(result.content).toEqual([
        {
          type: "reasoning",
          text: "I should answer briefly.",
          providerMetadata: undefined,
        },
        {
          type: "text",
          text: "Hello!",
          providerMetadata: undefined,
        },
      ]);

      await reasoningModel.dispose();
    });

    it("uses visible text for tool call parsing after reasoning", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '<|channel>thought\nNeed weather.<channel|>{"name":"weather","arguments":{"location":"Tokyo"}}',
        promptTokens: 50,
        completionTokens: 20,
        finishReason: "stop",
      });

      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: gemma4Reasoning,
      });

      const result = await reasoningModel.doGenerate({
        prompt: testMessages,
        tools: [
          {
            type: "function",
            name: "weather",
            inputSchema: { type: "object" },
          },
        ],
      });

      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({
        type: "reasoning",
        text: "Need weather.",
        providerMetadata: undefined,
      });
      expect(result.content[1].type).toBe("tool-call");
      expect(result.content[1]).toHaveProperty("toolName", "weather");
      expect(result.finishReason.unified).toBe("tool-calls");

      await reasoningModel.dispose();
    });

    it("does not inject reasoning prompt when constrained by JSON grammar", async () => {
      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: gemma4Reasoning,
      });

      await reasoningModel.doGenerate({
        prompt: testMessages,
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: {
              answer: { type: "string" },
            },
            required: ["answer"],
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          messages: [{ role: "user", content: "Hello, how are you?" }],
          grammar: expect.stringContaining("root ::="),
        })
      );

      await reasoningModel.dispose();
    });
  });

  describe("doStream", () => {
    const testMessages: LanguageModelV4Message[] = [
      {
        role: "user",
        content: [{ type: "text", text: "Count to 3" }],
      },
    ];

    it("returns a ReadableStream", async () => {
      const result = await model.doStream({
        prompt: testMessages,
      });

      expect(result).toHaveProperty("stream");
      expect(result.stream).toBeInstanceOf(ReadableStream);
    });

    it("includes request body in result", async () => {
      const result = await model.doStream({
        prompt: testMessages,
        maxOutputTokens: 100,
      });

      expect(result.request).toHaveProperty("body");
      expect(result.request!.body).toHaveProperty("maxTokens", 100);
    });

    it("emits stream-start as first part", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const reader = stream.getReader();
      const { value: firstPart } = await reader.read();

      expect(firstPart?.type).toBe("stream-start");
      expect(firstPart).toHaveProperty("warnings", []);

      reader.releaseLock();
    });

    it("emits text-start after stream-start", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);

      expect(parts[0].type).toBe("stream-start");
      expect(parts[1].type).toBe("text-start");
      expect(parts[1]).toHaveProperty("id");
    });

    it("emits text-delta parts for each token", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const textDeltas = parts.filter((p) => p.type === "text-delta");

      expect(textDeltas.length).toBe(4); // "Hello", " ", "world", "!"
      expect(textDeltas[0]).toHaveProperty("delta", "Hello");
      expect(textDeltas[1]).toHaveProperty("delta", " ");
      expect(textDeltas[2]).toHaveProperty("delta", "world");
      expect(textDeltas[3]).toHaveProperty("delta", "!");
    });

    it("emits text-end after all text-deltas", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const textEndIdx = parts.findIndex((p) => p.type === "text-end");
      const lastTextDeltaIdx = parts.reduce(
        (acc, p, i) => (p.type === "text-delta" ? i : acc),
        -1
      );

      expect(textEndIdx).toBeGreaterThan(lastTextDeltaIdx);
    });

    it("emits finish as last part", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const lastPart = parts[parts.length - 1];

      expect(lastPart.type).toBe("finish");
      expect(lastPart).toHaveProperty("finishReason");
      expect(lastPart).toHaveProperty("usage");
    });

    it("finish part contains correct finish reason", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const finishPart = parts.find((p) => p.type === "finish");

      expect(finishPart?.finishReason?.unified).toBe("stop");
    });

    it("finish part contains correct usage", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const finishPart = parts.find((p) => p.type === "finish");

      expect(finishPart?.usage?.inputTokens.total).toBe(30);
      expect(finishPart?.usage?.outputTokens.total).toBe(4);
    });

    it("all text-delta parts share the same id", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const textStart = parts.find((p) => p.type === "text-start");
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      const textEnd = parts.find((p) => p.type === "text-end");

      const expectedId = textStart?.id;
      expect(expectedId).toBeDefined();

      for (const delta of textDeltas) {
        expect(delta.id).toBe(expectedId);
      }
      expect(textEnd?.id).toBe(expectedId);
    });

    it("streams Gemma 4 thinking as reasoning deltas", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken("<|channel>");
          onToken("thought\nI should");
          onToken(" answer");
          onToken(" briefly.<channel");
          onToken("|>Hello");
          onToken("!");
          return Promise.resolve({
            text: "<|channel>thought\nI should answer briefly.<channel|>Hello!",
            promptTokens: 30,
            completionTokens: 12,
            finishReason: "stop",
          });
        }
      );

      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: gemma4Reasoning,
      });

      const { stream } = await reasoningModel.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const reasoningDeltas = parts.filter((p) => p.type === "reasoning-delta");
      const textDeltas = parts.filter((p) => p.type === "text-delta");

      expect(parts.map((p) => p.type)).toEqual([
        "stream-start",
        "reasoning-start",
        "reasoning-delta",
        "reasoning-delta",
        "reasoning-delta",
        "reasoning-end",
        "text-start",
        "text-delta",
        "text-delta",
        "text-end",
        "finish",
      ]);
      expect(reasoningDeltas.map((p) => p.delta).join("")).toBe(
        "I should answer briefly."
      );
      expect(textDeltas.map((p) => p.delta).join("")).toBe("Hello!");

      await reasoningModel.dispose();
    });
  });

  describe("model loading", () => {
    it("loads model lazily on first generation", async () => {
      const freshModel = new LlamaCppLanguageModel({
        modelPath: "/test/lazy.gguf",
      });

      expect(nativeBinding.loadModel).not.toHaveBeenCalled();

      await freshModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledTimes(1);

      await freshModel.dispose();
    });

    it("reuses loaded model on subsequent calls", async () => {
      const messages: LanguageModelV4Message[] = [
        { role: "user", content: [{ type: "text", text: "test" }] },
      ];

      await model.doGenerate({ prompt: messages });
      await model.doGenerate({ prompt: messages });
      await model.doGenerate({ prompt: messages });

      expect(nativeBinding.loadModel).toHaveBeenCalledTimes(1);
    });

    it("passes correct options to loadModel", async () => {
      const customModel = new LlamaCppLanguageModel({
        modelPath: "/custom/path.gguf",
        contextSize: 4096,
        gpuLayers: 32,
        threads: 8,
        debug: true,
        chatTemplate: "llama3",
        mmprojPath: "/custom/mmproj.gguf",
        memorySafety: { mode: "off" },
      });

      await customModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledWith({
        modelPath: "/custom/path.gguf",
        contextSize: 4096,
        gpuLayers: 32,
        threads: 8,
        debug: true,
        chatTemplate: "llama3",
        mmprojPath: "/custom/mmproj.gguf",
      });

      await customModel.dispose();
    });

    it("throws before loading when context exceeds model maximum", async () => {
      const guardedModel = new LlamaCppLanguageModel({
        modelPath: "/custom/path.gguf",
        contextSize: 4097,
        memory: {
          maxContextSize: 4096,
          kvCache: {
            layers: [{ count: 1, keyValueHeads: 1, headDim: 1 }],
          },
        },
        memorySafety: {
          maxMemoryBytes: Number.MAX_SAFE_INTEGER,
        },
      });

      await expect(
        guardedModel.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        })
      ).rejects.toThrow("exceeds the model maximum context size");

      expect(nativeBinding.loadModel).not.toHaveBeenCalled();
    });

    it("clamps context before loading when configured", async () => {
      const clampedModel = new LlamaCppLanguageModel({
        modelPath: "/custom/path.gguf",
        contextSize: 4096,
        memory: {
          kvCache: {
            layers: [{ count: 1, keyValueHeads: 1, headDim: 1 }],
          },
        },
        memorySafety: {
          mode: "clamp",
          maxMemoryBytes: 512,
          computeOverheadBytes: 0,
        },
      });

      await clampedModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        maxOutputTokens: 128,
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledWith(
        expect.objectContaining({
          contextSize: 128,
        })
      );

      await clampedModel.dispose();
    });

    it("throws when maxOutputTokens exceeds loaded context size", async () => {
      const smallContextModel = new LlamaCppLanguageModel({
        modelPath: "/custom/path.gguf",
        contextSize: 16,
      });

      await expect(
        smallContextModel.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
          maxOutputTokens: 17,
        })
      ).rejects.toThrow("maxOutputTokens 17 exceeds the loaded contextSize 16");

      await smallContextModel.dispose();
    });

    it("uses default values for optional config", async () => {
      const minimalModel = new LlamaCppLanguageModel({
        modelPath: "/minimal.gguf",
      });

      await minimalModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledWith({
        modelPath: "/minimal.gguf",
        contextSize: 2048,
        gpuLayers: 99,
        threads: 4,
        debug: false,
        chatTemplate: "auto",
      });

      await minimalModel.dispose();
    });

    it("omits undefined optional native load options", async () => {
      const modelWithoutMmproj = new LlamaCppLanguageModel({
        modelPath: "/minimal.gguf",
        mmprojPath: undefined,
      });

      await modelWithoutMmproj.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledWith(
        expect.not.objectContaining({
          mmprojPath: undefined,
        })
      );

      await modelWithoutMmproj.dispose();
    });
  });

  describe("dispose", () => {
    it("calls unloadModel with handle", async () => {
      // First generate to load the model
      await model.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      await model.dispose();

      expect(nativeBinding.unloadModel).toHaveBeenCalledWith(1);
    });

    it("can be called multiple times safely", async () => {
      await model.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      await model.dispose();
      await model.dispose();
      await model.dispose();

      // Should only unload once since modelHandle is null after first dispose
      expect(nativeBinding.unloadModel).toHaveBeenCalledTimes(1);
    });

    it("does nothing if model was never loaded", async () => {
      const unusedModel = new LlamaCppLanguageModel({
        modelPath: "/unused.gguf",
      });

      await unusedModel.dispose();

      expect(nativeBinding.unloadModel).not.toHaveBeenCalled();
    });
  });

  describe("error handling", () => {
    it("propagates generation errors", async () => {
      vi.mocked(nativeBinding.generate).mockRejectedValueOnce(
        new Error("Generation failed")
      );

      await expect(
        model.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        })
      ).rejects.toThrow("Generation failed");
    });

    it("propagates model loading errors", async () => {
      vi.mocked(nativeBinding.loadModel).mockRejectedValueOnce(
        new Error("Model not found")
      );

      const badModel = new LlamaCppLanguageModel({
        modelPath: "/nonexistent.gguf",
      });

      await expect(
        badModel.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        })
      ).rejects.toThrow("Model not found");
    });

    it("errors the stream on failure", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(() => {
        throw new Error("Stream generation failed");
      });

      const { stream } = await model.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      await expect(collectStreamParts(stream)).rejects.toThrow(
        "Stream generation failed"
      );
    });
  });

  describe("doGenerate with tools", () => {
    const testMessages: LanguageModelV4Message[] = [
      {
        role: "user",
        content: [{ type: "text", text: "What is the weather in Tokyo?" }],
      },
    ];

    const testTools = [
      {
        type: "function" as const,
        name: "get_weather",
        description: "Get the current weather",
        inputSchema: {
          type: "object",
          properties: {
            location: { type: "string" },
          },
        },
      },
    ];

    it("returns tool-call content when model outputs tool call JSON", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
        promptTokens: 100,
        completionTokens: 20,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
      });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe("tool-call");
      expect(result.content[0]).toHaveProperty("toolName", "get_weather");
      expect(result.content[0]).toHaveProperty("toolCallId");
      expect(result.content[0]).toHaveProperty("input", '{"location":"Tokyo"}');
    });

    it("returns tool-calls finish reason when tool call detected", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '{"name": "get_weather", "arguments": {"location": "Paris"}}',
        promptTokens: 100,
        completionTokens: 20,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
      });

      expect(result.finishReason.unified).toBe("tool-calls");
      expect(result.finishReason.raw).toBe("tool-calls");
    });

    it("returns text content when model outputs regular text with tools", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "I cannot check the weather without more information.",
        promptTokens: 100,
        completionTokens: 15,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
      });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe("text");
      expect(result.content[0]).toHaveProperty(
        "text",
        "I cannot check the weather without more information."
      );
      expect(result.finishReason.unified).toBe("stop");
    });

    it("handles array of tool calls", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '[{"name": "get_weather", "arguments": {"location": "Tokyo"}}, {"name": "get_weather", "arguments": {"location": "Paris"}}]',
        promptTokens: 100,
        completionTokens: 40,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
      });

      expect(result.content).toHaveLength(2);
      expect(result.content[0].type).toBe("tool-call");
      expect(result.content[1].type).toBe("tool-call");
    });

    it("does not parse tool calls when toolChoice is none", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
        promptTokens: 100,
        completionTokens: 20,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
        toolChoice: { type: "none" },
      });

      // Should be returned as text, not parsed as tool call
      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe("text");
    });
  });

  describe("doStream with tools", () => {
    const testMessages: LanguageModelV4Message[] = [
      {
        role: "user",
        content: [{ type: "text", text: "What is the weather?" }],
      },
    ];

    const testTools = [
      {
        type: "function" as const,
        name: "weather",
        description: "Get weather",
        inputSchema: {
          type: "object",
          properties: {
            location: { type: "string" },
          },
        },
      },
    ];

    it("suppresses text deltas when output is a tool call", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          // Simulate streaming a tool call JSON
          onToken("{");
          onToken('"name"');
          onToken(": ");
          onToken('"weather"');
          onToken(", ");
          onToken('"arguments"');
          onToken(": {");
          onToken('"location"');
          onToken(": ");
          onToken('"Tokyo"');
          onToken("}}");
          return Promise.resolve({
            text: '{"name": "weather", "arguments": {"location": "Tokyo"}}',
            promptTokens: 50,
            completionTokens: 15,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);

      // Should NOT have text-start, text-delta, or text-end events
      const textParts = parts.filter(
        (p) =>
          p.type === "text-start" ||
          p.type === "text-delta" ||
          p.type === "text-end"
      );
      expect(textParts).toHaveLength(0);

      // Should have tool-call event
      const toolCallPart = parts.find((p) => p.type === "tool-call");
      expect(toolCallPart).toBeDefined();
      expect(toolCallPart?.toolName).toBe("weather");
      expect(toolCallPart?.input).toBe('{"location":"Tokyo"}');
    });

    it("emits tool-calls finish reason when tool call detected", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken('{"name": "weather", "arguments": {"location": "NYC"}}');
          return Promise.resolve({
            text: '{"name": "weather", "arguments": {"location": "NYC"}}',
            promptTokens: 50,
            completionTokens: 15,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);
      const finishPart = parts.find((p) => p.type === "finish");

      expect(finishPart?.finishReason?.unified).toBe("tool-calls");
    });

    it("streams text normally when output is not a tool call", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          // Simulate regular text output (starts with letter, not JSON)
          onToken("The");
          onToken(" weather");
          onToken(" is");
          onToken(" nice");
          return Promise.resolve({
            text: "The weather is nice",
            promptTokens: 50,
            completionTokens: 4,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);

      // Should have text events
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      expect(textDeltas.length).toBe(4);
      expect(textDeltas[0]?.delta).toBe("The");

      // Should NOT have tool-call event
      const toolCallPart = parts.find((p) => p.type === "tool-call");
      expect(toolCallPart).toBeUndefined();
    });

    it("does not suppress text when toolChoice is none", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken('{"name": "weather"}');
          return Promise.resolve({
            text: '{"name": "weather"}',
            promptTokens: 50,
            completionTokens: 5,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
        toolChoice: { type: "none" },
      });

      const parts = await collectStreamParts(stream);

      // Should have text events (not suppressed because toolChoice is none)
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      expect(textDeltas.length).toBeGreaterThan(0);

      // Should NOT have tool-call event
      const toolCallPart = parts.find((p) => p.type === "tool-call");
      expect(toolCallPart).toBeUndefined();
    });

    it("handles leading whitespace before tool call JSON", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          // Simulate output with leading whitespace
          onToken("  ");
          onToken("\n");
          onToken('{"name": "weather", "arguments": {}}');
          return Promise.resolve({
            text: '  \n{"name": "weather", "arguments": {}}',
            promptTokens: 50,
            completionTokens: 10,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);

      // Should suppress text deltas and emit tool-call
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      expect(textDeltas).toHaveLength(0);

      const toolCallPart = parts.find((p) => p.type === "tool-call");
      expect(toolCallPart).toBeDefined();
      expect(toolCallPart?.toolName).toBe("weather");
    });
  });
});

// Helper function to collect all stream parts
async function collectStreamParts(stream: ReadableStream<any>): Promise<any[]> {
  const parts: any[] = [];
  const reader = stream.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }

  return parts;
}
