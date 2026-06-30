import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { LanguageModelV4Message } from "@ai-sdk/provider";

// Mock the native binding module before importing the language model
vi.mock("../../src/native-binding.js", () => ({
  loadModel: vi.fn().mockResolvedValue(1),
  unloadModel: vi.fn().mockReturnValue(true),
  cancelGeneration: vi.fn().mockReturnValue(true),
  generate: vi.fn().mockResolvedValue({
    text: "Mock response text",
    promptTokens: 50,
    completionTokens: 10,
    cacheReadTokens: 0,
    cacheWriteTokens: 50,
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
      cacheReadTokens: 0,
      cacheWriteTokens: 30,
      finishReason: "stop",
    });
  }),
  isModelLoaded: vi.fn().mockReturnValue(true),
  embed: vi.fn().mockResolvedValue({
    embeddings: [new Float32Array([1, 2, 3])],
    totalTokens: 2,
  }),
}));

// Import after mocking
import { LlamaCppLanguageModel } from "../../src/llama-cpp-language-model.js";
import { LlamaCppEmbeddingModel } from "../../src/llama-cpp-embedding-model.js";
import * as nativeBinding from "../../src/native-binding.js";
import { gemma4Reasoning } from "../../src/gemma4.js";
import { qwen3_6_dense } from "../../src/qwen3.6.js";

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

    it("returns cache usage statistics when native result includes them", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "Cached response",
        promptTokens: 50,
        completionTokens: 10,
        cacheReadTokens: 40,
        cacheWriteTokens: 10,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.usage.inputTokens.noCache).toBe(10);
      expect(result.usage.inputTokens.cacheRead).toBe(40);
      expect(result.usage.inputTokens.cacheWrite).toBe(10);
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
      expect(result.request!.body).toHaveProperty("enableThinking", true);
    });

    it("applies default maxTokens when not specified", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.request!.body).toHaveProperty("maxTokens", -1);
    });

    it("applies server sampling defaults when not specified", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.request!.body).toHaveProperty("temperature", 0.8);
      expect(result.request!.body).toHaveProperty("topP", 0.95);
      expect(result.request!.body).toHaveProperty("topK", 40);
    });

    it("passes custom generation options", async () => {
      await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 500,
        temperature: 0.5,
        topP: 0.8,
        topK: 30,
        seed: 1234,
        stopSequences: ["END"],
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          maxTokens: 500,
          temperature: 0.5,
          topP: 0.8,
          topK: 30,
          seed: 1234,
          stopSequences: ["END"],
        })
      );
    });

    it("rejects invalid seed values", async () => {
      await expect(
        model.doGenerate({
          prompt: testMessages,
          seed: -1,
        })
      ).rejects.toThrow("seed must be an integer between 0 and 4294967295");

      expect(nativeBinding.generate).not.toHaveBeenCalled();
    });

    it("rejects and cancels native generation when aborted", async () => {
      const controller = new AbortController();
      let markGenerationStarted!: () => void;
      let resolveGeneration!: (value: {
        text: string;
        promptTokens: number;
        completionTokens: number;
        finishReason: "stop";
      }) => void;
      const generationStarted = new Promise<void>((resolve) => {
        markGenerationStarted = resolve;
      });
      vi.mocked(nativeBinding.generate).mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveGeneration = resolve;
            markGenerationStarted();
          })
      );

      const promise = model.doGenerate({
        prompt: testMessages,
        abortSignal: controller.signal,
      });
      let settled = false;
      promise.catch(() => {
        settled = true;
      });

      await generationStarted;
      controller.abort();
      await Promise.resolve();

      expect(settled).toBe(false);
      const requestId = vi.mocked(nativeBinding.generate).mock.calls[0][1]
        .requestId;
      expect(requestId).toEqual(expect.any(String));
      expect(nativeBinding.cancelGeneration).toHaveBeenCalledWith(1, requestId);

      resolveGeneration({
        text: "Late response",
        promptTokens: 50,
        completionTokens: 10,
        finishReason: "stop",
      });

      await expect(promise).rejects.toMatchObject({ name: "AbortError" });
    });

    it("waits for active generation before unloading during dispose", async () => {
      let markGenerationStarted!: () => void;
      let resolveGeneration!: (value: {
        text: string;
        promptTokens: number;
        completionTokens: number;
        finishReason: "stop";
      }) => void;
      const generationStarted = new Promise<void>((resolve) => {
        markGenerationStarted = resolve;
      });
      vi.mocked(nativeBinding.generate).mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveGeneration = resolve;
            markGenerationStarted();
          })
      );

      const generationPromise = model.doGenerate({
        prompt: testMessages,
      });

      await generationStarted;
      const disposePromise = model.dispose();

      expect(nativeBinding.cancelGeneration).toHaveBeenCalledWith(
        1,
        vi.mocked(nativeBinding.generate).mock.calls[0][1].requestId
      );
      expect(nativeBinding.unloadModel).not.toHaveBeenCalled();

      resolveGeneration({
        text: "Late response",
        promptTokens: 50,
        completionTokens: 10,
        finishReason: "stop",
      });

      await disposePromise;
      await generationPromise;

      expect(nativeBinding.unloadModel).toHaveBeenCalledWith(1);
    });

    it("does not start native generation when already aborted", async () => {
      const controller = new AbortController();
      controller.abort();

      await expect(
        model.doGenerate({
          prompt: testMessages,
          abortSignal: controller.signal,
        })
      ).rejects.toMatchObject({ name: "AbortError" });

      expect(nativeBinding.generate).not.toHaveBeenCalled();
      expect(nativeBinding.cancelGeneration).not.toHaveBeenCalled();
    });

    it("enables native prompt cache when configured", async () => {
      const cachedModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        cache: { mode: "prefix" },
      });

      await cachedModel.doGenerate({
        prompt: testMessages,
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          promptCache: true,
        })
      );

      await cachedModel.dispose();
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

    it("retries lazy model loading after a failed load", async () => {
      vi.mocked(nativeBinding.loadModel).mockRejectedValueOnce(
        new Error("first load failed")
      );

      await expect(
        model.doGenerate({
          prompt: testMessages,
        })
      ).rejects.toThrow("first load failed");

      await expect(
        model.doGenerate({
          prompt: testMessages,
        })
      ).resolves.toHaveProperty("content");

      expect(nativeBinding.loadModel).toHaveBeenCalledTimes(2);
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

    it("returns think tags as text when reasoning is omitted", async () => {
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
          type: "text",
          text: "<think>Think first.</think>Final answer.",
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
      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          enableThinking: false,
        })
      );
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
      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          enableThinking: true,
        })
      );
    });

    it("passes mapped reasoning effort budget to native generation", async () => {
      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: {
          promptPrefix: "<|think|>\n",
          effortTokenBudget: {
            high: 4096,
          },
        },
      });

      await reasoningModel.doGenerate({
        prompt: testMessages,
        reasoning: "high",
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          messages: [
            { role: "system", content: "<|think|>\n" },
            { role: "user", content: "Hello, how are you?" },
          ],
          enableThinking: true,
          reasoningBudgetTokens: 4096,
          reasoningBudgetStart: "<think>",
          reasoningBudgetEnd: "</think>",
        })
      );

      await reasoningModel.dispose();
    });

    it("uses provider-default reasoning budget when call reasoning is omitted", async () => {
      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: {
          effortTokenBudget: {
            "provider-default": 1024,
          },
        },
      });

      await reasoningModel.doGenerate({
        prompt: testMessages,
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          reasoningBudgetTokens: 1024,
          reasoningBudgetStart: "<think>",
          reasoningBudgetEnd: "</think>",
          enableThinking: true,
        })
      );

      await reasoningModel.dispose();
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

    it("parses Qwen-style XML tool calls without leaking control or thinking text", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: `<|im_start|>assistant
<think>
The user asks whether ffmpeg is available. I should check with the sandbox shell.
</think>

<sandboxshell:cmd>which ffmpeg && ffmpeg -version</sandboxshell:cmd>`,
        promptTokens: 50,
        completionTokens: 42,
        finishReason: "stop",
      });

      const qwenModel = new LlamaCppLanguageModel({
        modelPath: "/test/qwen3.6.gguf",
        reasoning: qwen3_6_dense.reasoning,
      });

      const result = await qwenModel.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              { type: "text", text: "can i use ffmpeg in the sandbox" },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            name: "sandboxshell",
            inputSchema: {
              type: "object",
              properties: { cmd: { type: "string" } },
              required: ["cmd"],
            },
          },
        ],
      });

      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toMatchObject({
        type: "reasoning",
        providerMetadata: undefined,
      });
      expect(result.content[0]).toHaveProperty(
        "text",
        expect.stringContaining("I should check with the sandbox shell")
      );
      expect(result.content[1]).toEqual({
        type: "tool-call",
        toolCallId: expect.any(String),
        toolName: "sandboxshell",
        input: '{"cmd":"which ffmpeg && ffmpeg -version"}',
      });
      expect(result.finishReason.unified).toBe("tool-calls");

      await qwenModel.dispose();
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

    it("constrains JSON response format without schema to a JSON object", async () => {
      await model.doGenerate({
        prompt: testMessages,
        responseFormat: {
          type: "json",
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("root ::= object"),
        })
      );
    });

    it("supports OpenAI-compatible json_object response format from provider options", async () => {
      await model.doGenerate({
        prompt: testMessages,
        providerOptions: {
          "llama.cpp": {
            responseFormat: {
              type: "json_object",
            },
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("root ::= object"),
        })
      );
    });

    it("supports OpenAI-compatible json_object response format with schema", async () => {
      await model.doGenerate({
        prompt: testMessages,
        providerOptions: {
          "llama.cpp": {
            responseFormat: {
              type: "json_object",
              schema: {
                type: "object",
                properties: {
                  answer: { type: "string" },
                },
                required: ["answer"],
              },
            },
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("answer"),
        })
      );
    });

    it("supports OpenAI-compatible json_schema response format from provider options", async () => {
      await model.doGenerate({
        prompt: testMessages,
        providerOptions: {
          "llama.cpp": {
            responseFormat: {
              type: "json_schema",
              json_schema: {
                name: "answer",
                schema: {
                  type: "object",
                  properties: {
                    value: { type: "number" },
                  },
                  required: ["value"],
                },
              },
            },
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("value"),
        })
      );
    });

    it("supports server-style response_format provider option spelling", async () => {
      await model.doGenerate({
        prompt: testMessages,
        providerOptions: {
          "llama.cpp": {
            response_format: {
              type: "json_schema",
              json_schema: {
                schema: {
                  type: "object",
                  properties: {
                    label: { type: "string" },
                  },
                  required: ["label"],
                },
              },
            },
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("label"),
        })
      );
    });

    it("supports llama.cpp top-level json_schema provider option", async () => {
      await model.doGenerate({
        prompt: testMessages,
        providerOptions: {
          "llama.cpp": {
            json_schema: {
              type: "object",
              properties: {
                status: { enum: ["ok"] },
              },
              required: ["status"],
            },
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("status"),
        })
      );
    });

    it("rejects duplicate AI SDK and llama.cpp structured output settings", async () => {
      await expect(
        model.doGenerate({
          prompt: testMessages,
          responseFormat: {
            type: "json",
          },
          providerOptions: {
            "llama.cpp": {
              responseFormat: {
                type: "json_schema",
                json_schema: {
                  schema: { type: "object" },
                },
              },
            },
          },
        })
      ).rejects.toThrow(
        "AI SDK responseFormat and llama.cpp providerOptions.responseFormat cannot both request structured output."
      );
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

    it("passes seed to streaming generation", async () => {
      const result = await model.doStream({
        prompt: testMessages,
        seed: 5678,
      });

      await collectStreamParts(result.stream);

      expect(nativeBinding.generateStream).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          seed: 5678,
        }),
        expect.any(Function)
      );
    });

    it("passes mapped reasoning effort budget to native stream generation", async () => {
      const reasoningModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        reasoning: {
          openingMarker: "<|channel>thought\n",
          closingMarker: "<channel|>",
          promptPrefix: "<|think|>\n",
          effortTokenBudget: {
            medium: 2048,
          },
        },
      });

      const result = await reasoningModel.doStream({
        prompt: testMessages,
        reasoning: "medium",
      });

      await collectStreamParts(result.stream);

      expect(nativeBinding.generateStream).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          reasoningBudgetTokens: 2048,
          reasoningBudgetStart: "<|channel>thought\n",
          reasoningBudgetEnd: "<channel|>",
          enableThinking: true,
        }),
        expect.any(Function)
      );

      await reasoningModel.dispose();
    });

    it("constrains streamed JSON response format without schema to a JSON object", async () => {
      const result = await model.doStream({
        prompt: testMessages,
        responseFormat: {
          type: "json",
        },
      });

      await collectStreamParts(result.stream);

      expect(nativeBinding.generateStream).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: expect.stringContaining("root ::= object"),
        }),
        expect.any(Function)
      );
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

    it("finish part contains cache usage", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken("Hello");
          return Promise.resolve({
            text: "Hello",
            promptTokens: 30,
            completionTokens: 1,
            cacheReadTokens: 20,
            cacheWriteTokens: 10,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const finishPart = parts.find((p) => p.type === "finish");

      expect(finishPart?.usage?.inputTokens.cacheRead).toBe(20);
      expect(finishPart?.usage?.inputTokens.cacheWrite).toBe(10);
    });

    it("cancels native generation when stream reader is cancelled", async () => {
      let markGenerationStarted!: () => void;
      let resolveGeneration!: (value: {
        text: string;
        promptTokens: number;
        completionTokens: number;
        finishReason: "stop";
      }) => void;
      const generationStarted = new Promise<void>((resolve) => {
        markGenerationStarted = resolve;
      });
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveGeneration = resolve;
            markGenerationStarted();
          })
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const reader = stream.getReader();
      await reader.read();
      await generationStarted;
      await reader.cancel();

      const requestId = vi.mocked(nativeBinding.generateStream).mock.calls[0][1]
        .requestId;
      expect(requestId).toEqual(expect.any(String));
      expect(nativeBinding.cancelGeneration).toHaveBeenCalledWith(1, requestId);

      resolveGeneration({
        text: "",
        promptTokens: 30,
        completionTokens: 0,
        finishReason: "stop",
      });
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

    it("errors the stream and cancels native generation when aborted", async () => {
      const controller = new AbortController();
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (_handle, _opts, onToken) =>
          new Promise((resolve) => {
            onToken("Hello");
            setTimeout(() => {
              resolve({
                text: "Hello",
                promptTokens: 30,
                completionTokens: 1,
                finishReason: "stop",
              });
            }, 50);
          })
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        abortSignal: controller.signal,
      });

      const reader = stream.getReader();
      expect((await reader.read()).value?.type).toBe("stream-start");
      expect((await reader.read()).value?.type).toBe("text-start");
      expect((await reader.read()).value?.type).toBe("text-delta");

      controller.abort();

      await expect(reader.read()).rejects.toMatchObject({
        name: "AbortError",
      });
      const requestId = vi.mocked(nativeBinding.generateStream).mock.calls[0][1]
        .requestId;
      expect(requestId).toEqual(expect.any(String));
      expect(nativeBinding.cancelGeneration).toHaveBeenCalledWith(1, requestId);

      reader.releaseLock();
    });

    it("rejects before creating a stream when already aborted", async () => {
      const controller = new AbortController();
      controller.abort();

      await expect(
        model.doStream({
          prompt: testMessages,
          abortSignal: controller.signal,
        })
      ).rejects.toMatchObject({ name: "AbortError" });

      expect(nativeBinding.generateStream).not.toHaveBeenCalled();
      expect(nativeBinding.cancelGeneration).not.toHaveBeenCalled();
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
        logPrompts: true,
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
        logPrompts: true,
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

    it("allows maxOutputTokens -1 for server-compatible unlimited prediction", async () => {
      const result = await model.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        maxOutputTokens: -1,
      });

      expect(result.request!.body).toHaveProperty("maxTokens", -1);
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
        logPrompts: false,
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

    it("passes native tools when toolChoice is required", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '{"tool_calls":[{"id":"call_123","name":"get_weather","arguments":{"location":"Tokyo"}}]}',
        promptTokens: 100,
        completionTokens: 20,
        finishReason: "stop",
      });

      const result = await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
        toolChoice: { type: "required" },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: undefined,
          toolChoice: "required",
          tools: [
            {
              name: "get_weather",
              description: "Get the current weather",
              parametersJson:
                '{"type":"object","properties":{"location":{"type":"string"}}}',
            },
          ],
        })
      );
      expect(result.content[0].type).toBe("tool-call");
      expect(result.finishReason.unified).toBe("tool-calls");
    });

    it("uses only the selected tool for toolChoice tool", async () => {
      const tools = [
        ...testTools,
        {
          type: "function" as const,
          name: "search",
          description: "Search",
          inputSchema: {
            type: "object",
            properties: {
              query: { type: "string" },
            },
          },
        },
      ];

      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: '{"tool_calls":[{"id":"call_123","name":"search","arguments":{"query":"llama.cpp"}}]}',
        promptTokens: 100,
        completionTokens: 20,
        finishReason: "stop",
      });

      await model.doGenerate({
        prompt: testMessages,
        tools,
        toolChoice: { type: "tool", toolName: "search" },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          grammar: undefined,
          toolChoice: "required",
          tools: [
            {
              name: "search",
              description: "Search",
              parametersJson:
                '{"type":"object","properties":{"query":{"type":"string"}}}',
            },
          ],
        })
      );
      const generateCall = vi.mocked(nativeBinding.generate).mock.calls.at(-1);
      expect(generateCall?.[1].messages).toEqual([
        {
          role: "user",
          content: "What is the weather in Tokyo?",
        },
      ]);
    });

    it("forwards parallel tool calls from provider options", async () => {
      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "No tool needed.",
        promptTokens: 100,
        completionTokens: 20,
        finishReason: "stop",
      });

      await model.doGenerate({
        prompt: testMessages,
        tools: testTools,
        providerOptions: {
          "llama.cpp": {
            parallelToolCalls: true,
          },
        },
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          parallelToolCalls: true,
        })
      );
    });

    it("throws when toolChoice tool references an unknown function tool", async () => {
      await expect(
        model.doGenerate({
          prompt: testMessages,
          tools: testTools,
          toolChoice: { type: "tool", toolName: "search" },
        })
      ).rejects.toThrow("toolChoice references unknown function tool: search");

      expect(nativeBinding.generate).not.toHaveBeenCalled();
    });

    it("preserves generated tool-call JSON in later cached prompts", async () => {
      const cachedModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        cache: { mode: "prefix" },
      });
      const rawToolCall =
        '{"name": "get_weather", "arguments": {"location": "Tokyo"}}';

      vi.mocked(nativeBinding.generate)
        .mockResolvedValueOnce({
          text: rawToolCall,
          promptTokens: 100,
          completionTokens: 20,
          finishReason: "stop",
        })
        .mockResolvedValueOnce({
          text: "It is 72 degrees in Tokyo.",
          promptTokens: 130,
          completionTokens: 8,
          cacheReadTokens: 120,
          cacheWriteTokens: 10,
          finishReason: "stop",
        });

      const first = await cachedModel.doGenerate({
        prompt: testMessages,
        tools: testTools,
      });
      const toolCall = first.content.find((part) => part.type === "tool-call");

      expect(toolCall).toBeDefined();

      await cachedModel.doGenerate({
        prompt: [
          ...testMessages,
          {
            role: "assistant",
            content: [toolCall!],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: toolCall!.toolCallId,
                toolName: toolCall!.toolName,
                output: {
                  type: "json",
                  value: { location: "Tokyo", temperature: 72 },
                },
              },
            ],
          },
        ],
        tools: testTools,
      });

      expect(nativeBinding.generate).toHaveBeenNthCalledWith(
        2,
        expect.any(Number),
        expect.objectContaining({
          promptCache: true,
          messages: [
            { role: "user", content: "What is the weather in Tokyo?" },
            {
              role: "assistant",
              content: "",
              toolCalls: [
                {
                  id: toolCall!.toolCallId,
                  name: "get_weather",
                  arguments: '{"location":"Tokyo"}',
                },
              ],
            },
            {
              role: "tool",
              content: '{"location":"Tokyo","temperature":72}',
              toolName: "get_weather",
              toolCallId: toolCall!.toolCallId,
            },
          ],
        })
      );

      await cachedModel.dispose();
    });

    it("preserves generated tool-call JSON across later cached turns", async () => {
      const cachedModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        cache: { mode: "prefix" },
      });
      const rawToolCall =
        '{"name": "get_weather", "arguments": {"location": "Tokyo"}}';

      vi.mocked(nativeBinding.generate)
        .mockResolvedValueOnce({
          text: rawToolCall,
          promptTokens: 100,
          completionTokens: 20,
          finishReason: "stop",
        })
        .mockResolvedValueOnce({
          text: "It is 72 degrees in Tokyo.",
          promptTokens: 130,
          completionTokens: 8,
          cacheReadTokens: 120,
          cacheWriteTokens: 10,
          finishReason: "stop",
        })
        .mockResolvedValueOnce({
          text: "Yes, it is warm.",
          promptTokens: 150,
          completionTokens: 6,
          cacheReadTokens: 140,
          cacheWriteTokens: 10,
          finishReason: "stop",
        });

      const first = await cachedModel.doGenerate({
        prompt: testMessages,
        tools: testTools,
      });
      const toolCall = first.content.find((part) => part.type === "tool-call");

      expect(toolCall).toBeDefined();

      const afterToolPrompt: LanguageModelV4Message[] = [
        ...testMessages,
        {
          role: "assistant",
          content: [toolCall!],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: toolCall!.toolCallId,
              toolName: toolCall!.toolName,
              output: {
                type: "json",
                value: { location: "Tokyo", temperature: 72 },
              },
            },
          ],
        },
      ];

      const second = await cachedModel.doGenerate({
        prompt: afterToolPrompt,
        tools: testTools,
      });
      const secondText = second.content.find((part) => part.type === "text");

      expect(secondText).toBeDefined();

      await cachedModel.doGenerate({
        prompt: [
          ...afterToolPrompt,
          {
            role: "assistant",
            content: [secondText!],
          },
          {
            role: "user",
            content: [{ type: "text", text: "Is that warm?" }],
          },
        ],
        tools: testTools,
      });

      expect(nativeBinding.generate).toHaveBeenNthCalledWith(
        3,
        expect.any(Number),
        expect.objectContaining({
          promptCache: true,
          messages: expect.arrayContaining([
            {
              role: "assistant",
              content: "",
              toolCalls: [
                {
                  id: toolCall!.toolCallId,
                  name: "get_weather",
                  arguments: '{"location":"Tokyo"}',
                },
              ],
            },
            {
              role: "tool",
              content: '{"location":"Tokyo","temperature":72}',
              toolName: "get_weather",
              toolCallId: toolCall!.toolCallId,
            },
          ]),
        })
      );

      await cachedModel.dispose();
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

    it("streams tool input deltas when output is a tool call", async () => {
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

      const textParts = parts.filter(
        (p) =>
          p.type === "text-start" ||
          p.type === "text-delta" ||
          p.type === "text-end"
      );
      expect(textParts).toHaveLength(0);

      const toolInputStart = parts.find((p) => p.type === "tool-input-start");
      const toolInputDeltas = parts.filter(
        (p) => p.type === "tool-input-delta"
      );
      const toolInputEnd = parts.find((p) => p.type === "tool-input-end");
      const toolCallPart = parts.find((p) => p.type === "tool-call");

      expect(toolInputStart).toBeDefined();
      expect(toolInputStart?.toolName).toBe("weather");
      expect(toolInputDeltas.map((p) => p.delta).join("")).toBe(
        '{"location": "Tokyo"}'
      );
      expect(toolInputEnd).toBeDefined();
      expect(toolCallPart).toBeDefined();
      expect(toolCallPart?.toolName).toBe("weather");
      expect(toolCallPart?.input).toBe('{"location": "Tokyo"}');
    });

    it("emits streamed tool input before the final tool-call part", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken('{"name": "weather", "arguments": ');
          onToken('{"location": ');
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
      const startIndex = parts.findIndex((p) => p.type === "tool-input-start");
      const deltaIndex = parts.findIndex((p) => p.type === "tool-input-delta");
      const endIndex = parts.findIndex((p) => p.type === "tool-input-end");
      const callIndex = parts.findIndex((p) => p.type === "tool-call");

      expect(startIndex).toBeGreaterThan(-1);
      expect(deltaIndex).toBeGreaterThan(startIndex);
      expect(endIndex).toBeGreaterThan(deltaIndex);
      expect(callIndex).toBeGreaterThan(endIndex);
    });

    it("emits tool input parts for native parsed tool calls", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(() =>
        Promise.resolve({
          text: "",
          promptTokens: 50,
          completionTokens: 15,
          finishReason: "stop",
          toolCalls: [
            {
              id: "call_native",
              name: "weather",
              arguments: '{"location":"Tokyo"}',
            },
          ],
        })
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);
      const toolInputStart = parts.find((p) => p.type === "tool-input-start");
      const toolInputDelta = parts.find((p) => p.type === "tool-input-delta");
      const toolInputEnd = parts.find((p) => p.type === "tool-input-end");
      const toolCallPart = parts.find((p) => p.type === "tool-call");

      expect(toolInputStart).toEqual({
        type: "tool-input-start",
        id: "call_native",
        toolName: "weather",
      });
      expect(toolInputDelta).toEqual({
        type: "tool-input-delta",
        id: "call_native",
        delta: '{"location":"Tokyo"}',
      });
      expect(toolInputEnd).toEqual({
        type: "tool-input-end",
        id: "call_native",
      });
      expect(toolCallPart).toEqual({
        type: "tool-call",
        toolCallId: "call_native",
        toolName: "weather",
        input: '{"location":"Tokyo"}',
      });
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

    it("falls back to text when JSON-looking streamed output is not a tool call", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken("{");
          onToken('"answer"');
          onToken(': "Use a JSON object here."');
          onToken("}");
          return Promise.resolve({
            text: '{"answer": "Use a JSON object here."}',
            promptTokens: 50,
            completionTokens: 8,
            finishReason: "stop",
          });
        }
      );

      const { stream } = await model.doStream({
        prompt: testMessages,
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      const toolCallPart = parts.find((p) => p.type === "tool-call");
      const finishPart = parts.find((p) => p.type === "finish");

      expect(textDeltas.map((p) => p.delta).join("")).toBe(
        '{"answer": "Use a JSON object here."}'
      );
      expect(toolCallPart).toBeUndefined();
      expect(finishPart?.finishReason?.unified).toBe("stop");
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

      // Active tools buffer regular text until native parsing completes.
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      expect(textDeltas).toHaveLength(1);
      expect(textDeltas[0]?.delta).toBe("The weather is nice");

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

      const textDeltas = parts.filter((p) => p.type === "text-delta");
      expect(textDeltas).toHaveLength(0);

      const toolInputStart = parts.find((p) => p.type === "tool-input-start");
      expect(toolInputStart).toBeDefined();

      const toolCallPart = parts.find((p) => p.type === "tool-call");
      expect(toolCallPart).toBeDefined();
      expect(toolCallPart?.toolName).toBe("weather");
    });

    it("uses visible text for streamed tool call parsing after reasoning", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken("<|channel>thought\nNeed weather.<channel|>");
          onToken('{"name":"weather","arguments":{"location":"Tokyo"}}');
          return Promise.resolve({
            text: '<|channel>thought\nNeed weather.<channel|>{"name":"weather","arguments":{"location":"Tokyo"}}',
            promptTokens: 50,
            completionTokens: 20,
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
        tools: testTools,
      });

      const parts = await collectStreamParts(stream);
      const reasoningDeltas = parts.filter((p) => p.type === "reasoning-delta");
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      const toolCallPart = parts.find((p) => p.type === "tool-call");
      const finishPart = parts.find((p) => p.type === "finish");

      expect(reasoningDeltas.map((p) => p.delta).join("")).toBe(
        "Need weather."
      );
      expect(textDeltas).toHaveLength(0);
      expect(toolCallPart).toEqual(
        expect.objectContaining({
          toolName: "weather",
          input: '{"location":"Tokyo"}',
        })
      );
      expect(finishPart?.finishReason.unified).toBe("tool-calls");

      await reasoningModel.dispose();
    });

    it("preserves streamed tool-call JSON in later cached prompts", async () => {
      const cachedModel = new LlamaCppLanguageModel({
        modelPath: "/test/model.gguf",
        cache: { mode: "prefix" },
      });
      const rawToolCall =
        '{"name": "weather", "arguments": {"location": "Tokyo"}}';

      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(
        (handle, opts, onToken) => {
          onToken(rawToolCall);
          return Promise.resolve({
            text: rawToolCall,
            promptTokens: 50,
            completionTokens: 15,
            finishReason: "stop",
          });
        }
      );

      const first = await cachedModel.doStream({
        prompt: testMessages,
        tools: testTools,
      });
      const firstParts = await collectStreamParts(first.stream);
      const toolCall = firstParts.find((part) => part.type === "tool-call");

      expect(toolCall).toBeDefined();

      vi.mocked(nativeBinding.generate).mockResolvedValueOnce({
        text: "It is sunny in Tokyo.",
        promptTokens: 80,
        completionTokens: 6,
        cacheReadTokens: 70,
        cacheWriteTokens: 10,
        finishReason: "stop",
      });

      await cachedModel.doGenerate({
        prompt: [
          ...testMessages,
          {
            role: "assistant",
            content: [toolCall],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                output: {
                  type: "json",
                  value: { location: "Tokyo", condition: "sunny" },
                },
              },
            ],
          },
        ],
        tools: testTools,
      });

      expect(nativeBinding.generate).toHaveBeenLastCalledWith(
        expect.any(Number),
        expect.objectContaining({
          promptCache: true,
          messages: expect.arrayContaining([
            {
              role: "assistant",
              content: "",
              toolCalls: [
                {
                  id: toolCall.toolCallId,
                  name: "weather",
                  arguments: '{"location": "Tokyo"}',
                },
              ],
            },
            {
              role: "tool",
              content: '{"location":"Tokyo","condition":"sunny"}',
              toolName: "weather",
              toolCallId: toolCall.toolCallId,
            },
          ]),
        })
      );

      await cachedModel.dispose();
    });
  });
});

describe("LlamaCppEmbeddingModel Integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("retries lazy model loading after a failed load", async () => {
    const model = new LlamaCppEmbeddingModel({
      modelPath: "/test/embedding-model.gguf",
    });
    vi.mocked(nativeBinding.loadModel).mockRejectedValueOnce(
      new Error("embedding load failed")
    );

    await expect(
      model.doEmbed({
        values: ["hello"],
      })
    ).rejects.toThrow("embedding load failed");

    await expect(
      model.doEmbed({
        values: ["hello"],
      })
    ).resolves.toMatchObject({
      embeddings: [[1, 2, 3]],
      usage: { tokens: 2 },
    });

    expect(nativeBinding.loadModel).toHaveBeenCalledTimes(2);

    await model.dispose();
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
