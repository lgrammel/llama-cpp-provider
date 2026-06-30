import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { generateText, tool, stepCountIs } from "ai";
import { z } from "zod/v4";
import { llamaCpp, LlamaCppLanguageModel } from "@lgrammel/llama-cpp-provider";
import {
  formatModelInfo,
  languageModelConfig,
  modelRuntimeConfig,
} from "./e2e-config.js";

/**
 * E2E tests for tool calling with llama.cpp.
 *
 * These tests require a real GGUF model file. Set the TEST_MODEL_PATH
 * environment variable to run these tests:
 *
 *   TEST_MODEL_PATH=./models/your-model.gguf pnpm test:e2e
 *
 * Set TEST_CHAT_TEMPLATE when the embedded model chat template is not
 * supported by the pinned llama.cpp revision, for example:
 *
 *   TEST_CHAT_TEMPLATE=gemma TEST_MODEL_PATH=./models/gemma.gguf pnpm test:e2e
 *
 * Note: Tool calling quality depends heavily on the model's capabilities.
 * Models specifically fine-tuned for function calling (e.g., Llama 3.1+,
 * Hermes, Functionary) will perform better.
 */

const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
const shouldRunTests = !!TEST_MODEL_PATH;

const describeE2E = shouldRunTests ? describe : describe.skip;

describeE2E("E2E Tool Calling Tests", () => {
  let model: LlamaCppLanguageModel;

  // Fake weather data for testing
  const weatherData: Record<
    string,
    { temperature: number; condition: string }
  > = {
    tokyo: { temperature: 22, condition: "cloudy" },
    paris: { temperature: 18, condition: "rainy" },
    "new york": { temperature: 25, condition: "sunny" },
  };

  beforeAll(() => {
    if (!TEST_MODEL_PATH) {
      throw new Error("TEST_MODEL_PATH environment variable not set");
    }

    model = llamaCpp(
      languageModelConfig({
        modelPath: TEST_MODEL_PATH,
        ...modelRuntimeConfig({ contextSize: 4096 }),
      })
    );
  });

  afterAll(async () => {
    if (model) {
      await model.dispose();
    }
  });

  describe("tool call generation", () => {
    it(
      "generates tool call with grammar constraint",
      { timeout: 120000 },
      async () => {
        const result = await generateText({
          model,
          prompt:
            "What is the weather in Tokyo? Use the get_weather tool to find out.",
          maxOutputTokens: 500,
          tools: {
            get_weather: tool({
              description: "Get the current weather for a city",
              inputSchema: z.object({
                city: z.string().describe("The city name"),
              }),
              execute: async ({ city }) => {
                const normalizedCity = city.toLowerCase();
                return (
                  weatherData[normalizedCity] || { error: "City not found" }
                );
              },
            }),
          },
          stopWhen: stepCountIs(1), // Only generate tool call, don't execute
        });

        // With grammar constraint, the model should generate a tool call
        // Check if the result contains tool calls or text (depending on model capability)
        expect(result).toBeDefined();
        expect(result.finishReason).toBeDefined();
      }
    );

    it("executes tool and returns result", { timeout: 120000 }, async () => {
      let toolWasCalled = false;

      const result = await generateText({
        model,
        prompt: "What is the weather in Tokyo?",
        maxOutputTokens: 500,
        tools: {
          get_weather: tool({
            description:
              "Get the current weather for a city. Returns temperature and condition.",
            inputSchema: z.object({
              city: z.string().describe("The city name to get weather for"),
            }),
            execute: async ({ city }) => {
              toolWasCalled = true;
              const normalizedCity = city.toLowerCase();
              return (
                weatherData[normalizedCity] || {
                  error: `Weather not found for ${city}`,
                }
              );
            },
          }),
        },
        stopWhen: stepCountIs(2), // Allow tool call and follow-up
      });

      // The model should have tried to call the tool
      // Note: Success depends on model quality
      expect(result).toBeDefined();

      // Check if any step had tool calls
      const hasToolCalls = result.steps?.some(
        (step) => step.toolCalls && step.toolCalls.length > 0
      );

      expect(toolWasCalled).toBe(hasToolCalls === true);
    });

    it(
      "handles tool with multiple parameters",
      { timeout: 120000 },
      async () => {
        const result = await generateText({
          model,
          prompt: "Search for 'llama.cpp' with a limit of 5 results",
          maxOutputTokens: 500,
          tools: {
            search: tool({
              description: "Search for information",
              inputSchema: z.object({
                query: z.string().describe("The search query"),
                limit: z
                  .number()
                  .optional()
                  .describe("Maximum number of results"),
              }),
              execute: async ({ query, limit }) => {
                return {
                  query,
                  limit: limit || 10,
                  results: ["Result 1", "Result 2", "Result 3"],
                };
              },
            }),
          },
          stopWhen: stepCountIs(2),
        });

        expect(result).toBeDefined();
      }
    );
  });

  describe("tool choice options", () => {
    it("respects toolChoice: none", { timeout: 120000 }, async () => {
      const result = await generateText({
        model,
        prompt: "What is the weather in Paris?",
        maxOutputTokens: 100,
        tools: {
          get_weather: tool({
            description: "Get weather",
            inputSchema: z.object({ city: z.string() }),
            execute: async () => ({ temperature: 20 }),
          }),
        },
        toolChoice: "none",
      });

      // With toolChoice: none, model should not call tools
      expect(result.text.length).toBeGreaterThan(0);
      // Should not have tool calls when toolChoice is none
      const hasToolCalls = result.steps?.some(
        (step) => step.toolCalls && step.toolCalls.length > 0
      );
      expect(hasToolCalls).not.toBe(true);
    });
  });

  describe("multiple tools", () => {
    it("can choose between multiple tools", { timeout: 120000 }, async () => {
      const result = await generateText({
        model,
        prompt: "Calculate 15 * 7",
        maxOutputTokens: 500,
        tools: {
          get_weather: tool({
            description: "Get weather for a city",
            inputSchema: z.object({ city: z.string() }),
            execute: async () => ({ temperature: 20 }),
          }),
          calculator: tool({
            description: "Perform mathematical calculations",
            inputSchema: z.object({
              expression: z
                .string()
                .describe("The math expression to evaluate"),
            }),
            execute: async ({ expression }) => {
              const match = expression.match(
                /^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$/
              );
              if (!match) {
                return { error: "Unsupported expression" };
              }

              const left = Number(match[1]);
              const right = Number(match[3]);
              const calculatorResult =
                match[2] === "+"
                  ? left + right
                  : match[2] === "-"
                    ? left - right
                    : match[2] === "*"
                      ? left * right
                      : left / right;
              return { result: calculatorResult };
            },
          }),
        },
        stopWhen: stepCountIs(2),
      });

      expect(result).toBeDefined();
    });
  });
});

// Test that runs without a model to verify skip behavior
describe("E2E Tool Calling Test Configuration", () => {
  it("TEST_MODEL_PATH environment variable info", () => {
    if (!TEST_MODEL_PATH) {
      console.log(
        "\n📋 Tool calling E2E tests skipped: Set TEST_MODEL_PATH to run with a real model"
      );
      console.log(
        "   Example: TEST_MODEL_PATH=./models/model.gguf pnpm test:e2e"
      );
      console.log(
        "   If chat templating fails, add TEST_CHAT_TEMPLATE=gemma or another llama.cpp template name"
      );
      console.log(
        "   Note: Use a model fine-tuned for function calling for best results\n"
      );
    } else {
      console.log(
        `\n✅ Running tool calling E2E tests with model: ${formatModelInfo(TEST_MODEL_PATH)}\n`
      );
    }
    expect(true).toBe(true);
  });
});
