import { describe, it, expect } from "vitest";
import {
  generateToolCallGrammar,
  parseToolCalls,
  buildToolSystemPrompt,
  convertMessages,
} from "../../src/llama-cpp-language-model.js";
import type {
  LanguageModelV4FunctionTool,
  LanguageModelV4Message,
} from "@ai-sdk/provider";

describe("Tool Calling", () => {
  describe("generateToolCallGrammar", () => {
    it("generates grammar for a single tool with simple parameters", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "get_weather",
          description: "Get the current weather",
          inputSchema: {
            type: "object",
            properties: {
              city: { type: "string" },
            },
            required: ["city"],
          },
        },
      ];

      const grammar = generateToolCallGrammar(tools);

      // Grammar should contain tool call structure rules
      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("tool-calls-kv");
      expect(grammar).toContain("tool-call");
      expect(grammar).toContain("get_weather");
      expect(grammar).toContain("space");
    });

    it("generates grammar for multiple tools", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "get_weather",
          inputSchema: {
            type: "object",
            properties: {
              city: { type: "string" },
            },
          },
        },
        {
          type: "function",
          name: "search",
          inputSchema: {
            type: "object",
            properties: {
              query: { type: "string" },
            },
          },
        },
      ];

      const grammar = generateToolCallGrammar(tools);

      // Should contain both tool names
      expect(grammar).toContain("get_weather");
      expect(grammar).toContain("search");
      // Should have per-tool call alternatives that bind name and args together.
      expect(grammar).toContain("get_weather-call");
      expect(grammar).toContain("search-call");
    });

    it("handles tools with complex parameters", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "create_event",
          inputSchema: {
            type: "object",
            properties: {
              title: { type: "string" },
              date: { type: "string" },
              attendees: {
                type: "array",
                items: { type: "string" },
              },
              priority: {
                type: "integer",
                minimum: 1,
                maximum: 5,
              },
            },
            required: ["title", "date"],
          },
        },
      ];

      const grammar = generateToolCallGrammar(tools);

      expect(grammar).toContain("create_event");
      expect(grammar).toContain("root ::=");
    });

    it("handles tools with no description", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "simple_tool",
          inputSchema: {
            type: "object",
            properties: {},
          },
        },
      ];

      const grammar = generateToolCallGrammar(tools);

      expect(grammar).toContain("simple_tool");
    });
  });

  describe("parseToolCalls", () => {
    it("parses valid tool call JSON", () => {
      const json = JSON.stringify({
        tool_calls: [
          {
            id: "call_123",
            name: "get_weather",
            arguments: { city: "Tokyo" },
          },
        ],
      });

      const result = parseToolCalls(json);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0].id).toBe("call_123");
      expect(result![0].name).toBe("get_weather");
      expect(result![0].arguments).toEqual({ city: "Tokyo" });
    });

    it("parses multiple tool calls", () => {
      const json = JSON.stringify({
        tool_calls: [
          {
            id: "call_1",
            name: "get_weather",
            arguments: { city: "Tokyo" },
          },
          {
            id: "call_2",
            name: "get_weather",
            arguments: { city: "Paris" },
          },
        ],
      });

      const result = parseToolCalls(json);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
      expect(result![0].name).toBe("get_weather");
      expect(result![0].arguments).toEqual({ city: "Tokyo" });
      expect(result![1].name).toBe("get_weather");
      expect(result![1].arguments).toEqual({ city: "Paris" });
    });

    it("returns null for non-JSON text", () => {
      const text = "I don't know the weather in Tokyo.";

      const result = parseToolCalls(text);

      expect(result).toBeNull();
    });

    it("returns null for JSON without tool_calls", () => {
      const json = JSON.stringify({ message: "Hello" });

      const result = parseToolCalls(json);

      expect(result).toBeNull();
    });

    it("returns null for empty tool_calls array", () => {
      const json = JSON.stringify({ tool_calls: [] });

      const result = parseToolCalls(json);

      expect(result).toBeNull();
    });

    it("returns null for invalid tool call structure", () => {
      const json = JSON.stringify({
        tool_calls: [{ invalid: "structure" }],
      });

      const result = parseToolCalls(json);

      expect(result).toBeNull();
    });

    it("handles whitespace around JSON", () => {
      const json = `
        {
          "tool_calls": [
            {
              "id": "call_456",
              "name": "search",
              "arguments": { "query": "test" }
            }
          ]
        }
      `;

      const result = parseToolCalls(json);

      expect(result).not.toBeNull();
      expect(result![0].name).toBe("search");
    });

    it("handles tool calls with complex arguments", () => {
      const json = JSON.stringify({
        tool_calls: [
          {
            id: "call_complex",
            name: "create_event",
            arguments: {
              title: "Meeting",
              date: "2024-01-15",
              attendees: ["Alice", "Bob"],
              metadata: { priority: 1, tags: ["important"] },
            },
          },
        ],
      });

      const result = parseToolCalls(json);

      expect(result).not.toBeNull();
      expect(result![0].arguments).toEqual({
        title: "Meeting",
        date: "2024-01-15",
        attendees: ["Alice", "Bob"],
        metadata: { priority: 1, tags: ["important"] },
      });
    });

    it("parses Qwen-style namespaced XML tool calls inside generated text", () => {
      const text = `<|im_start|>assistant
<think>
Need to check the sandbox.
</think>

<sandboxshell:cmd>which ffmpeg && ffmpeg -version</sandboxshell:cmd>`;

      const result = parseToolCalls(text);

      expect(result).not.toBeNull();
      expect(result).toHaveLength(1);
      expect(result![0].name).toBe("sandboxshell");
      expect(result![0].arguments).toEqual({
        cmd: "which ffmpeg && ffmpeg -version",
      });
    });
  });

  describe("buildToolSystemPrompt", () => {
    it("builds system prompt with single tool", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "get_weather",
          description: "Get current weather for a city",
          inputSchema: {
            type: "object",
            properties: {
              city: { type: "string" },
            },
          },
        },
      ];

      const prompt = buildToolSystemPrompt(tools);

      expect(prompt).toContain("get_weather");
      expect(prompt).toContain("Get current weather for a city");
      expect(prompt).toContain('"name"');
      expect(prompt).toContain('"arguments"');
    });

    it("builds system prompt with multiple tools", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "get_weather",
          description: "Get weather",
          inputSchema: { type: "object", properties: {} },
        },
        {
          type: "function",
          name: "search",
          description: "Search the web",
          inputSchema: { type: "object", properties: {} },
        },
      ];

      const prompt = buildToolSystemPrompt(tools);

      expect(prompt).toContain("get_weather");
      expect(prompt).toContain("search");
      expect(prompt).toContain("Get weather");
      expect(prompt).toContain("Search the web");
    });

    it("handles tools without description", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "unnamed_tool",
          inputSchema: { type: "object", properties: {} },
        },
      ];

      const prompt = buildToolSystemPrompt(tools);

      expect(prompt).toContain("unnamed_tool");
      expect(prompt).toContain("No description");
    });

    it("includes parameter schema in prompt", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "get_weather",
          description: "Get weather",
          inputSchema: {
            type: "object",
            properties: {
              city: { type: "string", description: "City name" },
              unit: { type: "string", enum: ["celsius", "fahrenheit"] },
            },
            required: ["city"],
          },
        },
      ];

      const prompt = buildToolSystemPrompt(tools);

      expect(prompt).toContain("city");
      expect(prompt).toContain("string");
    });

    it("adds required-tool instruction for forced tool choice", () => {
      const tools: LanguageModelV4FunctionTool[] = [
        {
          type: "function",
          name: "get_weather",
          inputSchema: { type: "object", properties: {} },
        },
      ];

      const prompt = buildToolSystemPrompt(tools, {
        type: "tool",
        toolName: "get_weather",
      });

      expect(prompt).toContain('You must call the "get_weather" tool');
      expect(prompt).toContain("Do not answer with normal text");
    });
  });

  describe("convertMessages with tools", () => {
    it("does not inject a generic tool system prompt", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "What's the weather?" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        {
          role: "user",
          content: "What's the weather?",
        },
      ]);
    });

    it("does not add tool prompt when no tools provided", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "user", content: [{ type: "text", text: "Hello" }] },
      ];

      const result = convertMessages(messages);

      // Should only have the user message
      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("user");
    });

    it("handles assistant messages with tool calls", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "get_weather",
              input: JSON.stringify({ city: "Tokyo" }),
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("assistant");
      expect(result[0].content).toBe("");
      expect(result[0].toolCalls).toEqual([
        {
          id: "call_123",
          name: "get_weather",
          arguments: '{"city":"Tokyo"}',
        },
      ]);
    });

    it("handles mixed text and tool call content", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check the weather." },
            {
              type: "tool-call",
              toolCallId: "call_456",
              toolName: "get_weather",
              input: JSON.stringify({ city: "Paris" }),
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "assistant",
        content: "Let me check the weather.",
        toolCalls: [
          {
            id: "call_456",
            name: "get_weather",
            arguments: '{"city":"Paris"}',
          },
        ],
      });
    });

    it("handles tool result messages", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_789",
              toolName: "get_weather",
              output: {
                type: "json",
                value: { temperature: 25, condition: "sunny" },
              },
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "tool",
        content: '{"temperature":25,"condition":"sunny"}',
        toolName: "get_weather",
        toolCallId: "call_789",
      });
    });

    it("handles complete tool calling conversation", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "What's the weather in Tokyo?" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_abc",
              toolName: "get_weather",
              input: JSON.stringify({ city: "Tokyo" }),
            },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_abc",
              toolName: "get_weather",
              output: {
                type: "json",
                value: { temperature: 22, condition: "cloudy" },
              },
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toHaveLength(3);
      expect(result[0].role).toBe("user");
      expect(result[1]).toEqual({
        role: "assistant",
        content: "",
        toolCalls: [
          {
            id: "call_abc",
            name: "get_weather",
            arguments: '{"city":"Tokyo"}',
          },
        ],
      });
      expect(result[2]).toEqual({
        role: "tool",
        content: '{"temperature":22,"condition":"cloudy"}',
        toolName: "get_weather",
        toolCallId: "call_abc",
      });
    });
  });
});
