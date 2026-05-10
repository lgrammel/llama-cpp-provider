import { describe, it, expect } from "vitest";
import { convertMessages } from "../../src/llama-cpp-language-model.js";
import type { LanguageModelV4Message } from "@ai-sdk/provider";

describe("convertMessages", () => {
  describe("system messages", () => {
    it("converts system message to role/content format", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "system", content: "You are a helpful assistant." },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "system", content: "You are a helpful assistant." },
      ]);
    });
  });

  describe("user messages", () => {
    it("converts user message with text content", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello, how are you?" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "Hello, how are you?" },
      ]);
    });

    it("concatenates multiple text parts in user message", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "First part. " },
            { type: "text", text: "Second part." },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "First part. Second part." },
      ]);
    });

    it("ignores non-text parts in user message", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "Hello" },
            // File parts are not supported, should be ignored
            {
              type: "file",
              url: "data:image/png;base64,abc",
              mediaType: "image/png",
            } as any,
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([{ role: "user", content: "Hello" }]);
    });
  });

  describe("assistant messages", () => {
    it("converts assistant message to role/content format", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "assistant",
          content: [{ type: "text", text: "I am doing well, thank you!" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "assistant", content: "I am doing well, thank you!" },
      ]);
    });

    it("concatenates multiple text parts in assistant message", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Part one. " },
            { type: "text", text: "Part two." },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "assistant", content: "Part one. Part two." },
      ]);
    });

    it("does not include assistant reasoning in conversation history", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "assistant",
          content: [
            { type: "reasoning", text: "Private thinking." },
            { type: "text", text: "Final answer." },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "assistant", content: "Final answer." },
      ]);
    });
  });

  describe("reasoning prompt prefix", () => {
    it("adds the prefix to the first system message", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: [{ type: "text", text: "Hi" }] },
      ];

      const result = convertMessages(messages, undefined, {
        promptPrefix: "<|think|>\n",
      });

      expect(result).toEqual([
        { role: "system", content: "<|think|>\nYou are helpful." },
        { role: "user", content: "Hi" },
      ]);
    });

    it("adds a system message when no system message exists", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "user", content: [{ type: "text", text: "Hi" }] },
      ];

      const result = convertMessages(messages, undefined, {
        promptPrefix: "<|think|>\n",
      });

      expect(result).toEqual([
        { role: "system", content: "<|think|>\n" },
        { role: "user", content: "Hi" },
      ]);
    });
  });

  describe("tool messages", () => {
    it("converts tool result message to user message format", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_123",
              toolName: "get_weather",
              output: {
                type: "json",
                value: { temperature: 72 },
              },
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      // Tool results are converted to user messages
      expect(result).toEqual([
        {
          role: "user",
          content:
            'Tool "get_weather" (id: call_123) returned:\n{"temperature":72}',
        },
      ]);
    });

    it("handles text output from tool result", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_456",
              toolName: "search",
              output: {
                type: "text",
                value: "Search result: found 10 items",
              },
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        {
          role: "user",
          content:
            'Tool "search" (id: call_456) returned:\nSearch result: found 10 items',
        },
      ]);
    });

    it("handles error output from tool result", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_789",
              toolName: "api_call",
              output: {
                type: "error-text",
                value: "Connection refused",
              },
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        {
          role: "user",
          content:
            'Tool "api_call" (id: call_789) returned:\nError: Connection refused',
        },
      ]);
    });

    it("handles execution-denied output from tool result", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_abc",
              toolName: "dangerous_action",
              output: {
                type: "execution-denied",
                reason: "User denied permission",
              },
            },
          ],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        {
          role: "user",
          content:
            'Tool "dangerous_action" (id: call_abc) returned:\nExecution denied: User denied permission',
        },
      ]);
    });
  });

  describe("multi-turn conversations", () => {
    it("converts a complete conversation with system, user, and assistant", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: [{ type: "text", text: "Hi!" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "Hello there!" }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "How are you?" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi!" },
        { role: "assistant", content: "Hello there!" },
        { role: "user", content: "How are you?" },
      ]);
    });

    it("handles multiple user-assistant exchanges", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "user", content: [{ type: "text", text: "Question 1" }] },
        { role: "assistant", content: [{ type: "text", text: "Answer 1" }] },
        { role: "user", content: [{ type: "text", text: "Question 2" }] },
        { role: "assistant", content: [{ type: "text", text: "Answer 2" }] },
        { role: "user", content: [{ type: "text", text: "Question 3" }] },
      ];

      const result = convertMessages(messages);

      expect(result).toHaveLength(5);
      expect(result[0]).toEqual({ role: "user", content: "Question 1" });
      expect(result[1]).toEqual({ role: "assistant", content: "Answer 1" });
      expect(result[2]).toEqual({ role: "user", content: "Question 2" });
      expect(result[3]).toEqual({ role: "assistant", content: "Answer 2" });
      expect(result[4]).toEqual({ role: "user", content: "Question 3" });
    });
  });

  describe("edge cases", () => {
    it("handles empty messages array", () => {
      const messages: LanguageModelV4Message[] = [];

      const result = convertMessages(messages);

      expect(result).toEqual([]);
    });

    it("handles empty text content", () => {
      const messages: LanguageModelV4Message[] = [
        { role: "user", content: [{ type: "text", text: "" }] },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([{ role: "user", content: "" }]);
    });

    it("preserves newlines in content", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Line 1\nLine 2\nLine 3" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "Line 1\nLine 2\nLine 3" },
      ]);
    });

    it("preserves special characters in content", () => {
      const messages: LanguageModelV4Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Special chars: <>&\"'`${}[]" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "Special chars: <>&\"'`${}[]" },
      ]);
    });
  });
});
