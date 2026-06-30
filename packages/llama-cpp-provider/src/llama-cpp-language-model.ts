import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4FunctionTool,
  LanguageModelV4GenerateResult,
  LanguageModelV4Message,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4ToolChoice,
  LanguageModelV4Usage,
  SharedV4Warning,
} from "@ai-sdk/provider";
import { stat } from "node:fs/promises";

import {
  loadModel,
  unloadModel,
  generate,
  generateStream,
  cancelGeneration,
  isModelLoaded,
  type LoadModelOptions,
  type GenerateOptions,
  type ChatMessage,
  type NativeToolCall,
  type ToolDefinition,
} from "./native-binding.js";

import type { JSONSchema7 } from "@ai-sdk/provider";
import {
  convertJsonSchemaToGrammar,
  SchemaConverter,
} from "./json-schema-to-grammar.js";
import {
  thinkTagsReasoning,
  type LlamaCppCacheConfig,
  type LlamaCppMemorySafetyConfig,
  type LlamaCppModelMemoryInfo,
  type LlamaCppReasoningEffort,
  type LlamaCppReasoningConfig,
} from "./llama-cpp-provider-config.js";
import { checkMemorySafety } from "./memory-estimation.js";

export interface LlamaCppModelConfig {
  modelPath: string;
  /**
   * Path to the multimodal projector GGUF file.
   *
   * Required when image/file parts are included in user messages.
   */
  mmprojPath?: string;
  /**
   * Maximum context size.
   *
   * This setting is highly dependent on the model and the memory available on
   * the machine. Higher values can consume significant memory and may freeze
   * the machine if set too high. Monitor system memory when increasing this
   * value.
   */
  contextSize?: number;
  memorySafety?: LlamaCppMemorySafetyConfig;
  memory?: LlamaCppModelMemoryInfo;
  gpuLayers?: number;
  threads?: number;
  /**
   * Enable verbose debug output from llama.cpp.
   * Default: false
   */
  debug?: boolean;
  /**
   * Print the final chat-template-rendered prompt sent to llama.cpp to stderr.
   *
   * This may include private user data. Intended for local debugging only.
   * Default: false
   */
  logPrompts?: boolean;
  /**
   * Chat template to use for formatting messages.
   * - "auto" (default): Use the template embedded in the GGUF model file
   * - Template name: Use a specific built-in template (e.g., "llama3", "chatml", "gemma")
   *
   * Available templates: chatml, llama2, llama2-sys, llama3, llama4, mistral-v1,
   * mistral-v3, mistral-v7, phi3, phi4, gemma, falcon3, zephyr, deepseek, deepseek2,
   * deepseek3, command-r, and more.
   */
  chatTemplate?: string;
  /**
   * Extract model thinking into AI SDK reasoning parts.
   */
  reasoning?: LlamaCppReasoningConfig;
  cache?: LlamaCppCacheConfig;
}

export interface LlamaCppGenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

interface ResolvedReasoningConfig {
  opening: string;
  closing: string;
  promptPrefix?: string;
  budgetTokens?: number;
}

const mediaMarker = "<__media__>";
const DEFAULT_MAX_TOKENS = -1;
const DEFAULT_TEMPERATURE = 0.8;
const DEFAULT_TOP_P = 0.95;
const DEFAULT_TOP_K = 40;

function isImageMediaType(mediaType?: string): boolean {
  const normalized = mediaType?.toLowerCase();
  return normalized === "image" || normalized?.startsWith("image/") === true;
}

function decodeBase64(base64: string): Uint8Array {
  return Uint8Array.from(Buffer.from(base64, "base64"));
}

function createAbortError(): Error {
  const error = new Error("The operation was aborted");
  error.name = "AbortError";
  return error;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw createAbortError();
  }
}

function parseDataUrl(url: string): { data: Uint8Array; mediaType?: string } {
  const match = url.match(/^data:([^;,]+)?(?:;[^,]*)?;base64,(.*)$/s);
  if (!match) {
    throw new Error(
      "Unsupported image data URL. Only base64-encoded data URLs are supported."
    );
  }

  return {
    mediaType: match[1],
    data: decodeBase64(match[2]),
  };
}

function toImageData(
  part: Extract<LanguageModelV4Message, { role: "user" }>["content"][number]
): { data: Uint8Array; mediaType?: string } | undefined {
  if (part.type !== "file") {
    return undefined;
  }

  if (!isImageMediaType(part.mediaType)) {
    return undefined;
  }

  if (part.data.type === "data" && part.data.data instanceof Uint8Array) {
    return {
      data: part.data.data,
      mediaType: part.mediaType,
    };
  }

  if (part.data.type === "data" && typeof part.data.data === "string") {
    return {
      data: decodeBase64(part.data.data),
      mediaType: part.mediaType,
    };
  }

  if (part.data.type === "url") {
    if (part.data.url.protocol === "data:") {
      return parseDataUrl(part.data.url.href);
    }
    if (part.data.url.protocol !== "file:") {
      throw new Error(
        `Unsupported image URL protocol: ${part.data.url.protocol}. Use data URLs or Uint8Array data.`
      );
    }
    throw new Error(
      "File URL image inputs must be loaded by the AI SDK before reaching this provider."
    );
  }

  return undefined;
}

export interface ParsedReasoningPart {
  type: "text" | "reasoning";
  text: string;
}

export function resolveReasoningConfig(
  reasoning?: LlamaCppReasoningConfig,
  effort: LlamaCppReasoningEffort = "provider-default"
): ResolvedReasoningConfig | undefined {
  if (!reasoning) {
    return undefined;
  }

  const config = reasoning;
  const defaultConfig = thinkTagsReasoning;
  const budgetTokens = config.effortTokenBudget?.[effort];

  if (
    budgetTokens !== undefined &&
    (!Number.isInteger(budgetTokens) || budgetTokens < 0)
  ) {
    throw new Error(
      `reasoning effort "${effort}" maps to invalid token budget ${budgetTokens}; expected a non-negative integer`
    );
  }

  return {
    opening: config.openingMarker ?? defaultConfig.openingMarker!,
    closing: config.closingMarker ?? defaultConfig.closingMarker!,
    promptPrefix:
      config.promptPrefix === false ? undefined : config.promptPrefix,
    ...(budgetTokens !== undefined ? { budgetTokens } : {}),
  };
}

function resolveCallReasoningConfig(
  options: LanguageModelV4CallOptions,
  modelReasoning: LlamaCppReasoningConfig | undefined,
  grammar: string | undefined
): ResolvedReasoningConfig | undefined {
  if (grammar || options.reasoning === "none") {
    return undefined;
  }

  const effort = (options.reasoning ??
    "provider-default") as LlamaCppReasoningEffort;

  if (modelReasoning) {
    return resolveReasoningConfig(modelReasoning, effort);
  }

  return options.reasoning === undefined
    ? undefined
    : resolveReasoningConfig({}, effort);
}

export function splitReasoningContent(
  text: string,
  reasoning: Pick<ResolvedReasoningConfig, "opening" | "closing">
): ParsedReasoningPart[] {
  const parts: ParsedReasoningPart[] = [];
  let cursor = 0;
  let foundReasoning = false;

  while (cursor < text.length) {
    const openingIndex = text.indexOf(reasoning.opening, cursor);

    if (openingIndex === -1) {
      parts.push({ type: "text", text: text.slice(cursor) });
      break;
    }

    foundReasoning = true;

    if (openingIndex > cursor) {
      parts.push({ type: "text", text: text.slice(cursor, openingIndex) });
    }

    const reasoningStart = openingIndex + reasoning.opening.length;
    const closingIndex = text.indexOf(reasoning.closing, reasoningStart);

    if (closingIndex === -1) {
      parts.push({ type: "reasoning", text: text.slice(reasoningStart) });
      cursor = text.length;
      break;
    }

    parts.push({
      type: "reasoning",
      text: text.slice(reasoningStart, closingIndex),
    });
    cursor = closingIndex + reasoning.closing.length;
  }

  if (!foundReasoning) {
    return [{ type: "text", text }];
  }

  return parts;
}

function appendParsedContent(
  content: LanguageModelV4Content[],
  parts: ParsedReasoningPart[],
  options: { includeText: boolean } = { includeText: true }
): void {
  for (const part of parts) {
    if (
      part.text.length === 0 ||
      (part.type === "text" && !options.includeText)
    ) {
      continue;
    }

    if (part.type === "reasoning") {
      content.push({
        type: "reasoning",
        text: part.text,
        providerMetadata: undefined,
      });
    } else {
      content.push({
        type: "text",
        text: part.text,
        providerMetadata: undefined,
      });
    }
  }
}

function getVisibleText(parts: ParsedReasoningPart[]): string {
  return parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function getReasoningSuffixLength(buffer: string, marker: string): number {
  const maxLength = Math.min(buffer.length, marker.length - 1);

  for (let length = maxLength; length > 0; length--) {
    if (marker.startsWith(buffer.slice(buffer.length - length))) {
      return length;
    }
  }

  return 0;
}

function createReasoningTokenProcessor(
  reasoning: Pick<ResolvedReasoningConfig, "opening" | "closing">,
  emitText: (text: string) => void,
  emitReasoning: (text: string) => void,
  endReasoning: () => void
): { push: (token: string) => void; flush: () => void } {
  let buffer = "";
  let state: "text" | "reasoning" = "text";

  const process = () => {
    while (buffer.length > 0) {
      if (state === "text") {
        const openingIndex = buffer.indexOf(reasoning.opening);

        if (openingIndex !== -1) {
          if (openingIndex > 0) {
            emitText(buffer.slice(0, openingIndex));
          }
          buffer = buffer.slice(openingIndex + reasoning.opening.length);
          state = "reasoning";
          continue;
        }

        const suffixLength = getReasoningSuffixLength(
          buffer,
          reasoning.opening
        );
        const emitLength = buffer.length - suffixLength;
        if (emitLength === 0) {
          return;
        }

        emitText(buffer.slice(0, emitLength));
        buffer = buffer.slice(emitLength);
        return;
      }

      const closingIndex = buffer.indexOf(reasoning.closing);

      if (closingIndex !== -1) {
        if (closingIndex > 0) {
          emitReasoning(buffer.slice(0, closingIndex));
        }
        buffer = buffer.slice(closingIndex + reasoning.closing.length);
        endReasoning();
        state = "text";
        continue;
      }

      const suffixLength = getReasoningSuffixLength(buffer, reasoning.closing);
      const emitLength = buffer.length - suffixLength;
      if (emitLength === 0) {
        return;
      }

      emitReasoning(buffer.slice(0, emitLength));
      buffer = buffer.slice(emitLength);
      return;
    }
  };

  return {
    push(token: string) {
      buffer += token;
      process();
    },
    flush() {
      if (buffer.length > 0) {
        if (state === "reasoning") {
          emitReasoning(buffer);
        } else {
          emitText(buffer);
        }
        buffer = "";
      }

      if (state === "reasoning") {
        endReasoning();
        state = "text";
      }
    },
  };
}

export function convertFinishReason(
  reason: string
): LanguageModelV4FinishReason {
  let unified: LanguageModelV4FinishReason["unified"];
  switch (reason) {
    case "stop":
      unified = "stop";
      break;
    case "length":
      unified = "length";
      break;
    default:
      unified = "other";
  }
  return { unified, raw: reason };
}

export function convertUsage(
  promptTokens: number,
  completionTokens: number,
  cache?: {
    read?: number;
    write?: number;
  }
): LanguageModelV4Usage {
  return {
    inputTokens: {
      total: promptTokens,
      noCache:
        cache?.read !== undefined || cache?.write !== undefined
          ? promptTokens - (cache.read ?? 0)
          : undefined,
      cacheRead: cache?.read,
      cacheWrite: cache?.write,
    },
    outputTokens: {
      total: completionTokens,
      text: completionTokens,
      reasoning: undefined,
    },
  };
}

/**
 * Represents a parsed tool call from the model output.
 */
export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

function stringifyToolArguments(input: unknown): string {
  return typeof input === "string" ? input : JSON.stringify(input);
}

function toolCallInput(toolCall: ParsedToolCall | NativeToolCall): string {
  return typeof toolCall.arguments === "string"
    ? toolCall.arguments
    : JSON.stringify(toolCall.arguments);
}

function toNativeTools(tools: LanguageModelV4FunctionTool[]): ToolDefinition[] {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    parametersJson: JSON.stringify(tool.inputSchema ?? {}),
  }));
}

function resolveParallelToolCalls(
  options: LanguageModelV4CallOptions
): boolean {
  const providerOptions = options.providerOptions as
    | Record<string, Record<string, unknown> | undefined>
    | undefined;
  const llamaCppOptions =
    providerOptions?.["llama.cpp"] ?? providerOptions?.llamaCpp;
  return llamaCppOptions?.parallelToolCalls === true;
}

function resolveResponseGrammar(
  responseFormat: LanguageModelV4CallOptions["responseFormat"]
): string | undefined {
  if (responseFormat?.type !== "json") {
    return undefined;
  }

  return convertJsonSchemaToGrammar(
    (responseFormat.schema ?? { type: "object" }) as JSONSchema7
  );
}

/**
 * Generate a GBNF grammar for tool calls based on the provided tool definitions.
 * This grammar constrains the model to produce valid JSON tool calls.
 */
export function generateToolCallGrammar(
  tools: LanguageModelV4FunctionTool[]
): string {
  if (tools.length === 0) {
    throw new Error("At least one function tool is required");
  }

  // Generate the tool-specific argument schemas
  const toolGrammars: string[] = [];

  for (const tool of tools) {
    const converter = new SchemaConverter();
    converter.resolveRefs(tool.inputSchema as JSONSchema7);
    converter.visit(tool.inputSchema as JSONSchema7, `${tool.name}-args`);

    // Get the grammar rules for this tool's arguments
    const argGrammar = converter.formatGrammar();

    // Extract just the rules (without the root rule)
    const lines = argGrammar
      .split("\n")
      .filter((line) => line.trim() && !line.startsWith("root "));

    toolGrammars.push(...lines);
  }

  // Combine all grammars
  const uniqueRules = new Map<string, string>();

  // Add tool-specific argument rules
  for (const line of toolGrammars) {
    const match = line.match(/^(\S+)\s*::=\s*(.+)$/);
    if (match) {
      uniqueRules.set(match[1], match[2]);
    }
  }

  // Build the final grammar
  let grammar = "";

  // Root rule - a tool call object
  grammar += `root ::= "{" space tool-calls-kv "}" space\n`;
  grammar += `tool-calls-kv ::= "\\"tool_calls\\"" space ":" space "[" space tool-call (space "," space tool-call)* space "]"\n`;
  grammar += `tool-call ::= ${tools.map((tool) => `${tool.name}-call`).join(" | ")}\n`;
  grammar += `id-kv ::= "\\"id\\"" space ":" space string\n`;

  for (const tool of tools) {
    grammar += `${tool.name}-call ::= "{" space id-kv "," space "\\"name\\"" space ":" space "\\"${tool.name}\\"" space "," space "\\"arguments\\"" space ":" space ${tool.name}-args "}" space\n`;
  }

  // Add all the tool-specific rules
  for (const [name, rule] of uniqueRules) {
    grammar += `${name} ::= ${rule}\n`;
  }

  // Add common rules
  grammar += `space ::= | " " | "\\n"{1,2} [ \\t]{0,20}\n`;
  grammar += `string ::= "\\"" char* "\\"" space\n`;
  grammar += `char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n`;

  return grammar;
}

/**
 * Parse the model output to extract tool calls.
 * Supports multiple formats:
 * - Single tool call: {"name": "...", "arguments": {...}}
 * - Array of tool calls: [{"name": "...", "arguments": {...}}, ...]
 * - Legacy format with tool_calls wrapper: {"tool_calls": [...]}
 * Returns null if the output is not a valid tool call JSON.
 */
export function parseToolCalls(text: string): ParsedToolCall[] | null {
  try {
    const trimmed = text.trim();

    // Must start with { or [ to be JSON
    if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
      return null;
    }

    const parsed = JSON.parse(trimmed);
    const toolCalls: ParsedToolCall[] = [];

    // Handle array format: [{"name": "...", "arguments": {...}}, ...]
    if (Array.isArray(parsed)) {
      for (const call of parsed) {
        if (
          typeof call.name === "string" &&
          typeof call.arguments === "object"
        ) {
          toolCalls.push({
            id: call.id || generateToolCallId(),
            name: call.name,
            arguments: call.arguments,
          });
        }
      }
      return toolCalls.length > 0 ? toolCalls : null;
    }

    // Handle single object format: {"name": "...", "arguments": {...}}
    if (
      typeof parsed.name === "string" &&
      typeof parsed.arguments === "object"
    ) {
      return [
        {
          id: parsed.id || generateToolCallId(),
          name: parsed.name,
          arguments: parsed.arguments,
        },
      ];
    }

    // Handle legacy tool_calls wrapper format: {"tool_calls": [...]}
    if (parsed.tool_calls && Array.isArray(parsed.tool_calls)) {
      for (const call of parsed.tool_calls) {
        if (
          typeof call.name === "string" &&
          typeof call.arguments === "object"
        ) {
          toolCalls.push({
            id: call.id || generateToolCallId(),
            name: call.name,
            arguments: call.arguments,
          });
        }
      }
      return toolCalls.length > 0 ? toolCalls : null;
    }

    return null;
  } catch {
    return null;
  }
}

/**
 * Generate a unique tool call ID.
 */
function generateToolCallId(): string {
  return `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

/**
 * Build a system prompt that instructs the model to use tools.
 */
export function buildToolSystemPrompt(
  tools: LanguageModelV4FunctionTool[],
  toolChoice?: LanguageModelV4ToolChoice
): string {
  const toolDescriptions = tools
    .map((tool) => {
      const params = JSON.stringify(tool.inputSchema, null, 2);
      return `- ${tool.name}: ${tool.description || "No description"}\n  Parameters: ${params}`;
    })
    .join("\n\n");

  const toolChoiceInstruction =
    toolChoice?.type === "required"
      ? "\nYou must call one of these tools. Do not answer with normal text.\n"
      : toolChoice?.type === "tool"
        ? `\nYou must call the "${toolChoice.toolName}" tool. Do not answer with normal text.\n`
        : "";
  const finalRule =
    toolChoice?.type === "required" || toolChoice?.type === "tool"
      ? "- Do not answer with normal text"
      : "- If you don't need to use a tool, respond normally with text";

  return `You have access to the following tools:

${toolDescriptions}
${toolChoiceInstruction}

When you need to use a tool, respond ONLY with a JSON object in this exact format (no other text):
{"name": "<tool_name>", "arguments": {<tool_arguments>}}

For multiple tool calls, use an array:
[{"name": "<tool_name>", "arguments": {...}}, ...]

Rules:
- The "name" must exactly match one of the available tool names
- The "arguments" must be a valid JSON object matching the tool's parameter schema
- Output ONLY the JSON, no explanation or other text
${finalRule}`;
}

/**
 * Convert AI SDK messages to simple role/content format for the native layer.
 * The native layer will apply the appropriate chat template.
 */
export function convertMessages(
  messages: LanguageModelV4Message[],
  reasoning?: Pick<ResolvedReasoningConfig, "promptPrefix">
): ChatMessage[] {
  const result: ChatMessage[] = [];
  let reasoningPromptAdded = false;

  const addMessage = (message: ChatMessage) => {
    if (
      reasoning?.promptPrefix &&
      !reasoningPromptAdded &&
      message.role === "system"
    ) {
      result.push({
        ...message,
        content: `${reasoning.promptPrefix}${message.content}`,
      });
      reasoningPromptAdded = true;
      return;
    }

    result.push(message);
  };

  for (const message of messages) {
    switch (message.role) {
      case "system":
        addMessage({
          role: "system",
          content: message.content,
        });
        break;
      case "user":
        // Preserve ordered text and image parts for libmtmd. The native layer
        // replaces each media marker with the corresponding image embedding.
        let userContent = "";
        const images: ChatMessage["images"] = [];
        for (const part of message.content) {
          if (part.type === "text") {
            userContent += part.text;
          } else if (part.type === "file") {
            const image = toImageData(part);
            if (image) {
              if (userContent.length > 0 && !/\s$/.test(userContent)) {
                userContent += "\n";
              }
              userContent += mediaMarker;
              images.push(image);
            }
          }
        }
        addMessage({
          role: "user",
          content: userContent,
          ...(images.length > 0 ? { images } : {}),
        });
        break;
      case "assistant":
        // Extract text and tool call content from assistant messages
        let assistantContent = "";
        const toolCallParts: Array<{
          toolCallId: string;
          toolName: string;
          input: string;
        }> = [];

        for (const part of message.content) {
          if (part.type === "text") {
            assistantContent += part.text;
          } else if (part.type === "tool-call") {
            toolCallParts.push({
              toolCallId: part.toolCallId,
              toolName: part.toolName,
              input: stringifyToolArguments(part.input),
            });
          }
        }

        if (assistantContent || toolCallParts.length > 0) {
          addMessage({
            role: "assistant",
            content: assistantContent,
            ...(toolCallParts.length > 0
              ? {
                  toolCalls: toolCallParts.map((tc) => ({
                    id: tc.toolCallId,
                    name: tc.toolName,
                    arguments: tc.input,
                  })),
                }
              : {}),
          });
        }
        break;
      case "tool":
        // Preserve tool results as native tool messages for llama.cpp common-chat.
        for (const part of message.content) {
          if (part.type === "tool-result") {
            const output = part.output;
            let resultText = "";

            if (output.type === "text") {
              resultText = output.value;
            } else if (output.type === "json") {
              resultText = JSON.stringify(output.value);
            } else if (output.type === "error-text") {
              resultText = `Error: ${output.value}`;
            } else if (output.type === "error-json") {
              resultText = `Error: ${JSON.stringify(output.value)}`;
            } else if (output.type === "execution-denied") {
              resultText = `Execution denied${output.reason ? `: ${output.reason}` : ""}`;
            } else if (output.type === "content") {
              // Convert content array to text representation
              resultText = output.value
                .map((item) => {
                  if (item.type === "text") {
                    return item.text;
                  } else if (item.type === "file") {
                    return `[File: ${item.mediaType}]`;
                  } else {
                    return `[Unknown content type]`;
                  }
                })
                .join("\n");
            }

            addMessage({
              role: "tool",
              content: resultText,
              toolName: part.toolName,
              toolCallId: part.toolCallId,
            });
          }
        }
        break;
    }
  }

  if (reasoning?.promptPrefix && !reasoningPromptAdded) {
    result.unshift({
      role: "system",
      content: reasoning.promptPrefix,
    });
  }

  return result;
}

function resolveActiveFunctionTools(
  functionTools: LanguageModelV4FunctionTool[],
  toolChoice: LanguageModelV4ToolChoice | undefined
): {
  tools?: LanguageModelV4FunctionTool[];
  parseToolCalls: boolean;
  toolChoice?: "auto" | "required" | "none";
} {
  if (functionTools.length === 0 || toolChoice?.type === "none") {
    return { parseToolCalls: false, toolChoice: "none" };
  }

  if (toolChoice?.type === "tool") {
    const selectedTool = functionTools.find(
      (tool) => tool.name === toolChoice.toolName
    );

    if (!selectedTool) {
      throw new Error(
        `toolChoice references unknown function tool: ${toolChoice.toolName}`
      );
    }

    return {
      tools: [selectedTool],
      parseToolCalls: true,
      toolChoice: "required",
    };
  }

  if (toolChoice?.type === "required") {
    return {
      tools: functionTools,
      parseToolCalls: true,
      toolChoice: "required",
    };
  }

  return {
    tools: functionTools,
    parseToolCalls: true,
    toolChoice: "auto",
  };
}

export class LlamaCppLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;
  readonly contextSize: number;

  /**
   * Supported URL patterns - empty since we only support local files
   */
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private modelHandle: number | null = null;
  private readonly config: LlamaCppModelConfig;
  private initPromise: Promise<void> | null = null;
  private loadedContextSize: number | null = null;
  private readonly activeRequests = new Map<
    string,
    {
      handle: number;
      requestId: string;
      settled: Promise<void>;
    }
  >();
  constructor(config: LlamaCppModelConfig) {
    this.config = config;
    this.modelId = config.modelPath;
    this.contextSize = config.contextSize ?? 2048;
  }

  private async ensureModelLoaded(): Promise<number> {
    if (this.modelHandle !== null && isModelLoaded(this.modelHandle)) {
      return this.modelHandle;
    }

    if (this.initPromise) {
      await this.initPromise;
      if (this.modelHandle !== null) {
        return this.modelHandle;
      }
    }

    const initPromise = (async () => {
      const memorySafety = await checkModelMemorySafety({
        modelPath: this.config.modelPath,
        mmprojPath: this.config.mmprojPath,
        memory: this.config.memory,
        contextSize: this.contextSize,
        memorySafety: this.config.memorySafety,
      });
      const options: LoadModelOptions = {
        modelPath: this.config.modelPath,
        contextSize: memorySafety.contextSize,
        gpuLayers: this.config.gpuLayers ?? 99,
        threads: this.config.threads ?? 4,
        debug: this.config.debug ?? false,
        logPrompts: this.config.logPrompts ?? false,
        chatTemplate: this.config.chatTemplate ?? "auto",
      };

      if (this.config.mmprojPath !== undefined) {
        options.mmprojPath = this.config.mmprojPath;
      }

      this.modelHandle = await loadModel(options);
      this.loadedContextSize = memorySafety.contextSize;
    })();
    this.initPromise = initPromise;

    try {
      await initPromise;
    } finally {
      if (this.initPromise === initPromise) {
        this.initPromise = null;
      }
    }

    if (this.modelHandle === null) {
      throw new Error("Failed to load model");
    }

    return this.modelHandle;
  }

  async dispose(): Promise<void> {
    const handle = this.modelHandle;
    if (handle !== null) {
      const activeRequests = [...this.activeRequests.values()].filter(
        (request) => request.handle === handle
      );

      for (const request of activeRequests) {
        cancelGeneration(handle, request.requestId);
      }

      await Promise.all(activeRequests.map((request) => request.settled));

      if (this.modelHandle !== handle) {
        return;
      }

      unloadModel(handle);
      this.modelHandle = null;
      this.loadedContextSize = null;
    }
  }

  async doGenerate(
    options: LanguageModelV4CallOptions
  ): Promise<LanguageModelV4GenerateResult> {
    throwIfAborted(options.abortSignal);
    const handle = await this.ensureModelLoaded();
    const responseGrammar = resolveResponseGrammar(options.responseFormat);

    // Extract function tools from the tools array
    const functionTools =
      options.tools?.filter(
        (t): t is LanguageModelV4FunctionTool => t.type === "function"
      ) ?? [];

    const toolSettings = resolveActiveFunctionTools(
      functionTools,
      options.toolChoice
    );

    if (responseGrammar && toolSettings.parseToolCalls) {
      throw new Error(
        "Structured JSON response format cannot be combined with active tools."
      );
    }

    const grammar = responseGrammar;

    const reasoningConfig = resolveCallReasoningConfig(
      options,
      this.config.reasoning,
      grammar
    );

    const messages = convertMessages(options.prompt, reasoningConfig);
    const requestId = crypto.randomUUID();

    const generateOptions: GenerateOptions = {
      requestId,
      messages,
      tools: toolSettings.tools ? toNativeTools(toolSettings.tools) : undefined,
      toolChoice: toolSettings.toolChoice,
      parallelToolCalls: resolveParallelToolCalls(options),
      maxTokens: options.maxOutputTokens ?? DEFAULT_MAX_TOKENS,
      temperature: options.temperature ?? DEFAULT_TEMPERATURE,
      topP: options.topP ?? DEFAULT_TOP_P,
      topK: options.topK ?? DEFAULT_TOP_K,
      stopSequences: options.stopSequences,
      grammar,
    };
    if (reasoningConfig?.budgetTokens !== undefined) {
      generateOptions.reasoningBudgetTokens = reasoningConfig.budgetTokens;
      generateOptions.reasoningBudgetStart = reasoningConfig.opening;
      generateOptions.reasoningBudgetEnd = reasoningConfig.closing;
    }
    if (options.seed !== undefined) {
      validateGenerationSeed(options.seed);
      generateOptions.seed = options.seed;
    }
    if (this.config.cache?.mode === "prefix") {
      generateOptions.promptCache = true;
    }
    validateGenerationContextSize(
      this.loadedContextSize,
      generateOptions.maxTokens
    );

    const result = await this.runWithAbortSignal(
      handle,
      requestId,
      options.abortSignal,
      () => generate(handle, generateOptions)
    );
    const parsedContent = reasoningConfig
      ? splitReasoningContent(result.text, reasoningConfig)
      : [{ type: "text" as const, text: result.text }];
    const visibleText = getVisibleText(parsedContent);

    const warnings: SharedV4Warning[] = [];
    const content: LanguageModelV4Content[] = [];
    let finishReason = convertFinishReason(result.finishReason);

    const nativeToolCalls = result.toolCalls;
    if (nativeToolCalls && nativeToolCalls.length > 0) {
      appendParsedContent(content, parsedContent);

      for (const toolCall of nativeToolCalls) {
        const toolCallId = toolCall.id || generateToolCallId();
        content.push({
          type: "tool-call",
          toolCallId,
          toolName: toolCall.name,
          input: toolCallInput(toolCall),
        });
      }

      finishReason = {
        unified: "tool-calls",
        raw: "tool-calls",
      };
    } else if (toolSettings.parseToolCalls) {
      const toolCalls = parseToolCalls(visibleText);

      if (toolCalls && toolCalls.length > 0) {
        appendParsedContent(content, parsedContent, { includeText: false });

        // Add tool calls to content
        for (const toolCall of toolCalls) {
          const toolCallId = toolCall.id || generateToolCallId();
          content.push({
            type: "tool-call",
            toolCallId,
            toolName: toolCall.name,
            input: toolCallInput(toolCall),
          });
        }
        // Set finish reason to tool-calls
        finishReason = {
          unified: "tool-calls",
          raw: "tool-calls",
        };
      } else {
        // No valid tool calls found, return as text
        appendParsedContent(content, parsedContent);
      }
    } else {
      // No tools, return text content
      appendParsedContent(content, parsedContent);
    }

    return {
      content,
      finishReason,
      usage: convertUsage(result.promptTokens, result.completionTokens, {
        read: result.cacheReadTokens,
        write: result.cacheWriteTokens,
      }),
      warnings,
      request: {
        body: generateOptions,
      },
    };
  }

  async doStream(
    options: LanguageModelV4CallOptions
  ): Promise<LanguageModelV4StreamResult> {
    throwIfAborted(options.abortSignal);
    const handle = await this.ensureModelLoaded();
    const responseGrammar = resolveResponseGrammar(options.responseFormat);

    // Extract function tools from the tools array
    const functionTools =
      options.tools?.filter(
        (t): t is LanguageModelV4FunctionTool => t.type === "function"
      ) ?? [];

    const toolSettings = resolveActiveFunctionTools(
      functionTools,
      options.toolChoice
    );

    if (responseGrammar && toolSettings.parseToolCalls) {
      throw new Error(
        "Structured JSON response format cannot be combined with active tools."
      );
    }

    const grammar = responseGrammar;

    const reasoningConfig = resolveCallReasoningConfig(
      options,
      this.config.reasoning,
      grammar
    );

    const messages = convertMessages(options.prompt, reasoningConfig);
    const requestId = crypto.randomUUID();

    const generateOptions: GenerateOptions = {
      requestId,
      messages,
      tools: toolSettings.tools ? toNativeTools(toolSettings.tools) : undefined,
      toolChoice: toolSettings.toolChoice,
      parallelToolCalls: resolveParallelToolCalls(options),
      maxTokens: options.maxOutputTokens ?? DEFAULT_MAX_TOKENS,
      temperature: options.temperature ?? DEFAULT_TEMPERATURE,
      topP: options.topP ?? DEFAULT_TOP_P,
      topK: options.topK ?? DEFAULT_TOP_K,
      stopSequences: options.stopSequences,
      grammar,
    };
    if (reasoningConfig?.budgetTokens !== undefined) {
      generateOptions.reasoningBudgetTokens = reasoningConfig.budgetTokens;
      generateOptions.reasoningBudgetStart = reasoningConfig.opening;
      generateOptions.reasoningBudgetEnd = reasoningConfig.closing;
    }
    if (options.seed !== undefined) {
      validateGenerationSeed(options.seed);
      generateOptions.seed = options.seed;
    }
    if (this.config.cache?.mode === "prefix") {
      generateOptions.promptCache = true;
    }
    validateGenerationContextSize(
      this.loadedContextSize,
      generateOptions.maxTokens
    );

    const textId = crypto.randomUUID();

    let streamCancelled = false;

    const stream = new ReadableStream<LanguageModelV4StreamPart>({
      start: async (controller) => {
        try {
          // Emit stream start
          controller.enqueue({
            type: "stream-start",
            warnings: [],
          });

          let textStartEmitted = false;
          let reasoningId: string | undefined;

          const emitTextDelta = (delta: string) => {
            if (delta.length === 0) {
              return;
            }

            if (toolSettings.parseToolCalls) {
              return;
            }

            // Emit text start on first actual text delta
            if (!textStartEmitted) {
              controller.enqueue({
                type: "text-start",
                id: textId,
              });
              textStartEmitted = true;
            }

            controller.enqueue({
              type: "text-delta",
              id: textId,
              delta,
            });
          };

          const emitReasoningDelta = (delta: string) => {
            if (delta.length === 0) {
              return;
            }

            if (!reasoningId) {
              reasoningId = crypto.randomUUID();
              controller.enqueue({
                type: "reasoning-start",
                id: reasoningId,
              });
            }

            controller.enqueue({
              type: "reasoning-delta",
              id: reasoningId,
              delta,
            });
          };

          const endReasoning = () => {
            if (!reasoningId) {
              return;
            }

            controller.enqueue({
              type: "reasoning-end",
              id: reasoningId,
            });
            reasoningId = undefined;
          };

          const reasoningProcessor = reasoningConfig
            ? createReasoningTokenProcessor(
                reasoningConfig,
                emitTextDelta,
                emitReasoningDelta,
                endReasoning
              )
            : undefined;

          const result = await this.runWithAbortSignal(
            handle,
            requestId,
            options.abortSignal,
            () =>
              generateStream(handle, generateOptions, (token) => {
                if (options.abortSignal?.aborted || streamCancelled) {
                  return;
                }

                if (toolSettings.parseToolCalls) {
                  return;
                } else if (reasoningProcessor) {
                  reasoningProcessor.push(token);
                } else {
                  emitTextDelta(token);
                }
              })
          );

          if (streamCancelled) {
            return;
          }

          reasoningProcessor?.flush();

          // Check for tool calls if tools were provided
          let finishReason = convertFinishReason(result.finishReason);

          if (toolSettings.parseToolCalls) {
            const parsedText = result.text;
            const nativeToolCalls = result.toolCalls;
            const fallbackToolCalls = nativeToolCalls?.length
              ? undefined
              : parseToolCalls(parsedText);
            const toolCalls = nativeToolCalls?.length
              ? nativeToolCalls
              : fallbackToolCalls;

            if (toolCalls && toolCalls.length > 0) {
              if (nativeToolCalls?.length && parsedText.length > 0) {
                controller.enqueue({
                  type: "text-start",
                  id: textId,
                });
                controller.enqueue({
                  type: "text-delta",
                  id: textId,
                  delta: parsedText,
                });
                textStartEmitted = true;
              }

              // Emit tool call events
              for (const toolCall of toolCalls) {
                const toolCallId = toolCall.id || generateToolCallId();

                controller.enqueue({
                  type: "tool-call",
                  toolCallId,
                  toolName: toolCall.name,
                  input: toolCallInput(toolCall),
                });
              }
              // Set finish reason to tool-calls
              finishReason = {
                unified: "tool-calls",
                raw: "tool-calls",
              };
            } else if (parsedText.length > 0) {
              controller.enqueue({
                type: "text-start",
                id: textId,
              });
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: parsedText,
              });
              textStartEmitted = true;
            }
          }

          // Emit text end after tool-call fallback has had a chance to emit text.
          if (textStartEmitted) {
            controller.enqueue({
              type: "text-end",
              id: textId,
            });
          }

          // Emit finish
          controller.enqueue({
            type: "finish",
            finishReason,
            usage: convertUsage(result.promptTokens, result.completionTokens, {
              read: result.cacheReadTokens,
              write: result.cacheWriteTokens,
            }),
          });

          controller.close();
        } catch (error) {
          if (!streamCancelled) {
            controller.error(error);
          }
        }
      },
      cancel: () => {
        streamCancelled = true;
        cancelGeneration(handle, requestId);
      },
    });

    return {
      stream,
      request: {
        body: generateOptions,
      },
    };
  }

  private async runWithAbortSignal<T>(
    handle: number,
    requestId: string,
    signal: AbortSignal | undefined,
    run: () => Promise<T>
  ): Promise<T> {
    throwIfAborted(signal);

    let abortRequested = false;
    let abortListener: (() => void) | undefined;

    const runPromise = Promise.resolve().then(run);
    this.activeRequests.set(requestId, {
      handle,
      requestId,
      settled: runPromise.then(
        () => undefined,
        () => undefined
      ),
    });

    const cleanup = () => {
      this.activeRequests.delete(requestId);
      if (signal && abortListener) {
        signal.removeEventListener("abort", abortListener);
      }
    };

    if (signal) {
      abortListener = () => {
        abortRequested = true;
        cancelGeneration(handle, requestId);
      };

      signal.addEventListener("abort", abortListener, { once: true });

      if (signal.aborted) {
        abortListener();
      }
    }

    try {
      const result = await runPromise;
      if (abortRequested) {
        throw createAbortError();
      }
      return result;
    } catch (error) {
      if (abortRequested) {
        throw createAbortError();
      }
      throw error;
    } finally {
      cleanup();
    }
  }
}

async function checkModelMemorySafety(options: {
  modelPath: string;
  mmprojPath?: string;
  memory?: LlamaCppModelMemoryInfo;
  contextSize: number;
  memorySafety?: LlamaCppMemorySafetyConfig;
}) {
  const [modelFileSizeBytes, mmprojFileSizeBytes] = await Promise.all([
    getFileSize(options.modelPath),
    options.mmprojPath ? getFileSize(options.mmprojPath) : undefined,
  ]);

  return checkMemorySafety({
    model: options.memory,
    contextSize: options.contextSize,
    modelFileSizeBytes,
    mmprojFileSizeBytes,
    memorySafety: options.memorySafety,
  });
}

async function getFileSize(path: string): Promise<number | undefined> {
  try {
    return (await stat(path)).size;
  } catch {
    return undefined;
  }
}

function validateGenerationContextSize(
  contextSize: number | null,
  maxTokens: number | undefined
): void {
  if (contextSize === null || maxTokens === undefined) {
    return;
  }

  if (!Number.isInteger(maxTokens) || (maxTokens <= 0 && maxTokens !== -1)) {
    throw new Error("maxOutputTokens must be -1 or a positive integer");
  }

  if (maxTokens === -1) {
    return;
  }

  if (maxTokens > contextSize) {
    throw new Error(
      `maxOutputTokens ${maxTokens.toLocaleString()} exceeds the loaded contextSize ` +
        `${contextSize.toLocaleString()}.`
    );
  }
}

function validateGenerationSeed(seed: number): void {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error("seed must be an integer between 0 and 4294967295");
  }
}
