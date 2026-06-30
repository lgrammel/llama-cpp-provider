import { createRequire } from "node:module";
import { stat } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Load the native binding
const binding = require(
  join(__dirname, "..", "build", "Release", "llama_binding.node")
) as NativeBinding;

export interface LoadModelOptions {
  modelPath: string;
  mmprojPath?: string;
  gpuLayers?: number;
  contextSize?: number;
  threads?: number;
  debug?: boolean;
  /**
   * Print the final chat-template-rendered prompt sent to llama.cpp to stderr.
   *
   * This may include private user data. Intended for local debugging only.
   */
  logPrompts?: boolean;
  /**
   * Chat template to use for formatting messages.
   * - "auto" (default): Use the template embedded in the GGUF model file
   * - Template name: Use a specific built-in template (e.g., "llama3", "chatml", "gemma")
   */
  chatTemplate?: string;
  /**
   * Whether to load the model for embedding generation.
   * When true, creates an embedding context with mean pooling enabled.
   * Default: false
   */
  embedding?: boolean;
}

export interface ChatMessage {
  role: string;
  content: string;
  images?: ImageInput[];
  toolCalls?: NativeToolCall[];
  toolName?: string;
  toolCallId?: string;
}

export interface ImageInput {
  data: Uint8Array;
  mediaType?: string;
}

export interface NativeToolCall {
  id?: string;
  name: string;
  arguments: string;
}

export interface ToolDefinition {
  name: string;
  description?: string;
  parametersJson: string;
}

export interface GenerateOptions {
  /** Internal ID used to cancel one in-flight generation without affecting others. */
  requestId?: string;
  messages: ChatMessage[];
  tools?: ToolDefinition[];
  toolChoice?: "auto" | "required" | "none";
  parallelToolCalls?: boolean;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  seed?: number;
  stopSequences?: string[];
  /** GBNF grammar string for structured output */
  grammar?: string;
  /** Maximum generated tokens allowed inside the reasoning block. */
  reasoningBudgetTokens?: number;
  /** Marker that starts model reasoning content. */
  reasoningBudgetStart?: string;
  /** Marker that ends model reasoning content. */
  reasoningBudgetEnd?: string;
  /** Whether chat templates that support thinking should enable thinking. */
  enableThinking?: boolean;
  /** Reuse matching prompt prefixes from the previous request on this model. */
  promptCache?: boolean;
}

export interface GenerateResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  finishReason: "stop" | "length" | "error";
  errorMessage?: string;
  toolCalls?: NativeToolCall[];
}

export interface EmbedOptions {
  texts: string[];
}

export interface EmbedResult {
  embeddings: Float32Array[];
  totalTokens: number;
}

interface NativeBinding {
  loadModel(
    options: LoadModelOptions,
    callback: (error: string | null, handle: number | null) => void
  ): void;
  unloadModel(handle: number): boolean;
  generate(
    handle: number,
    options: GenerateOptions,
    callback: (error: string | null, result: GenerateResult | null) => void
  ): void;
  generateStream(
    handle: number,
    options: GenerateOptions,
    tokenCallback: (token: string) => void,
    doneCallback: (error: string | null, result: GenerateResult | null) => void
  ): void;
  cancelGeneration(handle: number, requestId?: string): boolean;
  isModelLoaded(handle: number): boolean;
  // Embedding functions
  embed(
    handle: number,
    options: EmbedOptions,
    callback: (error: string | null, result: EmbedResult | null) => void
  ): void;
}

export function loadModel(options: LoadModelOptions): Promise<number> {
  const nativeOptions = omitUndefinedLoadModelOptions(options);

  return Promise.all([
    validateModelPath(nativeOptions.modelPath, "modelPath"),
    nativeOptions.mmprojPath
      ? validateModelPath(nativeOptions.mmprojPath, "mmprojPath")
      : undefined,
  ]).then(
    () =>
      new Promise((resolve, reject) => {
        binding.loadModel(nativeOptions, (error, handle) => {
          if (error) {
            reject(new Error(error));
          } else if (handle !== null) {
            resolve(handle);
          } else {
            reject(new Error("Failed to load model: unknown error"));
          }
        });
      })
  );
}

function omitUndefinedLoadModelOptions(
  options: LoadModelOptions
): LoadModelOptions {
  const nativeOptions: LoadModelOptions = {
    modelPath: options.modelPath,
  };

  if (options.mmprojPath !== undefined) {
    nativeOptions.mmprojPath = options.mmprojPath;
  }
  if (options.gpuLayers !== undefined) {
    nativeOptions.gpuLayers = options.gpuLayers;
  }
  if (options.contextSize !== undefined) {
    nativeOptions.contextSize = options.contextSize;
  }
  if (options.threads !== undefined) {
    nativeOptions.threads = options.threads;
  }
  if (options.debug !== undefined) {
    nativeOptions.debug = options.debug;
  }
  if (options.logPrompts !== undefined) {
    nativeOptions.logPrompts = options.logPrompts;
  }
  if (options.chatTemplate !== undefined) {
    nativeOptions.chatTemplate = options.chatTemplate;
  }
  if (options.embedding !== undefined) {
    nativeOptions.embedding = options.embedding;
  }

  return nativeOptions;
}

async function validateModelPath(
  modelPath: string,
  optionName: "modelPath" | "mmprojPath"
): Promise<void> {
  if (modelPath.startsWith("~/")) {
    throw new Error(
      `Failed to load model: ${optionName} uses '~', which is not expanded automatically. ` +
        `Pass an absolute path instead. Received: ${modelPath}`
    );
  }

  let modelFile;
  try {
    modelFile = await stat(modelPath);
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      throw new Error(
        `Failed to load model: ${optionName} file does not exist at ${modelPath}`
      );
    }

    throw new Error(
      `Failed to load model: could not access ${optionName} at ${modelPath}: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  }

  if (!modelFile.isFile()) {
    throw new Error(
      `Failed to load model: expected ${optionName} to be a GGUF file but found a directory at ${modelPath}`
    );
  }
}

export function unloadModel(handle: number): boolean {
  return binding.unloadModel(handle);
}

export function generate(
  handle: number,
  options: GenerateOptions
): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    binding.generate(handle, options, (error, result) => {
      if (error) {
        reject(new Error(error));
      } else if (result) {
        if (result.finishReason === "error") {
          reject(
            new Error(
              result.errorMessage ?? "Failed to generate: unknown error"
            )
          );
        } else {
          resolve(result);
        }
      } else {
        reject(new Error("Failed to generate: unknown error"));
      }
    });
  });
}

export function cancelGeneration(handle: number, requestId?: string): boolean {
  return binding.cancelGeneration(handle, requestId);
}

export function generateStream(
  handle: number,
  options: GenerateOptions,
  onToken: (token: string) => void
): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    binding.generateStream(handle, options, onToken, (error, result) => {
      if (error) {
        reject(new Error(error));
      } else if (result) {
        if (result.finishReason === "error") {
          reject(
            new Error(
              result.errorMessage ?? "Failed to generate stream: unknown error"
            )
          );
        } else {
          resolve(result);
        }
      } else {
        reject(new Error("Failed to generate stream: unknown error"));
      }
    });
  });
}

export function isModelLoaded(handle: number): boolean {
  return binding.isModelLoaded(handle);
}

export function embed(
  handle: number,
  options: EmbedOptions
): Promise<EmbedResult> {
  return new Promise((resolve, reject) => {
    binding.embed(handle, options, (error, result) => {
      if (error) {
        reject(new Error(error));
      } else if (result) {
        resolve(result);
      } else {
        reject(new Error("Failed to generate embeddings: unknown error"));
      }
    });
  });
}
