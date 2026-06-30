import type { LanguageModelV4CallOptions } from "@ai-sdk/provider";

export type LlamaCppReasoningEffort = Exclude<
  NonNullable<LanguageModelV4CallOptions["reasoning"]>,
  "none"
>;

export interface LlamaCppReasoningConfig {
  /**
   * Opening marker that starts reasoning content.
   * Default: `<think>`
   */
  openingMarker?: string;
  /**
   * Closing marker that ends reasoning content.
   * Default: `</think>`
   */
  closingMarker?: string;
  /**
   * Prefix added to the first system prompt to enable thinking.
   * Set to false to disable prompt injection.
   */
  promptPrefix?: string | false;
  /**
   * Map AI SDK reasoning effort values to a llama.cpp reasoning token budget.
   *
   * `provider-default` applies when the call omits `reasoning` or explicitly
   * passes `"provider-default"`. Omit an effort to leave that effort
   * unrestricted.
   */
  effortTokenBudget?: Partial<Record<LlamaCppReasoningEffort, number>>;
}

export const thinkTagsReasoning: LlamaCppReasoningConfig = {
  openingMarker: "<think>",
  closingMarker: "</think>",
};

export interface LlamaCppKvCacheLayerMemoryInfo {
  /**
   * Number of layers that share this KV-cache shape.
   */
  count: number;

  /**
   * Number of key/value heads when K and V use the same head count.
   */
  keyValueHeads?: number;

  /**
   * Number of key heads. Defaults to `keyValueHeads`.
   */
  keyHeads?: number;

  /**
   * Number of value heads. Defaults to `keyValueHeads`.
   */
  valueHeads?: number;

  /**
   * Head dimension when K and V use the same dimension.
   */
  headDim?: number;

  /**
   * Key head dimension. Defaults to `headDim`.
   */
  keyHeadDim?: number;

  /**
   * Value head dimension. Defaults to `headDim`.
   */
  valueHeadDim?: number;
}

export interface LlamaCppModelMemoryInfo {
  /**
   * Maximum context size supported by the model architecture.
   */
  maxContextSize?: number;

  /**
   * KV-cache shape used to estimate memory pressure before creating a
   * llama.cpp context.
   */
  kvCache: {
    /**
     * Bytes per K/V element. llama.cpp defaults to f16 KV cache.
     */
    bytesPerValue?: number;
    layers: LlamaCppKvCacheLayerMemoryInfo[];
  };
}

export interface LlamaCppMemorySafetyConfig {
  /**
   * - "throw" (default): reject unsafe context sizes before native allocation
   * - "clamp": reduce context size to the estimated safe maximum
   * - "off": skip dynamic memory checks
   */
  mode?: "throw" | "clamp" | "off";

  /**
   * Explicit memory budget for model weights, KV cache, and overhead.
   * Defaults to a conservative budget derived from available system memory.
   */
  maxMemoryBytes?: number;

  /**
   * Fraction of currently available system memory that may be used.
   * Default: 0.9.
   */
  memoryUtilization?: number;

  /**
   * Memory kept free for macOS, other applications, and allocation spikes.
   * Default: 10% of total memory, bounded to 4-16 GiB.
   */
  reserveMemoryBytes?: number;

  /**
   * Override bytes per K/V cache element. Defaults to model metadata or f16.
   */
  kvCacheBytesPerValue?: number;

  /**
   * Extra overhead budget for llama.cpp compute buffers and Metal allocations.
   * Default: max(1 GiB, 15% of KV cache).
   */
  computeOverheadBytes?: number;
}

export interface LlamaCppCacheConfig {
  /**
   * Reuse matching prompt prefixes from the previous request on the same model
   * instance. This makes the model stateful and is intended for single-threaded
   * chat loops where each request extends the previous conversation.
   */
  mode?: "prefix";
}

export interface LlamaCppModelInfo {
  /**
   * Chat template to use for formatting messages.
   * - "auto" (default): Use the template embedded in the GGUF model file
   * - Template name: Use a specific built-in template (e.g., "llama3", "gemma")
   */
  chatTemplate?: string;

  /**
   * Extract model thinking into AI SDK reasoning parts.
   */
  reasoning?: LlamaCppReasoningConfig;

  /**
   * Architecture information used to estimate context memory usage.
   */
  memory?: LlamaCppModelMemoryInfo;
}

export interface LlamaCppProviderConfig {
  /**
   * Path to the GGUF model file.
   */
  modelPath: string;

  /**
   * Path to the multimodal projector GGUF file.
   *
   * Required when passing image inputs to language model calls.
   */
  mmprojPath?: string;

  /**
   * Model-specific metadata such as chat template and reasoning format.
   */
  model?: LlamaCppModelInfo;

  /**
   * Maximum context size (default: 2048).
   *
   * This setting is highly dependent on the model and the memory available on
   * the machine. Higher values can consume significant memory and may freeze
   * the machine if set too high. Monitor system memory when increasing this
   * value.
   */
  contextSize?: number;

  /**
   * Protect against context sizes that are likely to exhaust available memory.
   *
   * Enabled by default when `model.memory` metadata is provided. Set to
   * `{ mode: "off" }` to bypass the check.
   */
  memorySafety?: LlamaCppMemorySafetyConfig;

  /**
   * Number of layers to offload to GPU (default: 99, meaning all layers).
   * Set to 0 to disable GPU acceleration.
   */
  gpuLayers?: number;

  /**
   * Number of CPU threads to use (default2: 4).
   */
  threads?: number;

  /**
   * Enable verbose debug output from llama.cpp (default: false).
   */
  debug?: boolean;

  /**
   * Print the final chat-template-rendered prompt sent to llama.cpp to stderr.
   *
   * This may include private user data. Intended for local debugging only.
   */
  logPrompts?: boolean;

  /**
   * Optional prompt cache. Set to `{ mode: "prefix" }` to reuse llama.cpp KV
   * cache state when a request starts with the previous request/response tokens.
   */
  cache?: LlamaCppCacheConfig;
}
