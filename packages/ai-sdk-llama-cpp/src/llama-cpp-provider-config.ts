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
}

export const thinkTagsReasoning: LlamaCppReasoningConfig = {
  openingMarker: "<think>",
  closingMarker: "</think>",
};

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
}
