export interface LlamaCppReasoningFormat {
  opening: string;
  closing: string;
}

export const gemma4Reasoning: LlamaCppReasoningFormat = {
  opening: "<|channel>thought\n",
  closing: "<channel|>",
};

export const thinkTagsReasoning: LlamaCppReasoningFormat = {
  opening: "<think>",
  closing: "</think>",
};

export interface LlamaCppReasoningConfig {
  /**
   * Reasoning marker format to extract from model output.
   * Default: `thinkTagsReasoning` (`<think>...</think>`)
   */
  format?: LlamaCppReasoningFormat;
  /**
   * Prefix added to the first system prompt to enable thinking.
   * When using `gemma4Reasoning`, defaults to `<|think|>\n`.
   * Set to false to disable prompt injection.
   */
  promptPrefix?: string | false;
}

export interface LlamaCppProviderConfig {
  /**
   * Path to the GGUF model file.
   */
  modelPath: string;

  /**
   * Maximum context size (default: 2048).
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
