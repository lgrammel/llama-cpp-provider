export { llamaCpp, type LlamaCppProvider } from "./llama-cpp-provider.js";
export {
  thinkTagsReasoning,
  type LlamaCppCacheConfig,
  type LlamaCppKvCacheLayerMemoryInfo,
  type LlamaCppMemorySafetyConfig,
  type LlamaCppModelInfo,
  type LlamaCppModelMemoryInfo,
  type LlamaCppProviderConfig,
  type LlamaCppReasoningConfig,
} from "./llama-cpp-provider-config.js";
export { gemma4_26b_a4b, gemma4_31b_it, gemma4Reasoning } from "./gemma4.js";
export { qwen3_6_dense, qwen3_6_moe } from "./qwen3.6.js";
export {
  checkMemorySafety,
  estimateMemoryUsage,
  type EstimateMemoryUsageOptions,
  type MemorySafetyCheckOptions,
  type MemorySafetyCheckResult,
  type MemoryUsageEstimate,
} from "./memory-estimation.js";
export {
  LlamaCppLanguageModel,
  type LlamaCppModelConfig,
  type LlamaCppGenerationConfig,
  // Exported for testing
  convertMessages,
  convertFinishReason,
  convertUsage,
  resolveReasoningConfig,
  splitReasoningContent,
  // Tool calling exports
  generateToolCallGrammar,
  parseToolCalls,
  buildToolSystemPrompt,
  type ParsedToolCall,
  type ParsedReasoningPart,
} from "./llama-cpp-language-model.js";
export { LlamaCppEmbeddingModel } from "./llama-cpp-embedding-model.js";

// Export JSON schema to grammar converter for advanced use cases
export {
  convertJsonSchemaToGrammar,
  SchemaConverter,
  type SchemaConverterOptions,
} from "./json-schema-to-grammar.js";

// Re-export JSONSchema7 from @ai-sdk/provider for convenience
export type { JSONSchema7 } from "@ai-sdk/provider";

// Default export
export { default } from "./llama-cpp-provider.js";
