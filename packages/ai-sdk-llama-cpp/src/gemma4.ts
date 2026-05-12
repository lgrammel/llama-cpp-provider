import type {
  LlamaCppModelInfo,
  LlamaCppReasoningConfig,
} from "./llama-cpp-provider-config.js";

export const gemma4Reasoning: LlamaCppReasoningConfig = {
  openingMarker: "<|channel>thought\n",
  closingMarker: "<channel|>",
  promptPrefix: "<|think|>\n",
};

export const gemma4_31b_it: LlamaCppModelInfo = {
  chatTemplate: "gemma",
  reasoning: gemma4Reasoning,
};

export const gemma4_26b_a4b: LlamaCppModelInfo = {
  chatTemplate: "gemma",
  reasoning: gemma4Reasoning,
};
