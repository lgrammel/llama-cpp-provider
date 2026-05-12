import type {
  LlamaCppModelInfo,
  LlamaCppModelMemoryInfo,
  LlamaCppReasoningConfig,
} from "./llama-cpp-provider-config.js";

export const gemma4Reasoning: LlamaCppReasoningConfig = {
  openingMarker: "<|channel>thought\n",
  closingMarker: "<channel|>",
  promptPrefix: "<|think|>\n",
};

const gemma4_31b_it_memory: LlamaCppModelMemoryInfo = {
  maxContextSize: 262144,
  kvCache: {
    bytesPerValue: 2,
    layers: [
      {
        count: 50,
        keyValueHeads: 16,
        headDim: 256,
      },
      {
        count: 10,
        keyValueHeads: 4,
        headDim: 512,
      },
    ],
  },
};

const gemma4_26b_a4b_memory: LlamaCppModelMemoryInfo = {
  maxContextSize: 262144,
  kvCache: {
    bytesPerValue: 2,
    layers: [
      {
        count: 25,
        keyValueHeads: 8,
        headDim: 256,
      },
      {
        count: 5,
        keyValueHeads: 2,
        headDim: 512,
      },
    ],
  },
};

export const gemma4_31b_it: LlamaCppModelInfo = {
  chatTemplate: "gemma",
  reasoning: gemma4Reasoning,
  memory: gemma4_31b_it_memory,
};

export const gemma4_26b_a4b: LlamaCppModelInfo = {
  chatTemplate: "gemma",
  reasoning: gemma4Reasoning,
  memory: gemma4_26b_a4b_memory,
};
