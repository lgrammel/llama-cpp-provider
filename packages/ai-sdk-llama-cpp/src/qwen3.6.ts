import type { LlamaCppModelInfo } from "./llama-cpp-provider-config.js";

export const qwen3_6_dense: LlamaCppModelInfo = {
  chatTemplate: "chatml",
  memory: {
    maxContextSize: 262144,
    kvCache: {
      bytesPerValue: 2,
      layers: [
        {
          count: 16,
          keyValueHeads: 4,
          headDim: 256,
        },
        {
          count: 48,
          keyHeads: 16,
          valueHeads: 48,
          keyHeadDim: 128,
          valueHeadDim: 128,
        },
      ],
    },
  },
};

export const qwen3_6_moe: LlamaCppModelInfo = {
  chatTemplate: "chatml",
  memory: {
    maxContextSize: 262144,
    kvCache: {
      bytesPerValue: 2,
      layers: [
        {
          count: 10,
          keyValueHeads: 2,
          headDim: 256,
        },
        {
          count: 30,
          keyHeads: 16,
          valueHeads: 32,
          keyHeadDim: 128,
          valueHeadDim: 128,
        },
      ],
    },
  },
};
