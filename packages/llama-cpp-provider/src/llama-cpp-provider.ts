import {
  LlamaCppLanguageModel,
  type LlamaCppModelConfig,
} from "./llama-cpp-language-model.js";
import { LlamaCppEmbeddingModel } from "./llama-cpp-embedding-model.js";
import type { LlamaCppProviderConfig } from "./llama-cpp-provider-config.js";

export interface LlamaCppProvider {
  (config: LlamaCppProviderConfig): LlamaCppLanguageModel;
  languageModel(config: LlamaCppProviderConfig): LlamaCppLanguageModel;
  embedding(config: LlamaCppProviderConfig): LlamaCppEmbeddingModel;
}

function createLlamaCpp(): LlamaCppProvider {
  const provider = (config: LlamaCppProviderConfig): LlamaCppLanguageModel => {
    const modelConfig: LlamaCppModelConfig = {
      modelPath: config.modelPath,
      mmprojPath: config.mmprojPath,
      contextSize: config.contextSize,
      memorySafety: config.memorySafety,
      memory: config.model?.memory,
      gpuLayers: config.gpuLayers,
      threads: config.threads,
      debug: config.debug,
      cache: config.cache,
      chatTemplate: config.model?.chatTemplate,
      reasoning: config.model?.reasoning,
    };

    return new LlamaCppLanguageModel(modelConfig);
  };

  provider.languageModel = provider;

  provider.embedding = (config: LlamaCppProviderConfig) => {
    return new LlamaCppEmbeddingModel(config);
  };

  return provider as LlamaCppProvider;
}

/**
 * Creates a llama.cpp model provider.
 *
 * @example
 * ```typescript
 * import { llamaCpp } from '@lgrammel/llama-cpp-provider';
 * import { ToolLoopAgent, tool } from 'ai';
 * import { z } from 'zod';
 *
 * const model = llamaCpp({
 *   modelPath: './models/llama-3.2-1b.gguf'
 * });
 *
 * const agent = new ToolLoopAgent({
 *   model,
 *   instructions: 'You are a concise assistant.',
 *   tools: {
 *     weather: tool({
 *       description: 'Get the weather in a location',
 *       inputSchema: z.object({ location: z.string() }),
 *       execute: async ({ location }) => ({ location, temperature: 72 }),
 *     }),
 *   },
 * });
 * ```
 */
export const llamaCpp = createLlamaCpp();

/**
 * Default export for convenience.
 */
export default llamaCpp;
