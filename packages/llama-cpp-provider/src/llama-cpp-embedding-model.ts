import type {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
  SharedV4Warning,
} from "@ai-sdk/provider";
import { stat } from "node:fs/promises";

import {
  loadModel,
  unloadModel,
  embed,
  isModelLoaded,
  type LoadModelOptions,
  type EmbedOptions,
} from "./native-binding.js";
import type { LlamaCppProviderConfig } from "./llama-cpp-provider-config.js";
import { checkMemorySafety } from "./memory-estimation.js";

export class LlamaCppEmbeddingModel implements EmbeddingModelV4 {
  readonly specificationVersion = "v4" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;

  /**
   * Maximum number of embeddings that can be generated in a single call.
   * Local models can handle large batches, but we limit to prevent memory issues.
   */
  readonly maxEmbeddingsPerCall: number = 2048;

  /**
   * Whether the model supports parallel calls.
   * We use a single model instance, so parallel calls are not supported.
   */
  readonly supportsParallelCalls: boolean = false;

  private modelHandle: number | null = null;
  private readonly config: LlamaCppProviderConfig;
  private initPromise: Promise<void> | null = null;

  constructor(config: LlamaCppProviderConfig) {
    this.config = config;
    this.modelId = config.modelPath;
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
      const modelFileSizeBytes = await getFileSize(this.config.modelPath);
      const memorySafety = checkMemorySafety({
        model: this.config.model?.memory,
        contextSize: this.config.contextSize ?? 2048,
        modelFileSizeBytes,
        memorySafety: this.config.memorySafety,
      });
      const options: LoadModelOptions = {
        modelPath: this.config.modelPath,
        contextSize: memorySafety.contextSize,
        gpuLayers: this.config.gpuLayers ?? 99,
        threads: this.config.threads ?? 4,
        debug: this.config.debug ?? false,
        embedding: true,
      };

      this.modelHandle = await loadModel(options);
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
      throw new Error("Failed to load embedding model");
    }

    return this.modelHandle;
  }

  /**
   * Dispose of the model and free resources.
   */
  async dispose(): Promise<void> {
    if (this.modelHandle !== null) {
      unloadModel(this.modelHandle);
      this.modelHandle = null;
    }
  }

  async doEmbed(
    options: EmbeddingModelV4CallOptions
  ): Promise<EmbeddingModelV4Result> {
    const handle = await this.ensureModelLoaded();

    const embedOptions: EmbedOptions = {
      texts: options.values,
    };

    const result = await embed(handle, embedOptions);

    // Convert Float32Array[] to number[][]
    const embeddings: number[][] = result.embeddings.map((embedding) =>
      Array.from(embedding)
    );

    const warnings: SharedV4Warning[] = [];

    return {
      embeddings,
      usage: {
        tokens: result.totalTokens,
      },
      warnings,
    };
  }
}

async function getFileSize(path: string): Promise<number | undefined> {
  try {
    return (await stat(path)).size;
  } catch {
    return undefined;
  }
}
