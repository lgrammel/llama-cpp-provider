import type { LlamaCppProviderConfig } from "@lgrammel/llama-cpp-provider";

export const TEST_CHAT_TEMPLATE = process.env.TEST_CHAT_TEMPLATE;
export const TEST_E2E_CONTEXT_SIZE = readPositiveInteger(
  "TEST_E2E_CONTEXT_SIZE"
);
export const TEST_E2E_GPU_LAYERS = readInteger("TEST_E2E_GPU_LAYERS");
export const TEST_E2E_THREADS = readPositiveInteger("TEST_E2E_THREADS");

export function modelRuntimeConfig(defaults?: {
  contextSize?: number;
  gpuLayers?: number;
  threads?: number;
}): Pick<LlamaCppProviderConfig, "contextSize" | "gpuLayers" | "threads"> {
  return {
    contextSize: TEST_E2E_CONTEXT_SIZE ?? defaults?.contextSize ?? 2048,
    gpuLayers: TEST_E2E_GPU_LAYERS ?? defaults?.gpuLayers ?? 0,
    threads: TEST_E2E_THREADS ?? defaults?.threads ?? 4,
  };
}

export function languageModelConfig(
  config: LlamaCppProviderConfig
): LlamaCppProviderConfig {
  if (!TEST_CHAT_TEMPLATE || config.model?.chatTemplate) {
    return config;
  }

  return {
    ...config,
    model: {
      ...config.model,
      chatTemplate: config.model?.chatTemplate ?? TEST_CHAT_TEMPLATE,
    },
  };
}

export function formatModelInfo(modelPath: string): string {
  return TEST_CHAT_TEMPLATE
    ? `${modelPath} (chatTemplate: ${TEST_CHAT_TEMPLATE})`
    : modelPath;
}

function readInteger(name: string): number | undefined {
  const value = process.env[name];
  if (!value) {
    return undefined;
  }

  const parsed = Number(value);
  if (!Number.isInteger(parsed)) {
    throw new Error(`${name} must be an integer, got ${value}`);
  }

  return parsed;
}

function readPositiveInteger(name: string): number | undefined {
  const parsed = readInteger(name);
  if (parsed === undefined) {
    return undefined;
  }

  if (parsed <= 0) {
    throw new Error(`${name} must be greater than 0, got ${parsed}`);
  }

  return parsed;
}
