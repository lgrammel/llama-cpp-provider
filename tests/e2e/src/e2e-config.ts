import type { LlamaCppProviderConfig } from "@lgrammel/llama-cpp-provider";

export const TEST_CHAT_TEMPLATE = process.env.TEST_CHAT_TEMPLATE;

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
