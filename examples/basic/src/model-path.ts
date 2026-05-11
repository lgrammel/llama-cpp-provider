import { homedir } from "node:os";
import { join } from "node:path";

export const modelPath = join(
  homedir(),
  "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
);

export const modelOptions = {
  modelPath,
  model: {
    chatTemplate: "gemma",
    reasoning: {
      promptPrefix: false as const,
    },
  },
};
