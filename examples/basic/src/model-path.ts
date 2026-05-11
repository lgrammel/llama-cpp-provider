import { homedir } from "node:os";
import { join } from "node:path";
import { gemma4_31b_it } from "ai-sdk-llama-cpp";

export const modelPath = join(
  homedir(),
  "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
);

export const modelOptions = {
  modelPath,
  model: gemma4_31b_it,
};
