import { gemma4_31b_it } from "ai-sdk-llama-cpp";
import { homedir } from "node:os";
import { join } from "node:path";

export const modelPath = join(
  homedir(),
  "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
);

export const exampleModel = {
  ...gemma4_31b_it,
};

export const exampleContextSize = 4096;
