import { embedMany } from "ai";
import { gemma4_31b_it, llamaCpp } from "ai-sdk-llama-cpp";
import { homedir } from "node:os";
import { join } from "node:path";
import { reportError } from "./report-error.js";

const modelPath = join(
  homedir(),
  "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
);

const model = llamaCpp.embedding({
  modelPath,
  model: gemma4_31b_it,
});

try {
  const result = await embedMany({
    model,
    values: ["sunny day at the beach", "rainy afternoon in the city"],
  });

  console.log("Embeddings:", result.embeddings.length);
  console.log("Dimensions:", result.embeddings[0].length);
  console.log();
  console.log("Usage:", result.usage);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
