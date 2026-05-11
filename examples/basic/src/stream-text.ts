import { streamText } from "ai";
import { gemma4_31b_it, llamaCpp } from "ai-sdk-llama-cpp";
import { homedir } from "node:os";
import { join } from "node:path";
import { reportError } from "./report-error.js";

const modelPath = join(
  homedir(),
  "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
);

const model = llamaCpp({
  modelPath,
  model: gemma4_31b_it,
});

try {
  const result = streamText({
    model,
    prompt: "Invent a new holiday and describe its traditions.",
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  console.log();
  console.log();
  console.log("Usage:", await result.usage);
  console.log("Finish reason:", await result.finishReason);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
