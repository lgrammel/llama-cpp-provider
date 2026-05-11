import { generateText, Output } from "ai";
import { z } from "zod";
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
  const recipeSchema = z.object({
    name: z.string(),
    ingredients: z.array(
      z.object({
        name: z.string(),
        amount: z.string(),
      })
    ),
    steps: z.array(z.string()),
  });

  const result = await generateText({
    model,
    prompt: "Generate a lasagna recipe.",
    output: Output.object({ schema: recipeSchema }),
  });

  console.log(JSON.stringify(result.output, null, 2));
  console.log();
  console.log("Usage:", result.usage);
  console.log("Finish reason:", result.finishReason);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
