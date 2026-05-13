import { generateText, Output } from "ai";
import { z } from "zod";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";
import {
  exampleContextSize,
  exampleModel,
  modelPath,
} from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  contextSize: exampleContextSize,
  model: exampleModel,
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
