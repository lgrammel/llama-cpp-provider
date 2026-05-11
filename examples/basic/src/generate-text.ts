import { generateText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";
import { exampleModel, modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  model: exampleModel,
});

try {
  const result = await generateText({
    model,
    prompt: "Invent a new holiday and describe its traditions.",
  });

  console.log(result.text);
  console.log();
  console.log("Usage:", result.usage);
  console.log("Finish reason:", result.finishReason);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
