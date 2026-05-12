import { streamText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";
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
