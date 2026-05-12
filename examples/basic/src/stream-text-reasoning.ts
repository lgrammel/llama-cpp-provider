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
    prompt: "Solve 17 * 23 and explain the calculation briefly.",
  });

  let reasoningStarted = false;
  let answerStarted = false;

  for await (const part of result.fullStream) {
    switch (part.type) {
      case "reasoning-delta": {
        if (!reasoningStarted) {
          console.log("Reasoning:");
          reasoningStarted = true;
        }
        process.stdout.write(part.text);
        break;
      }
      case "text-delta": {
        if (!answerStarted) {
          if (reasoningStarted) {
            console.log();
            console.log();
          }
          console.log("Answer:");
          answerStarted = true;
        }
        process.stdout.write(part.text);
        break;
      }
      case "error": {
        throw part.error;
      }
    }
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
