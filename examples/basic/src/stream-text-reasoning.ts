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
  model: {
    ...gemma4_31b_it,
    reasoning: {},
  },
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
