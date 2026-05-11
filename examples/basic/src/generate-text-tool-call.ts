import { generateText, stepCountIs, tool } from "ai";
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
  const result = await generateText({
    model,
    prompt: "What is the weather in Tokyo?",
    tools: {
      weather: tool({
        description: "Get the weather in a location",
        inputSchema: z.object({
          location: z.string().describe("The location to get weather for"),
        }),
        execute: async ({ location }) => ({
          location,
          temperature: 72,
        }),
      }),
    },
    stopWhen: stepCountIs(3),
  });

  console.log(result.text);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
