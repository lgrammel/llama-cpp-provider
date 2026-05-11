import { streamText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";
import { exampleModel, modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  model: exampleModel,
});

try {
  const result = streamText({
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

  for await (const part of result.fullStream) {
    console.log(JSON.stringify(part));
  }

  console.log("\nFinal text:", await result.text);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
