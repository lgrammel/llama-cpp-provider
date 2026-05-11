import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent, stepCountIs, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";
import { exampleModel, modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  model: exampleModel,
});

try {
  await runAgentTUI({
    name: "Llama.cpp Weather Agent",
    agent: new ToolLoopAgent({
      model,
      instructions:
        "You are a concise weather assistant. Use the weather tool when the user asks about weather, then answer in markdown.",
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
    }),
  });
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
