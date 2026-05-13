import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent, tool } from "ai";
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

const agent = new ToolLoopAgent({
  model,
  instructions:
    "You are a concise assistant. Use the weather tool when users ask about weather.",
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
});

try {
  await runAgentTUI({ name: "Weather agent", agent, tools: "full" });
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
