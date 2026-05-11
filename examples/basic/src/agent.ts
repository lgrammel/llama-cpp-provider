import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";
import { gemma4Reasoning, llamaCpp } from "ai-sdk-llama-cpp";
import { modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  model: {
    contextSize: 65536,
    chatTemplate: "gemma",
    reasoning: gemma4Reasoning,
  },
});

try {
  await runAgentTUI({
    name: "Gemma 4 31B",
    agent: new ToolLoopAgent({
      model,
      instructions:
        "You are a concise assistant. Answer in markdown when it improves readability.",
    }),
  });
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
