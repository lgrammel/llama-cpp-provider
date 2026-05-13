import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";
import { gemma4Reasoning, llamaCpp } from "@lgrammel/llama-cpp-provider";
import { modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  contextSize: 65536,
  model: {
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
