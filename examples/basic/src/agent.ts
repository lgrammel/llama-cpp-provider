import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";
import { gemma4_31b_it, llamaCpp } from "ai-sdk-llama-cpp";
import { modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";

const model = llamaCpp({
  modelPath,
  model: gemma4_31b_it,
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
