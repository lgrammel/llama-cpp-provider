import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";
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
    "You are a concise assistant. Show the final answer clearly after reasoning.",
});

try {
  await runAgentTUI({
    name: "Reasoning agent",
    agent,
    reasoning: "full",
  });
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
