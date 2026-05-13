import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";
import { gemma4_31b_it, llamaCpp } from "@lgrammel/llama-cpp-provider";
import { modelPath } from "./example-model.js";
import { reportError } from "./report-error.js";
import { homedir } from "node:os";
import { join } from "node:path";

const gemma4 = llamaCpp({
  modelPath: join(
    homedir(),
    "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
  ),
  contextSize: 100000,
  memorySafety: { mode: "off" },
  model: gemma4_31b_it,
});

const agent = new ToolLoopAgent({
  model: gemma4,
  instructions:
    "You are a concise assistant." +
    "Answer in markdown when it improves readability.",
});

try {
  await runAgentTUI({ name: "Gemma 4 31B", agent });
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await gemma4.dispose();
}
