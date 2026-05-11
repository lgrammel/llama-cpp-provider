import { gemma4_31b_it, llamaCpp } from "ai-sdk-llama-cpp";
import { stepCountIs, ModelMessage, streamText, tool } from "ai";
import { homedir } from "node:os";
import { join } from "node:path";
import * as readline from "node:readline/promises";
import { z } from "zod";
import { reportError } from "./report-error.js";

const modelPath = join(
  homedir(),
  "opt/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf"
);

const model = llamaCpp({
  modelPath,
  model: gemma4_31b_it,
});

const terminal = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const messages: ModelMessage[] = [];

try {
  while (true) {
    const userInput = await terminal.question("You: ");

    messages.push({ role: "user", content: userInput });

    const result = streamText({
      model,
      onError(error) {
        console.error(error);
      },
      system: `You are a helpful, respectful and honest assistant.`,
      tools: {
        weather: tool({
          description: "Get the weather in a location",
          inputSchema: z.object({
            location: z
              .string()
              .describe("The location to get the weather for"),
          }),
          execute: async ({ location }) => ({
            location,
            temperature: 72 + Math.floor(Math.random() * 21) - 10,
          }),
        }),
      },
      stopWhen: stepCountIs(5),
      messages,
    });

    process.stdout.write("\nAssistant: ");
    for await (const delta of result.textStream) {
      process.stdout.write(delta);
    }
    process.stdout.write("\n\n");

    messages.push(...(await result.response).messages);
  }
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
  terminal.close();
}
