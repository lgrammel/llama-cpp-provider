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
  cache: { mode: "prefix" },
});

type Prompt = Parameters<typeof model.doGenerate>[0]["prompt"];
type GenerateResult = Awaited<ReturnType<typeof model.doGenerate>>;

function textFromResult(content: GenerateResult["content"]): string {
  return content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function printUsage(label: string, result: GenerateResult): void {
  const input = result.usage.inputTokens;

  console.log(`${label} input tokens:`);
  console.log(`  total: ${input.total}`);
  console.log(`  cache read: ${input.cacheRead ?? 0}`);
  console.log(`  cache write: ${input.cacheWrite ?? 0}`);
  console.log(`  no cache: ${input.noCache ?? input.total}`);
  console.log(`  output: ${result.usage.outputTokens.total}`);
}

try {
  const firstMessages: Prompt = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text:
            "Give me a concise three bullet overview of why prompt caching " +
            "can speed up local chat inference.",
        },
      ],
    },
  ];

  const first = await model.doGenerate({
    prompt: firstMessages,
    maxOutputTokens: 160,
    temperature: 0,
  });
  const firstText = textFromResult(first.content);

  console.log("\nFirst response:\n");
  console.log(firstText);
  console.log();
  printUsage("First turn", first);

  const secondMessages: Prompt = [
    ...firstMessages,
    {
      role: "assistant",
      content: [{ type: "text", text: firstText }],
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "Now summarize that in one sentence.",
        },
      ],
    },
  ];

  const second = await model.doGenerate({
    prompt: secondMessages,
    maxOutputTokens: 80,
    temperature: 0,
  });
  const secondText = textFromResult(second.content);

  console.log("\nSecond response:\n");
  console.log(secondText);
  console.log();
  printUsage("Second turn", second);
} catch (error) {
  reportError(error, modelPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
