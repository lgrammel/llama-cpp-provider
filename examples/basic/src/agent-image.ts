import { ToolLoopAgent } from "ai";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";
import {
  exampleContextSize,
  exampleModel,
  mmprojPath,
  modelPath,
} from "./example-model.js";
import { reportError } from "./report-error.js";

const imagePath = process.argv[2];

if (!imagePath) {
  throw new Error("Usage: pnpm agent-image <path-to-image>");
}

for (const [label, path] of [
  ["model", modelPath],
  ["mmproj", mmprojPath],
  ["image", imagePath],
] as const) {
  if (!existsSync(path)) {
    throw new Error(`Missing ${label} file: ${path}`);
  }
}

const model = llamaCpp({
  modelPath,
  mmprojPath,
  contextSize: exampleContextSize,
  model: {
    chatTemplate: exampleModel.chatTemplate,
  },
});

const agent = new ToolLoopAgent({
  model,
  instructions:
    "You are a concise assistant. Describe images accurately when users attach them.",
  maxOutputTokens: 128,
});

try {
  const result = await agent.generate({
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Describe this image in one short sentence." },
          {
            type: "file",
            data: {
              type: "data",
              data: await readFile(imagePath),
            },
            mediaType: "image/png",
          },
        ],
      },
    ],
  });

  console.log(result.text);
} catch (error) {
  reportError(error, modelPath, mmprojPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
