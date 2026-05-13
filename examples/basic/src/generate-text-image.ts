import { generateText } from "ai";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
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
  throw new Error("Usage: pnpm generate-text-image <path-to-image>");
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

console.log("Loading model:", modelPath);
console.log("Loading mmproj:", mmprojPath);
console.log("Reading image:", imagePath);

const model = llamaCpp({
  modelPath,
  mmprojPath,
  contextSize: exampleContextSize,
  model: {
    chatTemplate: exampleModel.chatTemplate,
  },
});

try {
  const { text } = await generateText({
    model,
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
    maxOutputTokens: 128,
  });

  console.log(text);
} catch (error) {
  reportError(error, modelPath, mmprojPath);
  process.exitCode = 1;
} finally {
  await model.dispose();
}
