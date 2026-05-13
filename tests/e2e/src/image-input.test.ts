import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { readFile } from "node:fs/promises";
import { generateText } from "ai";
import { llamaCpp, LlamaCppLanguageModel } from "@lgrammel/llama-cpp-provider";

const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
const TEST_MMPROJ_PATH = process.env.TEST_MMPROJ_PATH;
const TEST_IMAGE_PATH = process.env.TEST_IMAGE_PATH;
const shouldRunTests =
  !!TEST_MODEL_PATH && !!TEST_MMPROJ_PATH && !!TEST_IMAGE_PATH;

const describeE2E = shouldRunTests ? describe : describe.skip;

describeE2E("E2E Image Input Tests", () => {
  let model: LlamaCppLanguageModel;

  beforeAll(() => {
    if (!TEST_MODEL_PATH || !TEST_MMPROJ_PATH) {
      throw new Error(
        "TEST_MODEL_PATH and TEST_MMPROJ_PATH environment variables must be set"
      );
    }

    model = llamaCpp({
      modelPath: TEST_MODEL_PATH,
      mmprojPath: TEST_MMPROJ_PATH,
      contextSize: 4096,
      gpuLayers: 0,
      threads: 4,
    });
  });

  afterAll(async () => {
    if (model) {
      await model.dispose();
    }
  });

  it("generates text from an image input", { timeout: 180000 }, async () => {
    if (!TEST_IMAGE_PATH) {
      throw new Error("TEST_IMAGE_PATH environment variable must be set");
    }

    const { text } = await generateText({
      model,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe this image in one sentence." },
            {
              type: "file",
              data: {
                type: "data",
                data: await readFile(TEST_IMAGE_PATH),
              },
              mediaType: "image/png",
            },
          ],
        },
      ],
      maxOutputTokens: 80,
    });

    expect(text.length).toBeGreaterThan(0);
  });
});

describe("E2E Image Input Test Configuration", () => {
  it("image input environment variable info", () => {
    if (!shouldRunTests) {
      console.log(
        "\nImage input E2E tests skipped: set TEST_MODEL_PATH, TEST_MMPROJ_PATH, and TEST_IMAGE_PATH to run with a real multimodal model\n"
      );
    }

    expect(true).toBe(true);
  });
});
