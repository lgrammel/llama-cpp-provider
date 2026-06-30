import { describe, expect, it } from "vitest";

const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
const TEST_EMBEDDING_PATH = process.env.TEST_EMBEDDING_PATH;
const TEST_MMPROJ_PATH = process.env.TEST_MMPROJ_PATH;
const TEST_IMAGE_PATH = process.env.TEST_IMAGE_PATH;
const TEST_CHAT_TEMPLATE = process.env.TEST_CHAT_TEMPLATE;

describe("E2E test configuration", () => {
  it("prints a consolidated capability report", () => {
    const capabilities = [
      {
        name: "language",
        ready: Boolean(TEST_MODEL_PATH),
        required: ["TEST_MODEL_PATH"],
      },
      {
        name: "structured-output",
        ready: Boolean(TEST_MODEL_PATH),
        required: ["TEST_MODEL_PATH"],
      },
      {
        name: "tool-calling",
        ready: Boolean(TEST_MODEL_PATH),
        required: ["TEST_MODEL_PATH"],
      },
      {
        name: "embedding",
        ready: Boolean(TEST_EMBEDDING_PATH),
        required: ["TEST_EMBEDDING_PATH"],
      },
      {
        name: "image-input",
        ready: Boolean(TEST_MODEL_PATH && TEST_MMPROJ_PATH && TEST_IMAGE_PATH),
        required: ["TEST_MODEL_PATH", "TEST_MMPROJ_PATH", "TEST_IMAGE_PATH"],
      },
    ];

    console.log("\nE2E capability report");
    for (const capability of capabilities) {
      const missing = capability.required.filter((name) => !process.env[name]);
      console.log(
        `- ${capability.name}: ${capability.ready ? "ready" : `skipped; missing ${missing.join(", ")}`}`
      );
    }

    if (TEST_MODEL_PATH) {
      console.log(`- language model: ${TEST_MODEL_PATH}`);
    }

    if (TEST_EMBEDDING_PATH) {
      console.log(`- embedding model: ${TEST_EMBEDDING_PATH}`);
    }

    if (TEST_CHAT_TEMPLATE) {
      console.log(`- chat template override: ${TEST_CHAT_TEMPLATE}`);
    }

    if (!TEST_MODEL_PATH) {
      console.log(
        "- agent smoke command: pnpm test:e2e:agent downloads/caches a default GGUF model"
      );
    }

    expect(true).toBe(true);
  });
});
