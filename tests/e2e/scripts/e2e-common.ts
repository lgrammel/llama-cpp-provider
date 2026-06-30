import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { access, mkdir, rename, stat, writeFile } from "node:fs/promises";
import { once } from "node:events";
import { Readable } from "node:stream";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type IssueLevel = "error" | "warning" | "info";

export interface E2EIssue {
  level: IssueLevel;
  message: string;
  hint?: string;
}

export interface E2EReadiness {
  ok: boolean;
  repoRoot: string;
  e2eRoot: string;
  modelPath?: string;
  modelSource?: "env" | "cache" | "download";
  embeddingPath?: string;
  imageInput: {
    modelPath?: string;
    mmprojPath?: string;
    imagePath?: string;
    ready: boolean;
  };
  agent: {
    modelUrl: string;
    modelSha256?: string;
    cacheDir: string;
    resultPath: string;
  };
  issues: E2EIssue[];
}

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

export const e2eRoot = resolve(SCRIPT_DIR, "..");
export const repoRoot = resolve(e2eRoot, "../..");
export const resultDir = join(repoRoot, "test-results");
export const agentResultPath = join(resultDir, "e2e-agent.json");
export const agentModelCacheDir = join(repoRoot, "models", "e2e-cache");

export const DEFAULT_AGENT_MODEL_URL =
  "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";

export function getAgentModelUrl(): string {
  return process.env.TEST_AGENT_MODEL_URL ?? DEFAULT_AGENT_MODEL_URL;
}

export function getAgentModelSha256(): string | undefined {
  const value = process.env.TEST_AGENT_MODEL_SHA256?.trim();
  return value ? value.toLowerCase() : undefined;
}

export function resolveRepoPath(path: string): string {
  return resolve(repoRoot, path);
}

export async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export async function sha256File(path: string): Promise<string> {
  const { createReadStream } = await import("node:fs");
  const hash = createHash("sha256");
  const stream = createReadStream(path);

  for await (const chunk of stream) {
    hash.update(chunk);
  }

  return hash.digest("hex");
}

export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

export async function ensureAgentModel(): Promise<{
  path: string;
  source: "env" | "cache" | "download";
  sha256?: string;
}> {
  if (process.env.TEST_MODEL_PATH) {
    const modelPath = resolveRepoPath(process.env.TEST_MODEL_PATH);
    await assertReadableFile(modelPath, "TEST_MODEL_PATH");
    return { path: modelPath, source: "env" };
  }

  const url = getAgentModelUrl();
  const expectedSha256 = getAgentModelSha256();
  const filename = getFilenameFromUrl(url);
  const modelPath = join(agentModelCacheDir, filename);

  if (await pathExists(modelPath)) {
    if (expectedSha256) {
      await verifySha256(modelPath, expectedSha256);
    }

    return { path: modelPath, source: "cache", sha256: expectedSha256 };
  }

  await mkdir(agentModelCacheDir, { recursive: true });
  await downloadFile(url, modelPath, expectedSha256);
  return { path: modelPath, source: "download", sha256: expectedSha256 };
}

export async function collectReadiness(options?: {
  prepareModel?: boolean;
}): Promise<E2EReadiness> {
  const issues: E2EIssue[] = [];
  let modelPath: string | undefined;
  let modelSource: E2EReadiness["modelSource"];

  if (process.platform !== "darwin") {
    issues.push({
      level: "error",
      message: `Unsupported platform: ${process.platform}`,
      hint: "llama-cpp-provider E2E tests currently require macOS.",
    });
  }

  const nodeMajor = Number(process.versions.node.split(".")[0]);
  if (nodeMajor < 22) {
    issues.push({
      level: "error",
      message: `Unsupported Node.js version: ${process.versions.node}`,
      hint: "Use Node.js 22 or newer.",
    });
  }

  await addFileCheck(
    issues,
    join(repoRoot, "packages/llama-cpp-provider/dist/index.js"),
    "TypeScript build output is missing",
    "Run pnpm build:ts before E2E tests."
  );

  await addFileCheck(
    issues,
    join(
      repoRoot,
      "packages/llama-cpp-provider/build/Release/llama_binding.node"
    ),
    "Native addon is missing",
    "Run pnpm build:native or pnpm install before E2E tests."
  );

  try {
    if (options?.prepareModel) {
      const prepared = await ensureAgentModel();
      modelPath = prepared.path;
      modelSource = prepared.source;

      if (!prepared.sha256) {
        issues.push({
          level: "warning",
          message: "Agent model checksum verification is disabled",
          hint: "Set TEST_AGENT_MODEL_SHA256 to pin the downloaded GGUF file by content.",
        });
      }
    } else if (process.env.TEST_MODEL_PATH) {
      modelPath = resolveRepoPath(process.env.TEST_MODEL_PATH);
      await assertReadableFile(modelPath, "TEST_MODEL_PATH");
      modelSource = "env";
    } else {
      issues.push({
        level: "warning",
        message: "TEST_MODEL_PATH is not set",
        hint: "Run pnpm test:e2e:agent to download/cache a smoke-test model, or set TEST_MODEL_PATH manually.",
      });
    }
  } catch (error) {
    issues.push({
      level: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }

  const embeddingPath = process.env.TEST_EMBEDDING_PATH
    ? resolveRepoPath(process.env.TEST_EMBEDDING_PATH)
    : undefined;
  if (embeddingPath && !(await pathExists(embeddingPath))) {
    issues.push({
      level: "error",
      message: `TEST_EMBEDDING_PATH does not exist: ${embeddingPath}`,
    });
  }

  const imageModelPath = process.env.TEST_MODEL_PATH
    ? resolveRepoPath(process.env.TEST_MODEL_PATH)
    : undefined;
  const mmprojPath = process.env.TEST_MMPROJ_PATH
    ? resolveRepoPath(process.env.TEST_MMPROJ_PATH)
    : undefined;
  const imagePath = process.env.TEST_IMAGE_PATH
    ? resolveRepoPath(process.env.TEST_IMAGE_PATH)
    : undefined;

  const imageInput = {
    modelPath: imageModelPath,
    mmprojPath,
    imagePath,
    ready: Boolean(imageModelPath && mmprojPath && imagePath),
  };

  const hasErrors = issues.some((issue) => issue.level === "error");

  return {
    ok: !hasErrors,
    repoRoot,
    e2eRoot,
    modelPath,
    modelSource,
    embeddingPath,
    imageInput,
    agent: {
      modelUrl: getAgentModelUrl(),
      modelSha256: getAgentModelSha256(),
      cacheDir: agentModelCacheDir,
      resultPath: agentResultPath,
    },
    issues,
  };
}

export function printReadiness(readiness: E2EReadiness): void {
  console.log(`E2E readiness: ${readiness.ok ? "ok" : "blocked"}`);
  console.log(`Repo: ${readiness.repoRoot}`);

  if (readiness.modelPath) {
    console.log(`Language model: ${readiness.modelPath}`);
    console.log(`Language model source: ${readiness.modelSource}`);
  }

  if (readiness.embeddingPath) {
    console.log(`Embedding model: ${readiness.embeddingPath}`);
  }

  console.log(
    `Image input: ${readiness.imageInput.ready ? "configured" : "not configured"}`
  );

  for (const issue of readiness.issues) {
    const prefix =
      issue.level === "error"
        ? "ERROR"
        : issue.level === "warning"
          ? "WARN"
          : "INFO";
    console.log(`${prefix}: ${issue.message}`);
    if (issue.hint) {
      console.log(`      ${issue.hint}`);
    }
  }

  console.log(`JSON summary: ${readiness.agent.resultPath}`);
}

async function addFileCheck(
  issues: E2EIssue[],
  path: string,
  message: string,
  hint: string
): Promise<void> {
  if (!(await pathExists(path))) {
    issues.push({ level: "error", message: `${message}: ${path}`, hint });
  }
}

async function assertReadableFile(path: string, label: string): Promise<void> {
  const info = await stat(path);
  if (!info.isFile()) {
    throw new Error(`${label} is not a file: ${path}`);
  }
}

async function verifySha256(
  path: string,
  expectedSha256: string
): Promise<void> {
  const actualSha256 = await sha256File(path);
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `Checksum mismatch for ${path}: expected ${expectedSha256}, got ${actualSha256}`
    );
  }
}

async function downloadFile(
  url: string,
  targetPath: string,
  expectedSha256: string | undefined
): Promise<void> {
  console.log(`Downloading E2E model: ${url}`);
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to download E2E model from ${url}: ${response.status} ${response.statusText}`
    );
  }

  const temporaryPath = `${targetPath}.tmp`;
  const file = createWriteStream(temporaryPath);
  const hash = createHash("sha256");
  let bytes = 0;

  try {
    for await (const chunk of Readable.fromWeb(response.body)) {
      const buffer = Buffer.from(chunk);
      bytes += buffer.length;
      hash.update(buffer);
      if (!file.write(buffer)) {
        await once(file, "drain");
      }
    }
  } catch (error) {
    file.destroy();
    throw error;
  }

  file.end();
  await once(file, "finish");

  const actualSha256 = hash.digest("hex");
  if (expectedSha256 && actualSha256 !== expectedSha256) {
    throw new Error(
      `Checksum mismatch for downloaded model: expected ${expectedSha256}, got ${actualSha256}`
    );
  }

  await rename(temporaryPath, targetPath);
  console.log(`Downloaded ${bytes} bytes to ${targetPath}`);
}

function getFilenameFromUrl(url: string): string {
  const parsed = new URL(url);
  const filename = parsed.pathname.split("/").filter(Boolean).at(-1);
  if (!filename || !filename.endsWith(".gguf")) {
    throw new Error(`TEST_AGENT_MODEL_URL must point to a .gguf file: ${url}`);
  }

  return filename;
}
