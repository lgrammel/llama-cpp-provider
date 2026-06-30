import { spawn } from "node:child_process";
import {
  collectReadiness,
  e2eRoot,
  printReadiness,
  writeJson,
} from "./e2e-common.js";

const startedAt = new Date().toISOString();
const readiness = await collectReadiness({ prepareModel: true });

printReadiness(readiness);

if (!readiness.ok || !readiness.modelPath) {
  await writeJson(readiness.agent.resultPath, {
    command: "test:e2e:agent",
    startedAt,
    finishedAt: new Date().toISOString(),
    exitCode: 1,
    readiness,
  });
  process.exit(1);
}

const testEnv = {
  ...process.env,
  TEST_MODEL_PATH: readiness.modelPath,
  TEST_E2E_AGENT: "1",
  TEST_E2E_CONTEXT_SIZE: process.env.TEST_E2E_CONTEXT_SIZE ?? "1024",
  TEST_E2E_GPU_LAYERS: process.env.TEST_E2E_GPU_LAYERS ?? "0",
  TEST_E2E_THREADS: process.env.TEST_E2E_THREADS ?? "2",
};

const vitestArgs = ["exec", "vitest", "run", "src/agent-smoke.test.ts"];
const exitCode = await run("pnpm", vitestArgs, testEnv);

await writeJson(readiness.agent.resultPath, {
  command: "test:e2e:agent",
  startedAt,
  finishedAt: new Date().toISOString(),
  exitCode,
  vitest: {
    cwd: e2eRoot,
    args: vitestArgs,
  },
  readiness,
});

process.exit(exitCode);

async function run(
  command: string,
  commandArgs: string[],
  commandEnv: NodeJS.ProcessEnv
): Promise<number> {
  return await new Promise((resolve) => {
    const child = spawn(command, commandArgs, {
      cwd: e2eRoot,
      env: commandEnv,
      stdio: "inherit",
    });

    child.on("error", (error) => {
      console.error(error);
      resolve(1);
    });

    child.on("close", (code) => {
      resolve(code ?? 1);
    });
  });
}
