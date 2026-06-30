import { collectReadiness, printReadiness, writeJson } from "./e2e-common.js";

const prepareModel = process.argv.includes("--prepare");
const readiness = await collectReadiness({ prepareModel });

await writeJson(readiness.agent.resultPath, {
  command: "e2e:doctor",
  preparedModel: prepareModel,
  readiness,
});

printReadiness(readiness);

if (!readiness.ok) {
  process.exitCode = 1;
}
