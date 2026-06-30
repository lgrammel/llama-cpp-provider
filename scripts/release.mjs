#!/usr/bin/env node
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));
const packageName = "@lgrammel/llama-cpp-provider";
const packageJsonPath = join(rootDir, "packages/llama-cpp-provider/package.json");
const changesetDir = join(rootDir, ".changeset");
const releaseBranch = "main";

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printHelp();
  process.exit(0);
}

main();

function main() {
  assertCleanWorkingTree();
  assertReleaseBranch();
  assertUpToDateWithUpstream();

  const pendingChangesets = getPendingChangesets();
  const initialVersion = readPackageVersion();

  if (options.dryRun) {
    log("Running release dry run.");
    logPendingChangesets(pendingChangesets);
    runChecks();
    run("pnpm", ["--filter", packageName, "publish", "--dry-run", "--no-git-checks"]);
    log("Dry run complete. No versions, commits, npm publishes, or tags changed.");
    return;
  }

  if (pendingChangesets.length > 0) {
    logPendingChangesets(pendingChangesets);
    run("pnpm", ["run", "version"]);
  } else {
    log("No pending changesets. Assuming package version is already prepared.");
  }

  const releaseVersion = readPackageVersion();
  const releaseTag = getReleaseTag(releaseVersion);

  if (pendingChangesets.length > 0 && releaseVersion === initialVersion) {
    fail(`Changesets did not update ${packageName}; refusing to publish ${releaseTag}.`);
  }

  assertTagDoesNotExist(releaseTag);

  runChecks();
  run("pnpm", ["--filter", packageName, "publish", "--dry-run", "--no-git-checks"]);

  commitReleaseIfNeeded(releaseTag);

  const publishArgs = ["exec", "changeset", "publish"];
  if (options.otp !== undefined) {
    publishArgs.push("--otp", options.otp);
  }
  run("pnpm", publishArgs);

  run("pnpm", ["exec", "changeset", "tag"]);

  if (!tagExists(releaseTag)) {
    fail(`Expected release tag ${releaseTag} to exist after publishing.`);
  }

  if (options.push) {
    run("git", ["push", "origin", "HEAD"]);
    run("git", ["push", "origin", releaseTag]);
  }

  log(`Released ${releaseTag}.`);
  if (!options.push) {
    log(`Push the release commit and tag with: git push && git push origin ${releaseTag}`);
  }
}

function runChecks() {
  run("pnpm", ["format:check"]);
  run("pnpm", ["lint"]);
  run("pnpm", ["test:run"]);
  run("pnpm", ["build"]);
}

function commitReleaseIfNeeded(releaseTag) {
  const status = getOutput("git", ["status", "--porcelain"]);
  if (status.length === 0) {
    log("No release file changes to commit.");
    return;
  }

  run("git", ["add", "-A"]);
  run("git", ["commit", "-m", `Release ${releaseTag}`]);
}

function assertCleanWorkingTree() {
  const status = getOutput("git", ["status", "--porcelain"]);
  if (status.length > 0) {
    fail("Release requires a clean working tree. Commit, stash, or discard local changes first.");
  }
}

function assertReleaseBranch() {
  const branch = getOutput("git", ["branch", "--show-current"]);
  if (branch !== releaseBranch && !options.allowBranch) {
    fail(`Release must run from ${releaseBranch}. Current branch is ${branch || "(detached)"}.`);
  }
}

function assertUpToDateWithUpstream() {
  let upstream;
  try {
    upstream = getOutput("git", ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]);
  } catch {
    log("No upstream branch configured; skipping upstream sync check.");
    return;
  }

  const counts = getOutput("git", ["rev-list", "--left-right", "--count", `${upstream}...HEAD`]);
  const [behindText, aheadText] = counts.split(/\s+/);
  const behind = Number(behindText);
  const ahead = Number(aheadText);

  if (!options.allowBranch && (behind > 0 || ahead > 0)) {
    fail(
      `Release branch must be in sync with ${upstream}. Local branch is ${ahead} ahead and ${behind} behind.`,
    );
  }
}

function assertTagDoesNotExist(releaseTag) {
  if (tagExists(releaseTag)) {
    fail(`Release tag already exists: ${releaseTag}`);
  }
}

function tagExists(releaseTag) {
  return getOutput("git", ["tag", "--list", releaseTag]) === releaseTag;
}

function getPendingChangesets() {
  if (!existsSync(changesetDir)) {
    return [];
  }

  return readdirSync(changesetDir)
    .filter((file) => file.endsWith(".md"))
    .filter((file) => file !== "README.md")
    .sort();
}

function logPendingChangesets(changesets) {
  if (changesets.length === 0) {
    log("No pending changesets found.");
    return;
  }

  log(`Pending changesets: ${changesets.join(", ")}`);
}

function getReleaseTag(version) {
  return `${packageName}@${version}`;
}

function readPackageVersion() {
  const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
  return packageJson.version;
}

function run(command, args) {
  log(`$ ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: rootDir,
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function getOutput(command, args) {
  return execFileSync(command, args, {
    cwd: rootDir,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function parseArgs(args) {
  const parsed = {
    allowBranch: false,
    dryRun: false,
    help: false,
    otp: undefined,
    push: false,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === "--") {
      continue;
    } else if (arg === "--allow-branch") {
      parsed.allowBranch = true;
    } else if (arg === "--dry-run") {
      parsed.dryRun = true;
    } else if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--push") {
      parsed.push = true;
    } else if (arg === "--otp") {
      const otp = args[index + 1];
      if (otp === undefined) {
        fail("--otp requires a value.");
      }
      parsed.otp = otp;
      index += 1;
    } else if (arg.startsWith("--otp=")) {
      parsed.otp = arg.slice("--otp=".length);
    } else {
      fail(`Unknown release option: ${arg}`);
    }
  }

  return parsed;
}

function printHelp() {
  console.log(`Usage: pnpm release [options]

Prepare, commit, publish, and tag ${packageName}.

Options:
  --dry-run        Run checks and npm publish dry-run without changing versions.
  --otp <code>     Forward an npm two-factor auth code to changeset publish.
  --push           Push HEAD and the release tag after publishing.
  --allow-branch   Allow releasing from a branch other than ${releaseBranch}.
  -h, --help       Show this help message.
`);
}

function log(message) {
  console.log(`[release] ${message}`);
}

function fail(message) {
  console.error(`[release] ${message}`);
  process.exit(1);
}
