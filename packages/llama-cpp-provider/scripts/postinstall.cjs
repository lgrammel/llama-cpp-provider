const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

// Step 1: Check platform is macOS
if (process.platform !== "darwin") {
  console.error("\n===========================================");
  console.error("ERROR: @lgrammel/llama-cpp-provider only supports macOS");
  console.error("===========================================\n");
  console.error(`Detected platform: ${process.platform}`);
  console.error("This package requires macOS for native compilation.\n");
  process.exit(1);
}

// Step 2: Check if git is available
try {
  execSync("git --version", { stdio: "ignore" });
} catch {
  console.error("\n===========================================");
  console.error("ERROR: git is required but not found");
  console.error("===========================================\n");
  console.error("Please install git to continue.");
  console.error("  macOS: xcode-select --install");
  console.error("  or: brew install git\n");
  process.exit(1);
}

// Step 3: Check if llama.cpp already exists (for local development)
const llamaCppPath = path.join(__dirname, "..", "llama.cpp");
if (fs.existsSync(llamaCppPath)) {
  console.log("llama.cpp directory already exists, skipping clone...");
} else {
  // Step 4: Read commit hash from package.json
  const packageJsonPath = path.join(__dirname, "..", "package.json");
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
  const { repo, commit } = packageJson.llamaCpp;

  if (!repo || !commit) {
    console.error(
      "ERROR: llamaCpp.repo and llamaCpp.commit must be defined in package.json"
    );
    process.exit(1);
  }

  // Step 5: Clone llama.cpp at the specific commit
  console.log(`Cloning llama.cpp at commit ${commit}...`);
  try {
    // Clone with depth 1 for faster download, then fetch the specific commit
    execSync(`git clone --depth 1 ${repo} llama.cpp`, {
      stdio: "inherit",
      cwd: path.join(__dirname, ".."),
    });

    // Fetch the specific commit and checkout
    execSync(`git fetch --depth 1 origin ${commit}`, {
      stdio: "inherit",
      cwd: llamaCppPath,
    });
    execSync(`git checkout ${commit}`, {
      stdio: "inherit",
      cwd: llamaCppPath,
    });

    console.log("llama.cpp cloned successfully.");
  } catch (error) {
    console.error("ERROR: Failed to clone llama.cpp");
    console.error(error.message);
    process.exit(1);
  }
}

// Step 6: Build native bindings
console.log("Building native llama.cpp bindings for macOS...");
execSync("npx cmake-js compile", { stdio: "inherit" });
