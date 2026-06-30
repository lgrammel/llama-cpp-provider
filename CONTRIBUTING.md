# Contributing

Thank you for your interest in contributing to @lgrammel/llama-cpp-provider!

## Prerequisites

Before you begin, ensure you have the following installed:

- **macOS** (Apple Silicon or Intel) - required, Windows/Linux not supported
- **Node.js** >= 22.0.0
- **pnpm** >= 9.0.0
- **CMake** >= 3.15
- **Xcode Command Line Tools**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Install pnpm
npm install -g pnpm

# Optional: Install clang-format for C++ code formatting
brew install clang-format
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/lgrammel/ai-sdk-llama-cpp.git
cd ai-sdk-llama-cpp

# Install dependencies (this also builds the native addon)
pnpm install

# Build TypeScript
pnpm build:ts
```

The `pnpm install` step automatically compiles llama.cpp and builds the native Node.js addon.

## Development Commands

| Command | Description |
|---------|-------------|
| `pnpm build` | Build everything (native + TypeScript) |
| `pnpm build:ts` | Build TypeScript only |
| `pnpm build:native` | Build native addon only |
| `pnpm clean` | Remove build artifacts |
| `pnpm test` | Run tests in watch mode |
| `pnpm test:run` | Run all tests once |
| `pnpm test:unit` | Run unit tests only |
| `pnpm test:integration` | Run integration tests only |
| `pnpm test:e2e` | Run E2E tests (requires `TEST_MODEL_PATH`) |
| `pnpm test:e2e:agent` | Run short agent-friendly E2E smoke tests |
| `pnpm e2e:doctor` | Check E2E model/native-addon readiness |
| `pnpm lint` | Check TypeScript with oxlint |
| `pnpm lint:fix` | Fix auto-fixable oxlint issues |
| `pnpm format:check` | Check code formatting |
| `pnpm format:fix` | Fix code formatting |

## Running Examples

```bash
# From the repository root
pnpm --filter @examples/basic agent
pnpm --filter @examples/basic agent-tool-call
pnpm --filter @examples/basic agent-reasoning

# Or from the examples directory
cd examples/basic
pnpm agent
```

## Making Changes

### Workflow

1. **Fork the repository** and clone your fork
2. **Create a feature branch**: `git checkout -b my-feature`
3. **Make your changes**
4. **Run tests**: `pnpm test:run`
5. **Run the linter and formatter**: `pnpm lint && pnpm format:fix`
6. **Add a changeset** (if applicable): `pnpm changeset`
7. **Commit your changes** with a descriptive message
8. **Push to your fork** and open a pull request

### Code Style

- **TypeScript**: Strict mode, ES2022 target, ESM only
- **Imports**: Use `.js` extensions for local imports (e.g., `import { foo } from "./bar.js"`)
- **Async/Await**: Preferred over raw Promises
- **Error handling**: Use try/finally for model lifecycle management

### Testing Requirements

- Add **unit tests** for new pure functions in `packages/llama-cpp-provider/tests/unit/`
- Add **integration tests** for new model behavior in `packages/llama-cpp-provider/tests/integration/`
- Ensure all existing tests pass before submitting
- For automated local E2E verification, run `pnpm test:e2e:agent`. It downloads or reuses a cached GGUF model unless `TEST_MODEL_PATH` is set, then runs a compact smoke suite and writes `test-results/e2e-agent.json`.

### Pull Request Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Write clear descriptions**: Explain what changed and why
- **Include tests**: PRs without tests for new functionality may be delayed
- **Update documentation**: If your change affects the public API, update the README

## Changesets

This project uses [changesets](https://github.com/changesets/changesets) for versioning and changelog management.

### Adding a Changeset

When you make changes that should be released, add a changeset:

```bash
pnpm changeset
```

This will prompt you to:
1. Select the `@lgrammel/llama-cpp-provider` package
2. Choose the semver bump type:
   - **patch**: Bug fixes, documentation updates
   - **minor**: New features, non-breaking changes
   - **major**: Breaking changes
3. Write a summary of your changes (this appears in the changelog)

The changeset file will be created in the `.changeset` directory. Include this file in your pull request.

### When to Add a Changeset

**Add a changeset when your changes:**
- Fix a bug
- Add a new feature
- Change existing behavior
- Update dependencies in a meaningful way

**You don't need a changeset for:**
- Documentation-only changes (unless they fix incorrect docs)
- Changes to tests, examples, or dev tooling
- Refactoring with no user-facing changes

## Project Structure

For detailed information about the codebase structure, architecture, and internal components, see [AGENTS.md](./AGENTS.md).

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/lgrammel/ai-sdk-llama-cpp/discussions)
- **Bug reports**: Open a [GitHub Issue](https://github.com/lgrammel/ai-sdk-llama-cpp/issues) with reproduction steps
- **Feature requests**: Open a GitHub Issue describing the use case

## Releasing (Maintainers Only)

Releases are done manually by maintainers.

### 1. Version Packages

Run the version command to consume all changesets and update package versions:

```bash
pnpm changeset:version
```

This will:
- Update `package.json` versions
- Update `CHANGELOG.md` files
- Remove the consumed changeset files

Review the changes and commit them:

```bash
git add .
git commit -m "chore: version packages"
git push
```

### 2. Publish to npm

Ensure you're logged in to npm:

```bash
npm login
```

Then publish:

```bash
pnpm changeset:publish
```

This will build the TypeScript and publish the package to npm.

### 3. Create a Git Tag

After publishing, create and push a git tag:

```bash
git tag v$(node -p "require('./packages/llama-cpp-provider/package.json').version")
git push --tags
```

### 4. Create a GitHub Release

Optionally, create a GitHub release from the tag with the changelog contents.
