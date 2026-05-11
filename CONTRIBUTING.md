# Contributing

Thank you for your interest in contributing to ai-sdk-llama-cpp!

## Prerequisites

Before you begin, ensure you have the following installed:

- **macOS** (Apple Silicon or Intel) - required, Windows/Linux not supported
- **Node.js** >= 18.0.0
- **Bun** >= 1.3.10
- **CMake** >= 3.15
- **Xcode Command Line Tools**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Install Bun
curl -fsSL https://bun.sh/install | bash

# Optional: Install clang-format for C++ code formatting
brew install clang-format
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/lgrammel/ai-sdk-llama-cpp.git
cd ai-sdk-llama-cpp

# Install dependencies (this also builds the native addon)
bun install

# Build TypeScript
bun run build:ts
```

The `bun install` step automatically compiles llama.cpp and builds the native Node.js addon.

## Development Commands

| Command | Description |
|---------|-------------|
| `bun run build` | Build everything (native + TypeScript) |
| `bun run build:ts` | Build TypeScript only |
| `bun run build:native` | Build native addon only |
| `bun run clean` | Remove build artifacts |
| `bun run test` | Run tests in watch mode |
| `bun run test:run` | Run all tests once |
| `bun run test:unit` | Run unit tests only |
| `bun run test:integration` | Run integration tests only |
| `bun run test:e2e` | Run E2E tests (requires `TEST_MODEL_PATH`) |
| `bun run lint` | Check TypeScript with oxlint |
| `bun run lint:fix` | Fix auto-fixable oxlint issues |
| `bun run format:check` | Check code formatting |
| `bun run format:fix` | Fix code formatting |

## Running Examples

```bash
# From the repository root
bun run --filter @examples/basic generate-text
bun run --filter @examples/basic stream-text
bun run --filter @examples/basic generate-text-tool-call

# Or from the examples directory
cd examples/basic
bun run generate-text
```

## Making Changes

### Workflow

1. **Fork the repository** and clone your fork
2. **Create a feature branch**: `git checkout -b my-feature`
3. **Make your changes**
4. **Run tests**: `bun run test:run`
5. **Run the linter and formatter**: `bun run lint && bun run format:fix`
6. **Add a changeset** (if applicable): `bun run changeset`
7. **Commit your changes** with a descriptive message
8. **Push to your fork** and open a pull request

### Code Style

- **TypeScript**: Strict mode, ES2022 target, ESM only
- **Imports**: Use `.js` extensions for local imports (e.g., `import { foo } from "./bar.js"`)
- **Async/Await**: Preferred over raw Promises
- **Error handling**: Use try/finally for model lifecycle management

### Testing Requirements

- Add **unit tests** for new pure functions in `packages/ai-sdk-llama-cpp/tests/unit/`
- Add **integration tests** for new model behavior in `packages/ai-sdk-llama-cpp/tests/integration/`
- Ensure all existing tests pass before submitting

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
bun run changeset
```

This will prompt you to:
1. Select the `ai-sdk-llama-cpp` package
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
bun run changeset:version
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
bun run changeset:publish
```

This will build the TypeScript and publish the package to npm.

### 3. Create a Git Tag

After publishing, create and push a git tag:

```bash
git tag v$(node -p "require('./packages/ai-sdk-llama-cpp/package.json').version")
git push --tags
```

### 4. Create a GitHub Release

Optionally, create a GitHub release from the tag with the changelog contents.
