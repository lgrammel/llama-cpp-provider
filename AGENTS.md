# AGENTS.md

This file provides guidance for AI coding agents (Cursor, Copilot, Claude Code) working on this codebase.

## Project Overview

**ai-sdk-llama-cpp** is a llama.cpp provider for the Vercel AI SDK, implementing the `LanguageModelV4` interface. It loads llama.cpp directly into Node.js memory via native C++ bindings for local LLM inference.

**Platform Support**: macOS only (Apple Silicon or Intel)

**Monorepo Structure**: This project uses Bun workspaces with packages in `packages/` and examples in `examples/`.

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `bun install` |
| Build everything | `bun run build` |
| Build TypeScript only | `bun run build:ts` |
| Build native only | `bun run build:native` |
| Run all tests once | `bun run test:run` |
| Run unit tests | `bun run test:unit` |
| Run integration tests | `bun run test:integration` |
| Run E2E tests | `TEST_MODEL_PATH=./models/model.gguf bun run test:e2e` |
| Run example | `bun run --filter @examples/basic generate-text` |
| Clean build artifacts | `bun run clean` |

## Setup & Installation

### Prerequisites

- **macOS** (Apple Silicon or Intel) - required
- **Node.js** >= 18.0.0
- **Bun** >= 1.3.10
- **CMake** >= 3.15
- **Xcode Command Line Tools**

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install CMake via Homebrew (if not already installed)
brew install cmake

# Install Bun (if not already installed)
curl -fsSL https://bun.sh/install | bash
```

### Installation Steps

```bash
# Clone and enter the repository
git clone https://github.com/lgrammel/ai-sdk-llama-cpp.git
cd ai-sdk-llama-cpp

# Install dependencies (this also builds the native addon)
bun install

# Build TypeScript
bun run build:ts
```

The `bun install` step automatically:
1. Detects macOS and verifies platform compatibility
2. Compiles llama.cpp as a static library with Metal support
3. Builds the native Node.js addon

### Updating And Building llama.cpp

The llama.cpp source is fetched during package installation from the `llamaCpp` config in `packages/ai-sdk-llama-cpp/package.json`. To update the pinned upstream revision:

1. Choose the upstream commit from `https://github.com/ggerganov/llama.cpp`.
2. Update `packages/ai-sdk-llama-cpp/package.json`:
   - Keep `llamaCpp.repo` pointing at the upstream repository unless intentionally changing forks.
   - Set `llamaCpp.commit` to the new commit SHA.
3. Remove the existing local checkout and native build artifacts:

```bash
bun run --filter ai-sdk-llama-cpp clean
```

4. Reinstall or build the native addon so the postinstall script fetches the new llama.cpp revision:

```bash
bun install
# or, if dependencies are already installed and llama.cpp is present:
bun run build:native
```

5. Verify the TypeScript and native bindings still compile:

```bash
bun run build:ts
bun run test:run
```

If llama.cpp API changes break the native wrapper, update `packages/ai-sdk-llama-cpp/native/llama-wrapper.cpp`, `packages/ai-sdk-llama-cpp/native/llama-wrapper.h`, and `packages/ai-sdk-llama-cpp/native/binding.cpp` as needed, then rerun `bun run build:native`.

## Project Structure

```
├── packages/
│   └── ai-sdk-llama-cpp/       # Main library package
│       ├── src/                # TypeScript source code
│       │   ├── index.ts        # Public exports
│       │   ├── llama-cpp-provider.ts    # Provider factory function
│       │   ├── llama-cpp-language-model.ts  # LanguageModelV4 implementation
│       │   ├── native-binding.ts   # Native module bindings
│       │   └── json-schema-to-grammar.ts   # JSON schema to GBNF grammar converter
│       ├── native/             # C++ native bindings
│       │   ├── binding.cpp     # N-API binding layer
│       │   ├── llama-wrapper.cpp   # llama.cpp wrapper implementation
│       │   └── llama-wrapper.h # llama.cpp wrapper header
│       ├── tests/              # Unit and integration tests
│       │   ├── unit/           # Unit tests (no model required)
│       │   └── integration/    # Integration tests (mocked native bindings)
│       ├── dist/               # Compiled TypeScript output (generated)
│       └── build/              # Native addon build output (generated)
├── tests/
│   └── e2e/                    # End-to-end tests (requires real model)
│       └── src/                # E2E test files
├── examples/
│   └── basic/                  # Basic usage examples
│       └── src/                # Example source files
│           ├── generate-text.ts
│           ├── stream-text.ts
│           ├── generate-text-output.ts
│           ├── chatbot.ts
│           └── embed-many.ts
├── package.json                # Root package scripts and workspace configuration
└── package.json                # Root package.json with workspace scripts
```

## Testing

### Test Organization

- **Unit tests** (`packages/ai-sdk-llama-cpp/tests/unit/`): Test pure functions and class instantiation. No model or native bindings required.
- **Integration tests** (`packages/ai-sdk-llama-cpp/tests/integration/`): Test the language model class with mocked native bindings.
- **E2E tests** (`tests/e2e/`): Test actual inference with a real GGUF model file. This is a separate workspace package (`@tests/e2e`).

### Running Tests

```bash
# Run all tests once
bun run test:run

# Run tests in watch mode (for development)
bun run test

# Run specific test categories
bun run test:unit
bun run test:integration

# Run E2E tests (requires a GGUF model)
TEST_MODEL_PATH=./models/your-model.gguf bun run test:e2e

# Run tests with coverage
bun run test:coverage
```

### E2E Test Requirements

E2E tests require the `TEST_MODEL_PATH` environment variable to point to a valid GGUF model file. Without this, E2E tests are automatically skipped.

```bash
# Download a model for testing
mkdir -p models
wget -P models/ https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Run E2E tests
TEST_MODEL_PATH=./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf bun run test:e2e
```

### Writing Tests

- Use Vitest (`describe`, `it`, `expect`)
- Unit/integration tests are in `packages/ai-sdk-llama-cpp/tests/**/*.test.ts`
- E2E tests are in `tests/e2e/src/**/*.test.ts`
- Test timeout is 120 seconds (configured in `vitest.config.ts`)
- E2E tests use a `describeE2E` pattern that skips tests when `TEST_MODEL_PATH` is not set

## Examples

### Running Examples

Examples are in the `examples/basic` workspace package:

```bash
# Run examples using Bun workspace filter
bun run --filter @examples/basic generate-text
bun run --filter @examples/basic stream-text
bun run --filter @examples/basic generate-text-output
bun run --filter @examples/basic chatbot
bun run --filter @examples/basic embed-many

# Or run directly from the examples/basic directory
cd examples/basic
bun run generate-text
```

### Example Structure

Examples follow this pattern:

```typescript
import { generateText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

// Create model instance with config
const model = llamaCpp({ 
  modelPath: "./models/your-model.gguf",
  // Optional load config: gpuLayers, threads, debug
  // Optional model info: model.contextSize, model.chatTemplate, model.reasoning
});

try {
  // Use with AI SDK functions
  const result = await generateText({
    model,
    prompt: "Your prompt here",
  });

  console.log(result.text);
} finally {
  // Always dispose to free resources
  await model.dispose();
}
```

### Creating New Examples

1. Create a new file in `examples/basic/src/` directory
2. Import from `"ai-sdk-llama-cpp"` (workspace dependency)
3. Use `try/finally` to ensure `model.dispose()` is called
4. Update the model path to your local GGUF model
5. Add a script to `examples/basic/package.json` (e.g., `"my-example": "bun src/my-example.ts"`)
6. Run with `bun run --filter @examples/basic your-script-name`

Example template:

```typescript
import { generateText, streamText, Output } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({ 
  modelPath: "./models/your-model.gguf",
  model: {
    contextSize: 4096, // optional
  },
});

try {
  // Your example code here
  const { text } = await generateText({
    model,
    prompt: "Hello, world!",
    maxTokens: 100,
  });
  console.log(text);
} finally {
  await model.dispose();
}
```

## Code Style & Conventions

- **Module system**: ESM only (no CommonJS)
- **TypeScript**: Strict mode enabled
- **Target**: ES2022
- **Imports**: Use `.js` extensions for local imports (e.g., `import { foo } from "./bar.js"`)
- **Async/Await**: Preferred over raw Promises
- **Error handling**: Use try/finally for model lifecycle management

## Architecture & Internal Structure

### Core Components

| File | Purpose |
|------|---------|
| `llama-cpp-provider.ts` | Factory function `llamaCpp()` - creates model instances, handles config |
| `llama-cpp-language-model.ts` | `LanguageModelV4` implementation - `doGenerate()`, `doStream()`, tool call handling |
| `llama-cpp-embedding-model.ts` | `EmbeddingModelV4` implementation for embeddings |
| `native-binding.ts` | TypeScript bindings to the native C++ addon |
| `json-schema-to-grammar.ts` | Converts JSON Schema to GBNF grammar for structured output |
| `index.ts` | Public exports |

### Native Layer (C++)

| File | Purpose |
|------|---------|
| `binding.cpp` | N-API binding layer - exposes C++ functions to Node.js |
| `llama-wrapper.cpp` | Wraps llama.cpp API - model loading, inference, tokenization |
| `llama-wrapper.h` | Header file for the wrapper |

### Key Implementation Details

- **Tool calling**: Implemented in `llama-cpp-language-model.ts` via GBNF grammar constraints. The `buildToolCallGrammar()` function generates grammar that forces valid JSON tool call output. Tool call detection happens in both `doGenerate()` and `doStream()`.

- **Structured output**: Uses `json-schema-to-grammar.ts` to convert Zod schemas (via JSON Schema) to GBNF grammars. Supports primitives, objects, arrays, enums, and composition (`oneOf`, `anyOf`, `allOf`).

- **Streaming**: Native addon yields tokens via callback. `doStream()` converts these to AI SDK stream format with proper chunk types (`text-delta`, `tool-call`, `finish`).

- **Chat templates**: Applied in native layer via llama.cpp's built-in template system. The `chatTemplate` config option is passed through to the native binding.

- **Resource management**: Model memory is managed in C++. `dispose()` must be called to free GPU/CPU resources. The native binding handles cleanup.

### Data Flow

1. User calls `generateText()` / `streamText()` with AI SDK
2. AI SDK calls `doGenerate()` / `doStream()` on `LlamaCppLanguageModel`
3. Language model formats prompt using chat template
4. If tools/schema provided, GBNF grammar is generated
5. Native binding calls llama.cpp for inference
6. Results are parsed and returned in AI SDK format

## Common Tasks

### Adding a New Feature

1. Implement in appropriate `packages/ai-sdk-llama-cpp/src/` file
2. Export from `src/index.ts` if public API
3. Add unit tests in `packages/ai-sdk-llama-cpp/tests/unit/`
4. Add integration tests in `packages/ai-sdk-llama-cpp/tests/integration/`
5. Run `bun run test:run` to verify
6. Build with `bun run build:ts`

### Modifying Native Bindings

1. Edit files in `packages/ai-sdk-llama-cpp/native/`
2. Rebuild with `bun run build:native`
3. Test with `bun run test:run`

### Debugging

- Enable verbose llama.cpp output: `llamaCpp({ modelPath, debug: true })`
- Run specific test: `bun run --filter ai-sdk-llama-cpp vitest run tests/unit/provider.test.ts`
- Debug build: `bun run --filter ai-sdk-llama-cpp build:native:debug`

## Dependencies

- **Runtime**: `@ai-sdk/provider`, `cmake-js`, `node-addon-api`
- **Dev**: `ai`, `typescript`, `vitest`, `zod`

## Limitations

- macOS only (Windows/Linux not supported)
- No image input support (text only)
