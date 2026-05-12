# ai-sdk-llama-cpp

> **Alpha Software**: This package is in early development. APIs may change between versions.

> **macOS Only**: This package currently supports macOS on Apple Silicon and Intel processors.

A [llama.cpp](https://github.com/ggerganov/llama.cpp) provider for the [Vercel AI SDK](https://sdk.vercel.ai/) that implements `LanguageModelV4` and `EmbeddingModelV4`.

It loads llama.cpp directly into Node.js through native C++ bindings, so local GGUF models can run without a separate inference server.

## Features

- Native llama.cpp inference through `node-addon-api` / N-API.
- Metal GPU acceleration on macOS.
- Text generation with `generateText` and `streamText`.
- Structured JSON output with `generateObject`.
- AI SDK tool calling with generated grammar constraints.
- Embeddings with `embed` and `embedMany`.
- Configurable chat templates and reasoning extraction.
- ESM-only package with GGUF model support.

## Requirements

- macOS on Apple Silicon or Intel.
- Node.js >= 18.0.0.
- CMake >= 3.15.
- Xcode Command Line Tools.

```bash
xcode-select --install
brew install cmake
```

Installation on Windows or Linux will fail because this package currently builds only for macOS.

## Installation

```bash
npm install ai-sdk-llama-cpp
```

During installation, the package:

1. Verifies that it is running on macOS.
2. Downloads the pinned llama.cpp revision from GitHub.
3. Compiles llama.cpp as a static library with Metal support.
4. Builds the native Node.js addon.

## Quick Start

Download a GGUF model, then pass its local path to `llamaCpp`.

```typescript
import { generateText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/llama-3.2-1b-instruct.Q4_K_M.gguf",
});

try {
  const { text } = await generateText({
    model,
    prompt: "Explain quantum computing in simple terms.",
  });

  console.log(text);
} finally {
  await model.dispose();
}
```

Always call `dispose()` when you are done with a model so native CPU/GPU resources are released.

## Usage

### Streaming Text

```typescript
import { streamText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
});

try {
  const result = streamText({
    model,
    prompt: "Write a haiku about programming.",
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }
} finally {
  await model.dispose();
}
```

### Structured Output

Structured output uses GBNF grammar constraints so the model generates JSON that conforms to the provided schema.

```typescript
import { generateObject } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
});

try {
  const { object: recipe } = await generateObject({
    model,
    schema: z.object({
      name: z.string(),
      ingredients: z.array(
        z.object({
          name: z.string(),
          amount: z.string(),
        })
      ),
      steps: z.array(z.string()),
    }),
    prompt: "Generate a recipe for chocolate chip cookies.",
  });

  console.log(recipe.name);
} finally {
  await model.dispose();
}
```

Supported schema features include primitives, objects, arrays, enums, constants, `oneOf`, `anyOf`, `allOf`, string constraints, integer ranges, common string formats, and local `$ref` references.

### Tool Calling

Use AI SDK tools with local models. Tool calling also works with `streamText`; tool call JSON is emitted as `tool-call` events instead of raw text.

```typescript
import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
});

try {
  const result = await generateText({
    model,
    prompt: "What's the weather in Tokyo?",
    tools: {
      weather: tool({
        description: "Get the current weather for a location",
        parameters: z.object({
          location: z.string().describe("The location to get weather for"),
        }),
        execute: async ({ location }) => ({
          location,
          temperature: 72,
        }),
      }),
    },
    stopWhen: stepCountIs(3),
  });

  console.log(result.text);
} finally {
  await model.dispose();
}
```

Tool calling quality depends on the model. Models fine-tuned for function calling, such as Llama 3.1+, Hermes 2/3, Functionary, and Qwen 2.5, usually work best.

### Embeddings

```typescript
import { embed, embedMany } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp.embedding({
  modelPath: "./models/nomic-embed-text-v1.5.Q4_K_M.gguf",
});

try {
  const { embedding } = await embed({
    model,
    value: "Hello, world!",
  });

  const { embeddings } = await embedMany({
    model,
    values: ["Hello, world!", "Goodbye, world!"],
  });

  console.log(embedding, embeddings);
} finally {
  model.dispose();
}
```

### Reasoning

Set `model.reasoning` to extract model thinking into AI SDK reasoning parts. With `{}`, the provider extracts text between `<think>` and `</think>`.

```typescript
import { generateText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
  model: {
    reasoning: {},
  },
});

try {
  const result = await generateText({
    model,
    prompt: "Solve 17 * 23 and explain briefly.",
  });

  console.log(result.reasoningText);
  console.log(result.text);
} finally {
  await model.dispose();
}
```

For Gemma 4 thinking support, use the exported model info presets:

```typescript
import { gemma4_31b_it, llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/gemma-4-31b-it.Q4_K_M.gguf",
  model: gemma4_31b_it,
});
```

The package also exports `gemma4_26b_a4b`. For other thinking formats, pass custom delimiters:

```typescript
const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
  model: {
    reasoning: {
      openingMarker: "[reasoning]",
      closingMarker: "[/reasoning]",
    },
  },
});
```

## Configuration

```typescript
const model = llamaCpp({
  // Required: path to the GGUF model file.
  modelPath: "./models/your-model.gguf",

  // Optional: number of layers to offload to GPU.
  // Defaults to 99. Set to 0 to disable GPU offload.
  gpuLayers: 99,

  // Optional: number of CPU threads. Defaults to 4.
  threads: 8,

  // Optional: maximum context size. Defaults to 2048.
  // This is highly model and machine memory dependent. High settings can
  // consume significant memory and may freeze the machine, so monitor system
  // memory when increasing it.
  contextSize: 4096,

  // Optional: enable verbose llama.cpp output. Defaults to false.
  debug: true,

  model: {
    // Optional: "auto" uses the template embedded in the GGUF file.
    // You can also pass a built-in template name like "llama3" or "chatml".
    chatTemplate: "auto",

    // Optional: extract thinking into AI SDK reasoning parts.
    reasoning: {},
  },
});
```

Standard AI SDK generation options are supported, including `maxTokens`, `temperature`, `topP`, `topK`, and `stopSequences`.

Available chat templates include `chatml`, `llama2`, `llama2-sys`, `llama3`, `llama4`, `mistral-v1`, `mistral-v3`, `mistral-v7`, `phi3`, `phi4`, `gemma`, `falcon3`, `zephyr`, `deepseek`, `deepseek2`, `deepseek3`, and `command-r`. See the llama.cpp documentation for the full list.

## Model Downloads

You need to download GGUF models separately. Popular sources include:

- [Hugging Face GGUF models](https://huggingface.co/models?search=gguf)
- [TheBloke's models](https://huggingface.co/TheBloke)

Example:

```bash
mkdir -p models
wget -P models/ https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## API Reference

### `llamaCpp(config)`

Creates a llama.cpp language model compatible with the Vercel AI SDK.

Parameters:

- `config.modelPath` (string, required): Path to the GGUF model file.
- `config.gpuLayers` (number, optional): GPU layers to offload. Defaults to `99`.
- `config.threads` (number, optional): CPU threads. Defaults to `4`.
- `config.contextSize` (number, optional): Maximum context size. Defaults to `2048`. This is highly model and machine memory dependent. High settings can consume significant memory and may freeze the machine, so monitor system memory when increasing it.
- `config.debug` (boolean, optional): Enable verbose llama.cpp output. Defaults to `false`.
- `config.model.chatTemplate` (string, optional): Chat template used to format messages. Defaults to `"auto"`.
- `config.model.reasoning` (object, optional): Extract thinking into AI SDK reasoning parts.

Returns a `LlamaCppLanguageModel`.

### `llamaCpp.embedding(config)`

Creates a llama.cpp embedding model compatible with the Vercel AI SDK.

Parameters are the same base model loading options as `llamaCpp(config)`.

Returns a `LlamaCppEmbeddingModel`.

### Model Lifecycle

Both language and embedding models expose `dispose()`. Call it when finished to unload native resources, especially when loading multiple models in one process.

## Development

From the repository root:

```bash
pnpm install
pnpm build:ts
pnpm build:native
pnpm test:run
```

The native build is configured by `packages/ai-sdk-llama-cpp/CMakeLists.txt` and `packages/ai-sdk-llama-cpp/native/CMakeLists.txt`.

### Updating llama.cpp

The llama.cpp source is fetched during package installation from the `llamaCpp` config in `packages/ai-sdk-llama-cpp/package.json`.

To update the pinned upstream revision:

1. Choose the upstream commit from [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).
2. Update `llamaCpp.commit` in `packages/ai-sdk-llama-cpp/package.json`.
3. Keep `llamaCpp.repo` unchanged unless intentionally switching forks.
4. Remove the old checkout and build artifacts:

```bash
pnpm --filter ai-sdk-llama-cpp clean
```

5. Fetch the new llama.cpp revision and rebuild the native addon:

```bash
pnpm install
```

6. Verify the package:

```bash
pnpm build:native
pnpm build:ts
pnpm test:run
```

If upstream API changes break the native wrapper, update `packages/ai-sdk-llama-cpp/native/llama-wrapper.cpp`, `packages/ai-sdk-llama-cpp/native/llama-wrapper.h`, and `packages/ai-sdk-llama-cpp/native/binding.cpp`, then rerun `pnpm build:native`.

## Limitations

- macOS only. Windows and Linux are not supported.
- Text-only models. Image inputs are not supported.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and contribution guidelines.

## License

MIT

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the inference engine.
- [Vercel AI SDK](https://sdk.vercel.ai/) for the provider interface.
- [node-addon-api](https://github.com/nodejs/node-addon-api) for the N-API C++ wrapper.
