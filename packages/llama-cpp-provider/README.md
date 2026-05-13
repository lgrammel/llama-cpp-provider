# @lgrammel/llama-cpp-provider

> **Alpha software**: this package is in early development. APIs may change between versions.

> **macOS only**: Apple Silicon and Intel Macs are supported. Windows and Linux are not currently supported.

A [llama.cpp](https://github.com/ggerganov/llama.cpp) provider for the [Vercel AI SDK](https://sdk.vercel.ai/). It runs local GGUF models inside Node.js through native C++ bindings, without a separate inference server.

The provider implements AI SDK language and embedding model interfaces, including text generation, streaming, structured output, tool calling, image inputs for multimodal models, and embeddings.

## Requirements

- macOS on Apple Silicon or Intel.
- Node.js >= 18.
- CMake >= 3.15.
- Xcode Command Line Tools.

```bash
xcode-select --install
brew install cmake
```

```bash
npm install @lgrammel/llama-cpp-provider
```

Installation downloads the pinned llama.cpp revision, builds it with Metal support, and compiles the native Node.js addon.

## Quick Start

Download a GGUF model locally, then pass its path to `llamaCpp`.

```typescript
import { generateText } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

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

### Streaming

```typescript
import { streamText } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

const model = llamaCpp({ modelPath: "./models/your-model.gguf" });

try {
  const result = streamText({
    model,
    prompt: "Write a haiku about local inference.",
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }
} finally {
  await model.dispose();
}
```

### Structured Output

`generateObject` uses llama.cpp grammar constraints to produce JSON that matches the requested schema.

```typescript
import { generateObject } from "ai";
import { z } from "zod";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

const model = llamaCpp({ modelPath: "./models/your-model.gguf" });

try {
  const { object } = await generateObject({
    model,
    schema: z.object({
      name: z.string(),
      steps: z.array(z.string()),
    }),
    prompt: "Create a short cookie recipe.",
  });

  console.log(object);
} finally {
  await model.dispose();
}
```

Supported schema features include primitives, objects, arrays, enums, constants, `oneOf`, `anyOf`, `allOf`, string constraints, integer ranges, common string formats, and local `$ref` references.

### Tool Calling

Use AI SDK tools with local models. Tool calling quality depends on the model; function-calling tuned models usually work best.

```typescript
import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

const model = llamaCpp({ modelPath: "./models/your-model.gguf" });

try {
  const result = await generateText({
    model,
    prompt: "What's the weather in Tokyo?",
    tools: {
      weather: tool({
        description: "Get the weather for a location",
        parameters: z.object({ location: z.string() }),
        execute: async ({ location }) => ({ location, temperature: 22 }),
      }),
    },
    stopWhen: stepCountIs(3),
  });

  console.log(result.text);
} finally {
  await model.dispose();
}
```

### Image Inputs

Image inputs require a vision-capable GGUF model and its matching multimodal projector (`mmproj`) GGUF file.

```typescript
import { readFile } from "node:fs/promises";
import { generateText } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

const model = llamaCpp({
  modelPath: "./models/gemma-4-31b-it.gguf",
  mmprojPath: "./models/mmproj-gemma-4-31b-it.gguf",
  model: { chatTemplate: "gemma" },
});

try {
  const { text } = await generateText({
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Describe this image." },
          {
            type: "file",
            data: { type: "data", data: await readFile("./image.png") },
            mediaType: "image/png",
          },
        ],
      },
    ],
  });

  console.log(text);
} finally {
  await model.dispose();
}
```

Inline `Uint8Array`, base64 data, and data URL image parts are supported. Local file paths should be loaded into bytes before calling the model.

### Embeddings

```typescript
import { embed, embedMany } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

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
    values: ["Hello", "World"],
  });

  console.log(embedding, embeddings);
} finally {
  model.dispose();
}
```

### Reasoning

Set `model.reasoning` to extract thinking into AI SDK reasoning parts. With `{}`, the provider extracts text between `<think>` and `</think>`.

```typescript
import { generateText } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
  model: { reasoning: {} },
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

The package also exports model info presets such as `gemma4_31b_it`, `gemma4_26b_a4b`, `qwen3_6_dense`, and `qwen3_6_moe`.

## API

### `llamaCpp(config)`

Creates an AI SDK language model for `generateText`, `streamText`, `generateObject`, and tool calling.

```typescript
const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
  mmprojPath: "./models/mmproj.gguf",
  contextSize: 4096,
  gpuLayers: 99,
  threads: 8,
  debug: false,
  model: {
    chatTemplate: "auto",
    reasoning: {},
  },
});
```

Important options:

- `modelPath` is required and must point to a local GGUF model file.
- `mmprojPath` points to a multimodal projector GGUF file and is required for image inputs.
- `contextSize` defaults to `2048`. Higher values can use significant memory.
- `gpuLayers` defaults to `99`, which offloads all available layers to GPU. Use `0` to disable GPU offload.
- `threads` defaults to `4`.
- `debug` enables verbose llama.cpp output.
- `model.chatTemplate` defaults to `"auto"`, which uses the template embedded in the GGUF file. You can also pass a llama.cpp template name such as `"llama3"`, `"chatml"`, or `"gemma"`.
- `model.reasoning` extracts thinking text into AI SDK reasoning parts.
- `memorySafety` can reject or clamp context sizes that are estimated to exceed available memory when model memory metadata is provided.

Standard AI SDK generation options are supported, including `maxOutputTokens`, `temperature`, `topP`, `topK`, and `stopSequences`.

### `llamaCpp.embedding(config)`

Creates an AI SDK embedding model for `embed` and `embedMany`. It uses the same base loading options as `llamaCpp(config)`.

## Models

GGUF models are downloaded separately. Hugging Face is the most common source:

- [GGUF model search](https://huggingface.co/models?search=gguf)
- [bartowski GGUF models](https://huggingface.co/bartowski)

Model size, quantization, context size, and multimodal projector compatibility all matter. If a model has an embedded chat template, `chatTemplate: "auto"` is usually the best starting point.

## Limitations

- macOS only.
- ESM only.
- Model and projector paths must point to local GGUF files.
- Image inputs require a compatible multimodal model and `mmprojPath`.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and contribution guidelines.

## License

MIT
