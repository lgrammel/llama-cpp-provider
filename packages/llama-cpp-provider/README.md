# @lgrammel/llama-cpp-provider

> **Alpha software**: this package is in early development. APIs may change between versions.

> **macOS only**: Apple Silicon and Intel Macs are supported. Windows and Linux are not currently supported.

A [llama.cpp](https://github.com/ggerganov/llama.cpp) provider for the [Vercel AI SDK](https://sdk.vercel.ai/). It runs local GGUF models inside Node.js through native C++ bindings, without a separate inference server.

The provider implements AI SDK language and embedding model interfaces for local agents, including tool calling, structured output support, image inputs for multimodal models, reasoning extraction, and embeddings.

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

## Usage

Download a GGUF model locally, pass its path to `llamaCpp`, and use it with `ToolLoopAgent`. For interactive chat examples, use `@lgrammel/agent-tui`.

```typescript
import { runAgentTUI } from "@lgrammel/agent-tui";
import { ToolLoopAgent } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

const model = llamaCpp({
  modelPath: "./models/llama-3.2-1b-instruct.Q4_K_M.gguf",
});

const agent = new ToolLoopAgent({
  model,
  instructions: "You are a concise local assistant.",
});

try {
  await runAgentTUI({ name: "Local assistant", agent });
} finally {
  await model.dispose();
}
```

Always call `dispose()` when you are done with a model so native CPU/GPU resources are released.

## API

### `llamaCpp(config)`

Creates an AI SDK language model for `ToolLoopAgent` and other AI SDK consumers.

```typescript
import { ToolLoopAgent } from "ai";
import { llamaCpp } from "@lgrammel/llama-cpp-provider";

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

const agent = new ToolLoopAgent({ model });
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

Standard AI SDK generation settings are supported by the language model, including `maxOutputTokens`, `temperature`, `topP`, `topK`, and `stopSequences`.

### `llamaCpp.embedding(config)`

Creates an AI SDK embedding model for `embed` and `embedMany`. It uses the same base loading options as `llamaCpp(config)`.

## Limitations

- macOS only.
- ESM only.
- Model and projector paths must point to local GGUF files.
- Image inputs require a compatible multimodal model and `mmprojPath`.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and contribution guidelines.

## License

MIT
