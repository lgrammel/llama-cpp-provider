# Requirements

These requirements describe the current `@lgrammel/llama-cpp-provider` behavior that is clear from the package code and tests.

## Runtime And Build

- The package must run on macOS only.
- The package must support Apple Silicon and Intel Macs.
- The package must require Node.js 22 or newer.
- The package must be ESM-only.
- Installation must fail on non-macOS platforms before native compilation.
- Installation must require `git` so the pinned llama.cpp source can be fetched.
- Native builds must compile a Node N-API addon through `cmake-js`.
- The pinned llama.cpp repository and commit must be read from `llamaCpp.repo` and `llamaCpp.commit` in `packages/llama-cpp-provider/package.json`.

## Public API

- The package must export `llamaCpp` as the provider factory and as the default export.
- Calling `llamaCpp(config)` must create a `LanguageModelV4` implementation.
- Calling `llamaCpp.languageModel(config)` must behave the same as `llamaCpp(config)`.
- Calling `llamaCpp.embedding(config)` must create an `EmbeddingModelV4` implementation.
- Language and embedding models must report `specificationVersion: "v4"` and `provider: "llama.cpp"`.
- Language and embedding model IDs must equal the configured `modelPath`.
- The language model must not advertise URL support; `supportedUrls` must be empty.

## Model Loading

- Models must load lazily on first generation, stream, or embedding call.
- Loaded model handles must be reused while the native binding reports them loaded.
- `dispose()` must unload the native model and be safe to call multiple times.
- `modelPath` must point to a local file.
- `modelPath` and `mmprojPath` must reject paths starting with `~/` because shell expansion is not performed.
- `modelPath` and `mmprojPath` must reject missing files and directories before native loading.
- Default load options for language models must be `contextSize: 2048`, `gpuLayers: 99`, `threads: 4`, `debug: false`, and `chatTemplate: "auto"`.
- Default load options for embedding models must be `contextSize: 2048`, `gpuLayers: 99`, `threads: 4`, `debug: false`, and `embedding: true`.
- `mmprojPath` must be passed to native loading only when configured.

## Generation

- The language model must implement non-streaming generation with `doGenerate`.
- The language model must implement streaming generation with `doStream`.
- Generation calls must pass converted chat messages to the native binding.
- Default generation options must be `maxTokens: 256`, `temperature: 0.7`, `topP: 0.9`, and `topK: 40`.
- AI SDK `maxOutputTokens`, `temperature`, `topP`, `topK`, and `stopSequences` must be forwarded to native generation.
- AI SDK `seed` must be forwarded to native generation when provided.
- Omitted `seed` must use llama.cpp random seeding.
- `maxOutputTokens` must be a positive integer and must not exceed the loaded context size.
- `seed` must be an integer between 0 and 4294967295 when provided.
- Native `stop` and `length` finish reasons must map to AI SDK finish reasons with the same unified value; other native finish reasons must map to `other`.
- Usage must report prompt tokens as input tokens and completion tokens as output text tokens.
- Native generation errors must reject the generation call.
- Aborting one generation call must cancel only that call's native worker, not every worker on the same model handle.

## Streaming

- `doStream` must return a `ReadableStream`.
- Streams must begin with `stream-start`.
- Text streams must emit `text-start`, `text-delta`, and `text-end` parts with a stable text ID.
- Streams must end with a `finish` part containing finish reason and usage.
- Streaming token callbacks must surface native tokens as text deltas unless active tool calling requires buffering for native tool-call parsing.
- Streaming must not emit configured stop sequence text as text deltas.

## Message Conversion

- System messages must remain system messages.
- User text parts must be concatenated into the user message content.
- Assistant text parts must be concatenated into assistant message content.
- Assistant tool-call parts must be converted to structured native `toolCalls` fields.
- Tool results must be converted to `tool` role messages with the tool name, tool call ID, and result text.
- Tool result JSON values must be stringified.
- Tool result errors must be prefixed with `Error:`.

## Image Inputs

- User file parts with media type `image` or `image/*` must be treated as image inputs.
- Image data must be accepted as `Uint8Array`, base64 string data, or base64 data URLs.
- Non-image file parts must be ignored by the provider message converter.
- Remote image URLs must be rejected.
- File URL image inputs must be rejected with an error that says the AI SDK must load them before reaching the provider.
- Image prompts must include media markers in text order and pass image bytes to the native layer.
- Native multimodal generation must require `mmprojPath`; image generation without a loaded multimodal projector must fail.

## Structured Output

- JSON response formats with schemas must be converted to GBNF grammar and passed to native generation.
- Structured output grammar must disable reasoning extraction for that call.
- The JSON schema converter must support clear schema features used by the code: primitives, objects, arrays, enums, constants, `$ref`, `oneOf`, `anyOf`, `allOf`, string formats, string patterns, and numeric bounds.
- Remote schema fetching must not be required or supported.

## Tool Calling

- Function tools must be passed to the native llama.cpp common-chat template and parser path.
- Function tools must not add a generic TypeScript-generated tool system prompt.
- Tool prompting must be skipped when `toolChoice.type` is `"none"`.
- `toolChoice.type: "required"` must request llama.cpp required tool choice for the provided function tools.
- `toolChoice.type: "tool"` must request llama.cpp required tool choice for the selected function tool.
- `toolChoice.type: "tool"` must reject unknown function tool names before native generation.
- `providerOptions["llama.cpp"].parallelToolCalls: true` must forward `parallelToolCalls: true` to native generation.
- Native tool output parsing must use llama.cpp common-chat parser parameters produced by the applied chat template.
- Tool output parsing must retain a compatibility fallback that accepts a single JSON object, an array of JSON objects, or a legacy `tool_calls` wrapper.
- Parsed tool calls must produce AI SDK `tool-call` content with JSON-stringified arguments.
- Parsed tool calls must set the unified finish reason to `tool-calls`.
- Non-JSON or invalid tool-call output must be returned as normal text.
- Streaming with active tools must buffer generated text and emit parsed text/tool-call parts after native generation completes.

## Reasoning

- Reasoning extraction must split generated text into AI SDK reasoning and text parts.
- The default reasoning markers must be `<think>` and `</think>`.
- Model-specific reasoning markers and prompt prefixes must be configurable.
- Reasoning extraction must be disabled when no model reasoning config is set and the call does not request reasoning.
- `reasoning: "none"` in a call must disable reasoning extraction.
- When a reasoning prompt prefix is configured, it must be inserted into the first system message or prepended as a new system message.
- Reasoning extraction must operate on tool-call visible text so hidden reasoning does not prevent tool-call parsing.

## Embeddings

- The embedding model must implement `doEmbed`.
- Embedding calls must pass all input values to native embedding as texts.
- Embeddings returned as `Float32Array` values by the native binding must be converted to `number[][]`.
- Embedding usage must report native `totalTokens`.
- `maxEmbeddingsPerCall` must be `2048`.
- Embedding models must report `supportsParallelCalls: false`.
- Native embeddings must use mean pooling when loading an embedding context.
- Native embedding vectors must be L2-normalized when available.
- Empty input text must produce a zero vector from the native layer.

## Memory Safety

- `contextSize` must be a positive integer.
- Memory safety checks must be skipped when no model memory metadata is configured.
- `memorySafety.mode: "off"` must skip dynamic memory checks.
- When model metadata has `maxContextSize`, requested context sizes above it must throw unless clamp mode is enabled.
- When estimated memory exceeds the resolved budget, default behavior must throw.
- `memorySafety.mode: "clamp"` must reduce context size to the estimated safe value.
- Memory estimates must include model file size when available, multimodal projector file size when available, KV cache size, and compute overhead.
- KV-cache estimates must derive bytes per token from model memory metadata.

## Model Presets

- The package must export Gemma 4 and Qwen 3.6 model info presets.
- Presets must provide chat template information and memory metadata.
- Gemma 4 presets must provide reasoning marker configuration.

## Out Of Scope

- Windows and Linux runtime support are out of scope.
- Remote model loading is out of scope.
- Automatic expansion of `~` in model paths is out of scope.
- Loading image file URLs inside the provider is out of scope.
- Parallel calls on a single embedding model instance are out of scope.
