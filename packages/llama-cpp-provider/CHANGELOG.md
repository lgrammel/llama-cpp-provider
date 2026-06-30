# @lgrammel/llama-cpp-provider

## 0.4.2

### Patch Changes

- d4c8034: Strip leaked ChatML assistant headers and parse Qwen XML tool calls even when generated inside reasoning blocks.

## 0.4.1

### Patch Changes

- d0a6b5c: Fix structured output generation with grammar-constrained native sampling.
- 9f617a8: Fix Qwen 3.6 preset formatting and parse XML-style tool calls emitted by local agent models.

## 0.4.0

### Minor Changes

- d6043d0: Add reasoning effort token budgets, OpenAI-compatible structured output provider options, streamed tool-call input parts, and chat-template thinking controls.

### Patch Changes

- d6043d0: Cancel active native generation when streams are cancelled or models are disposed, constrain schema-less JSON responses to JSON objects, and improve streamed tool-call parsing with hidden reasoning content.

## 0.3.0

### Minor Changes

- dbbeb2d: Align tool calling and generation defaults with llama.cpp server behavior.

### Patch Changes

- d3a4da0: Fix lazy-load retry handling, streamed tool-call fallback text, and reasoning extraction defaults.
- e681e51: Add `logPrompts` to print the final llama.cpp rendered prompt for local debugging.

## 0.2.4

### Patch Changes

- 650a30f: Update the bundled llama.cpp revision.

## 0.2.3

### Patch Changes

- 20bb81d: Expose the configured context size on language model instances.
- 2a0f757: Add AbortSignal support for cancelling in-flight generation and streaming calls.

## 0.2.2

### Patch Changes

- 8e11c37: Fix prompt caching so generated assistant text stays in the native cache across later turns.

## 0.2.1

### Patch Changes

- 0037e59: Preserve the exact generated tool-call JSON when reusing cached prompts so follow-up tool-result turns can benefit from prefix cache hits.

## 0.2.0

### Minor Changes

- Add opt-in prefix prompt caching for language model calls, including cache read/write token accounting.

## 0.1.1

### Patch Changes

- Fix package installation on Apple Silicon by disabling llama.cpp native CPU feature probes during the native build.

## 0.1.0

### Minor Changes

- Initial public release.
