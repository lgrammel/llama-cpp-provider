# @lgrammel/llama-cpp-provider

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
