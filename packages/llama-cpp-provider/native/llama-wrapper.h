#ifndef LLAMA_WRAPPER_H
#define LLAMA_WRAPPER_H

#include "chat.h"
#include "prompt-cache.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_sampler;
struct mtmd_context;

namespace llama_wrapper {

struct ModelParams {
  std::string model_path;
  std::string mmproj_path;
  int n_gpu_layers = 99; // Use GPU by default if available
  int n_threads = 4;
  bool use_mmap = true;
  bool use_mlock = false;
  bool debug = false;       // Show verbose llama.cpp output
  bool log_prompts = false; // Print rendered prompts to stderr for debugging
  std::string chat_template =
      "auto"; // "auto" uses template from model, or specify a built-in template
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::vector<std::vector<unsigned char>> images;
  std::vector<common_chat_tool_call> tool_calls;
  std::string tool_name;
  std::string tool_call_id;
};

struct ToolDefinition {
  std::string name;
  std::string description;
  std::string parameters;
};

struct ContextParams {
  int n_ctx = 2048;       // Context size
  int n_batch = 512;      // Batch size for prompt processing
  int n_threads = 4;      // Number of threads
  bool embedding = false; // Enable embedding mode with mean pooling
};

struct GenerationParams {
  int max_tokens = -1;
  float temperature = 0.8f;
  float top_p = 0.95f;
  int top_k = 40;
  uint32_t seed = 0xFFFFFFFFu; // LLAMA_DEFAULT_SEED: use a random seed
  float repeat_penalty = 1.1f;
  std::vector<std::string> stop_sequences;
  std::string grammar; // GBNF grammar string for structured output
  int32_t reasoning_budget_tokens = -1;
  std::string reasoning_budget_start;
  std::string reasoning_budget_end;
  bool enable_thinking = true;
  bool prompt_cache = false;
  std::vector<ToolDefinition> tools;
  std::string tool_choice = "auto";
  bool parallel_tool_calls = false;
};

struct ParsedToolCall {
  std::string id;
  std::string name;
  std::string arguments;
};

struct GenerationResult {
  std::string text;
  int prompt_tokens;
  int completion_tokens;
  int cache_read_tokens;
  int cache_write_tokens;
  std::string finish_reason; // "stop", "length", or "error"
  std::string error_message;
  std::vector<ParsedToolCall> tool_calls;
};

struct EmbeddingResult {
  std::vector<std::vector<float>> embeddings; // One embedding vector per input text
  int total_tokens;
};

struct CancellationToken {
  std::atomic<bool> cancelled{false};
};

// Token callback for streaming: returns false to stop generation
using TokenCallback = std::function<bool(const std::string &token)>;

class LlamaModel {
public:
  LlamaModel();
  ~LlamaModel();

  // Disable copy
  LlamaModel(const LlamaModel &) = delete;
  LlamaModel &operator=(const LlamaModel &) = delete;

  // Enable move
  LlamaModel(LlamaModel &&other) noexcept;
  LlamaModel &operator=(LlamaModel &&other) noexcept;

  // Load a model from a GGUF file
  bool load(const ModelParams &params);

  // Check if model is loaded
  bool is_loaded() const;

  // Unload the model
  void unload();

  // Get the model path
  const std::string &get_model_path() const { return model_path_; }

  // Create a context for inference (or embeddings if params.embedding is true)
  bool create_context(const ContextParams &params);

  // Apply chat template to messages and return formatted prompt
  std::string apply_chat_template(const std::vector<ChatMessage> &messages);

  // Generate text from messages (non-streaming)
  GenerationResult generate(const std::vector<ChatMessage> &messages,
                            const GenerationParams &params, const CancellationToken &cancellation);

  // Generate text from messages (streaming)
  GenerationResult generate_streaming(const std::vector<ChatMessage> &messages,
                                      const GenerationParams &params, TokenCallback callback,
                                      const CancellationToken &cancellation);

  // Generate embeddings for multiple texts
  EmbeddingResult embed(const std::vector<std::string> &texts);

private:
  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  mtmd_context *mtmd_ctx_ = nullptr;
  common_chat_templates_ptr chat_templates_;
  std::string model_path_;
  std::string mmproj_path_;
  std::string chat_template_;
  bool log_prompts_ = false;
  int n_batch_ = 512; // Batch size for prompt processing
  TokenList cached_tokens_;
  std::mutex inference_mutex_;

  // Tokenize a string
  std::vector<int32_t> tokenize(const std::string &text, bool add_bos);

  // Normalize an embedding vector (L2 normalization)
  static void normalize_embedding(float *embedding, int n_embd);

  // Detokenize a single token
  std::string detokenize(int32_t token);

  // Create sampler with given params
  void create_sampler(const GenerationParams &params);

  // Check if token is end-of-sequence
  bool is_eos_token(int32_t token);

  void clear_context_memory(bool data);
  bool is_cancelled(GenerationResult &result, const CancellationToken &cancellation) const;
  bool trim_cached_tokens(size_t keep_tokens);
  bool decode_tokens(const std::vector<int32_t> &tokens, size_t offset, size_t count, int start_pos,
                     bool logits_last, GenerationResult &result, const std::string &error_message,
                     const CancellationToken &cancellation);
  bool sync_cached_tokens_to_text(const std::string &text);
  PromptCacheOps create_prompt_cache_ops(GenerationResult &result,
                                         const CancellationToken &cancellation);

  // Prefill the prompt into the context and return the number of consumed positions.
  bool prefill_prompt(const std::string &prompt,
                      const std::vector<std::vector<unsigned char>> &images,
                      const GenerationParams &params, GenerationResult &result, int &n_past,
                      const CancellationToken &cancellation);
  bool prepare_prompt(const std::vector<ChatMessage> &messages, const GenerationParams &params,
                      GenerationResult &result, std::string &prompt,
                      GenerationParams &effective_params, common_chat_parser_params &parser_params,
                      bool &parse_tool_calls);
  bool ensure_chat_templates(GenerationResult &result);
  bool apply_common_chat_template(const std::vector<common_chat_msg> &messages,
                                  const GenerationParams &params, bool add_generation_prompt,
                                  common_chat_params &chat_params, GenerationResult &result);
  std::vector<common_chat_msg>
  to_common_chat_messages(const std::vector<ChatMessage> &messages) const;
  std::vector<common_chat_tool>
  to_common_chat_tools(const std::vector<ToolDefinition> &tools) const;
  std::optional<common_chat_msg>
  parse_generated_message(GenerationResult &result, const common_chat_parser_params &parser_params,
                          bool parse_tool_calls) const;
  std::string
  prompt_cache_text_after_generation(const std::vector<ChatMessage> &messages,
                                     const GenerationParams &params, const std::string &prompt,
                                     const std::string &generated_text,
                                     const std::optional<common_chat_msg> &parsed_message);
};

} // namespace llama_wrapper

#endif // LLAMA_WRAPPER_H
