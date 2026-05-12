#ifndef LLAMA_WRAPPER_H
#define LLAMA_WRAPPER_H

#include <functional>
#include <memory>
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
  bool debug = false; // Show verbose llama.cpp output
  std::string chat_template =
      "auto"; // "auto" uses template from model, or specify a built-in template
};

struct ChatMessage {
  std::string role;
  std::string content;
  std::vector<std::vector<unsigned char>> images;
};

struct ContextParams {
  int n_ctx = 2048;       // Context size
  int n_batch = 512;      // Batch size for prompt processing
  int n_threads = 4;      // Number of threads
  bool embedding = false; // Enable embedding mode with mean pooling
};

struct GenerationParams {
  int max_tokens = 256;
  float temperature = 0.7f;
  float top_p = 0.9f;
  int top_k = 40;
  float repeat_penalty = 1.1f;
  std::vector<std::string> stop_sequences;
  std::string grammar; // GBNF grammar string for structured output
};

struct GenerationResult {
  std::string text;
  int prompt_tokens;
  int completion_tokens;
  std::string finish_reason; // "stop", "length", or "error"
  std::string error_message;
};

struct EmbeddingResult {
  std::vector<std::vector<float>> embeddings; // One embedding vector per input text
  int total_tokens;
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
                            const GenerationParams &params);

  // Generate text from messages (streaming)
  GenerationResult generate_streaming(const std::vector<ChatMessage> &messages,
                                      const GenerationParams &params, TokenCallback callback);

  // Generate embeddings for multiple texts
  EmbeddingResult embed(const std::vector<std::string> &texts);

private:
  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  mtmd_context *mtmd_ctx_ = nullptr;
  std::string model_path_;
  std::string mmproj_path_;
  std::string chat_template_;
  int n_batch_ = 512; // Batch size for prompt processing

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

  // Prefill the prompt into the context and return the number of consumed positions.
  bool prefill_prompt(const std::string &prompt,
                      const std::vector<std::vector<unsigned char>> &images,
                      GenerationResult &result, int &n_past);
};

} // namespace llama_wrapper

#endif // LLAMA_WRAPPER_H
