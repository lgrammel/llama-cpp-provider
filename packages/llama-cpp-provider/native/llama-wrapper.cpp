#include "llama-wrapper.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace llama_wrapper {

// Global debug flag for log callback
static bool g_debug_mode = false;

// Custom log callback that respects debug mode
static void llama_log_callback(ggml_log_level level, const char *text, void *user_data) {
  (void)level;
  (void)user_data;
  if (g_debug_mode) {
    fprintf(stderr, "%s", text);
  }
}

LlamaModel::LlamaModel() = default;

LlamaModel::~LlamaModel() {
  unload();
}

LlamaModel::LlamaModel(LlamaModel &&other) noexcept
    : model_(other.model_), ctx_(other.ctx_), sampler_(other.sampler_), mtmd_ctx_(other.mtmd_ctx_),
      chat_templates_(std::move(other.chat_templates_)), model_path_(std::move(other.model_path_)),
      mmproj_path_(std::move(other.mmproj_path_)), chat_template_(std::move(other.chat_template_)),
      log_prompts_(other.log_prompts_), n_batch_(other.n_batch_),
      cached_tokens_(std::move(other.cached_tokens_)) {
  other.model_ = nullptr;
  other.ctx_ = nullptr;
  other.sampler_ = nullptr;
  other.mtmd_ctx_ = nullptr;
  other.log_prompts_ = false;
}

LlamaModel &LlamaModel::operator=(LlamaModel &&other) noexcept {
  if (this != &other) {
    unload();
    model_ = other.model_;
    ctx_ = other.ctx_;
    sampler_ = other.sampler_;
    mtmd_ctx_ = other.mtmd_ctx_;
    chat_templates_ = std::move(other.chat_templates_);
    model_path_ = std::move(other.model_path_);
    mmproj_path_ = std::move(other.mmproj_path_);
    chat_template_ = std::move(other.chat_template_);
    log_prompts_ = other.log_prompts_;
    n_batch_ = other.n_batch_;
    cached_tokens_ = std::move(other.cached_tokens_);
    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.sampler_ = nullptr;
    other.mtmd_ctx_ = nullptr;
    other.log_prompts_ = false;
  }
  return *this;
}

bool LlamaModel::load(const ModelParams &params) {
  if (model_) {
    unload();
  }

  // Set debug mode and install log callback
  g_debug_mode = params.debug;
  llama_log_set(llama_log_callback, nullptr);

  // Initialize llama backend
  llama_backend_init();

  // Set up model parameters
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = params.n_gpu_layers;
  model_params.use_mmap = params.use_mmap;
  model_params.use_mlock = params.use_mlock;

  // Load the model
  model_ = llama_model_load_from_file(params.model_path.c_str(), model_params);
  if (!model_) {
    return false;
  }

  model_path_ = params.model_path;
  mmproj_path_ = params.mmproj_path;
  chat_template_ = params.chat_template;
  log_prompts_ = params.log_prompts;

  if (!mmproj_path_.empty()) {
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.n_gpu_layers > 0;
    mtmd_params.n_threads = params.n_threads;
    mtmd_params.print_timings = params.debug;

    mtmd_helper_log_set(llama_log_callback, nullptr);
    mtmd_ctx_ = mtmd_init_from_file(mmproj_path_.c_str(), model_, mtmd_params);
    if (!mtmd_ctx_ || !mtmd_support_vision(mtmd_ctx_)) {
      return false;
    }
  }

  return true;
}

bool LlamaModel::is_loaded() const {
  return model_ != nullptr;
}

void LlamaModel::unload() {
  if (sampler_) {
    llama_sampler_free(sampler_);
    sampler_ = nullptr;
  }
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  chat_templates_.reset();
  if (mtmd_ctx_) {
    mtmd_free(mtmd_ctx_);
    mtmd_ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
    llama_backend_free();
  }
  model_path_.clear();
  mmproj_path_.clear();
  chat_template_.clear();
  log_prompts_ = false;
  cached_tokens_.clear();
}

bool LlamaModel::create_context(const ContextParams &params) {
  if (!model_) {
    return false;
  }

  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
    cached_tokens_.clear();
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = params.n_ctx;
  ctx_params.n_batch = params.n_batch;
  ctx_params.n_threads = params.n_threads;
  ctx_params.n_threads_batch = params.n_threads;

  if (params.embedding) {
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  }

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (ctx_) {
    n_batch_ = params.n_batch; // Store batch size for chunked prefill
  }
  return ctx_ != nullptr;
}

void LlamaModel::normalize_embedding(float *embedding, int n_embd) {
  float sum = 0.0f;
  for (int i = 0; i < n_embd; i++) {
    sum += embedding[i] * embedding[i];
  }
  float norm = std::sqrt(sum);
  if (norm > 0.0f) {
    for (int i = 0; i < n_embd; i++) {
      embedding[i] /= norm;
    }
  }
}

EmbeddingResult LlamaModel::embed(const std::vector<std::string> &texts) {
  std::lock_guard<std::mutex> lock(inference_mutex_);
  EmbeddingResult result;
  result.total_tokens = 0;

  if (!ctx_ || !model_) {
    return result;
  }

  const int n_embd = llama_model_n_embd(model_);
  const enum llama_pooling_type pooling_type = llama_pooling_type(ctx_);

  // Process each text
  for (size_t seq_id = 0; seq_id < texts.size(); seq_id++) {
    const std::string &text = texts[seq_id];

    // Tokenize the text
    std::vector<int32_t> tokens = tokenize(text, true);
    result.total_tokens += tokens.size();

    if (tokens.empty()) {
      // Return zero embedding for empty text
      result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
      continue;
    }

    clear_context_memory(true);
    cached_tokens_.clear();

    // Create batch with sequence ID
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = seq_id;
      batch.logits[i] = true; // We want embeddings for all tokens
    }
    batch.n_tokens = tokens.size();

    // Decode to get embeddings
    if (llama_decode(ctx_, batch) != 0) {
      llama_batch_free(batch);
      result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
      continue;
    }

    // Extract embedding based on pooling type
    const float *embd = nullptr;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
      // Get embedding for last token
      embd = llama_get_embeddings_ith(ctx_, tokens.size() - 1);
    } else {
      // Get pooled embedding for the sequence
      embd = llama_get_embeddings_seq(ctx_, seq_id);
    }

    if (embd) {
      std::vector<float> embedding(n_embd);
      std::copy(embd, embd + n_embd, embedding.begin());
      // Normalize the embedding (L2 normalization)
      normalize_embedding(embedding.data(), n_embd);
      result.embeddings.push_back(std::move(embedding));
    } else {
      result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
    }

    llama_batch_free(batch);
  }

  return result;
}

std::string LlamaModel::apply_chat_template(const std::vector<ChatMessage> &messages) {
  if (!model_) {
    return "";
  }

  // Determine which template to use
  const char *tmpl = nullptr;
  if (chat_template_ == "auto") {
    // Use the template embedded in the model
    tmpl = llama_model_chat_template(model_, nullptr);
  } else {
    // Use the specified template name
    tmpl = chat_template_.c_str();
  }

  // Convert messages to llama_chat_message format
  std::vector<llama_chat_message> chat_messages;
  chat_messages.reserve(messages.size());
  for (const auto &msg : messages) {
    llama_chat_message chat_msg;
    chat_msg.role = msg.role.c_str();
    chat_msg.content = msg.content.c_str();
    chat_messages.push_back(chat_msg);
  }

  // First call to get required buffer size
  int32_t result_size = llama_chat_apply_template(tmpl, chat_messages.data(), chat_messages.size(),
                                                  true, // add_ass: add assistant prompt
                                                  nullptr, 0);

  if (result_size < 0) {
    // Template not supported, return empty string
    return "";
  }

  // Allocate buffer and apply template
  std::vector<char> buffer(result_size + 1);
  llama_chat_apply_template(tmpl, chat_messages.data(), chat_messages.size(), true, buffer.data(),
                            buffer.size());

  return std::string(buffer.data(), result_size);
}

bool LlamaModel::ensure_chat_templates(GenerationResult &result) {
  if (chat_templates_) {
    return true;
  }

  if (!model_) {
    result.error_message = "Model is not loaded";
    return false;
  }

  std::string template_override;
  if (chat_template_ == "chatml") {
    template_override = "chatml";
  } else if (chat_template_ != "auto" && (chat_template_.find("{%") != std::string::npos ||
                                          chat_template_.find("{{") != std::string::npos)) {
    template_override = chat_template_;
  }

  try {
    chat_templates_ = common_chat_templates_init(model_, template_override);
    return true;
  } catch (const std::exception &e) {
    result.error_message =
        std::string("Failed to initialize llama.cpp chat templates: ") + e.what();
    return false;
  }
}

std::vector<common_chat_msg>
LlamaModel::to_common_chat_messages(const std::vector<ChatMessage> &messages) const {
  std::vector<common_chat_msg> result;
  result.reserve(messages.size());

  for (const auto &message : messages) {
    common_chat_msg msg;
    msg.role = message.role;
    msg.content = message.content;
    msg.tool_calls = message.tool_calls;
    msg.tool_name = message.tool_name;
    msg.tool_call_id = message.tool_call_id;
    result.push_back(std::move(msg));
  }

  return result;
}

std::vector<common_chat_tool>
LlamaModel::to_common_chat_tools(const std::vector<ToolDefinition> &tools) const {
  std::vector<common_chat_tool> result;
  result.reserve(tools.size());

  for (const auto &tool : tools) {
    result.push_back({
        /* .name = */ tool.name,
        /* .description = */ tool.description,
        /* .parameters = */ tool.parameters.empty() ? "{}" : tool.parameters,
    });
  }

  return result;
}

static common_chat_tool_choice parse_tool_choice(const std::string &tool_choice) {
  if (tool_choice == "required") {
    return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
  }
  if (tool_choice == "none") {
    return COMMON_CHAT_TOOL_CHOICE_NONE;
  }
  return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

bool LlamaModel::prepare_prompt(const std::vector<ChatMessage> &messages,
                                const GenerationParams &params, GenerationResult &result,
                                std::string &prompt, GenerationParams &effective_params,
                                common_chat_parser_params &parser_params, bool &parse_tool_calls) {
  effective_params = params;
  parse_tool_calls = false;

  if (params.tools.empty()) {
    prompt = apply_chat_template(messages);
    return !prompt.empty();
  }

  if (!params.grammar.empty()) {
    result.error_message = "Cannot use custom grammar constraints with tools.";
    return false;
  }

  if (!ensure_chat_templates(result)) {
    return false;
  }

  common_chat_templates_inputs inputs;
  inputs.messages = to_common_chat_messages(messages);
  inputs.tools = to_common_chat_tools(params.tools);
  inputs.tool_choice = parse_tool_choice(params.tool_choice);
  inputs.parallel_tool_calls = params.parallel_tool_calls;
  inputs.use_jinja = true;
  inputs.add_generation_prompt = true;
  inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

  try {
    common_chat_params chat_params = common_chat_templates_apply(chat_templates_.get(), inputs);
    prompt = chat_params.prompt;
    effective_params.grammar = chat_params.grammar;
    effective_params.stop_sequences.insert(effective_params.stop_sequences.end(),
                                           chat_params.additional_stops.begin(),
                                           chat_params.additional_stops.end());
    parser_params = common_chat_parser_params(chat_params);
    parser_params.parse_tool_calls = true;
    parse_tool_calls = true;
    return !prompt.empty();
  } catch (const std::exception &e) {
    result.error_message = std::string("Failed to apply llama.cpp chat template: ") + e.what();
    return false;
  }
}

void LlamaModel::parse_generated_message(GenerationResult &result,
                                         const common_chat_parser_params &parser_params,
                                         bool parse_tool_calls) const {
  if (!parse_tool_calls || result.finish_reason == "error") {
    return;
  }

  try {
    common_chat_msg msg = common_chat_parse(result.text, false, parser_params);
    if (msg.empty()) {
      return;
    }

    result.text = msg.render_content();
    result.tool_calls.clear();
    result.tool_calls.reserve(msg.tool_calls.size());
    for (const auto &tool_call : msg.tool_calls) {
      result.tool_calls.push_back({
          /* .id = */ tool_call.id,
          /* .name = */ tool_call.name,
          /* .arguments = */ tool_call.arguments,
      });
    }
  } catch (const std::exception &) {
    // If the native parser cannot parse a model response, keep the raw text so
    // TypeScript can still fall back to the legacy JSON parser.
  }
}

void LlamaModel::create_sampler(const GenerationParams &params) {
  if (sampler_) {
    llama_sampler_free(sampler_);
  }

  // Create a sampler chain
  sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());

  // Add grammar sampler first if grammar is provided (constrains token generation)
  if (!params.grammar.empty()) {
    const llama_vocab *vocab = llama_model_get_vocab(model_);
    llama_sampler *grammar_sampler =
        llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
    if (grammar_sampler) {
      llama_sampler_chain_add(sampler_, grammar_sampler);
    }
  }

  // Add samplers to the chain
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(params.top_k));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(params.top_p, 1));
  llama_sampler_chain_add(sampler_, llama_sampler_init_temp(params.temperature));
  llama_sampler_chain_add(sampler_, llama_sampler_init_dist(params.seed));
}

static bool ends_with(const std::string &text, const std::string &suffix) {
  return suffix.size() <= text.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static size_t find_stop_sequence_suffix(const std::string &text,
                                        const std::vector<std::string> &stop_sequences,
                                        size_t &matched_stop_length) {
  matched_stop_length = 0;
  size_t keep_length = 0;

  for (const auto &stop_seq : stop_sequences) {
    if (stop_seq.empty()) {
      continue;
    }

    if (ends_with(text, stop_seq)) {
      matched_stop_length = stop_seq.size();
      return stop_seq.size();
    }

    const size_t max_prefix_length = std::min(stop_seq.size() - 1, text.size());
    for (size_t prefix_length = max_prefix_length; prefix_length > 0; prefix_length--) {
      if (text.compare(text.size() - prefix_length, prefix_length, stop_seq, 0, prefix_length) ==
          0) {
        keep_length = std::max(keep_length, prefix_length);
        break;
      }
    }
  }

  return keep_length;
}

std::vector<int32_t> LlamaModel::tokenize(const std::string &text, bool add_bos) {
  const llama_vocab *vocab = llama_model_get_vocab(model_);

  // First, get the number of tokens needed
  // When passing 0 for n_tokens_max, llama_tokenize returns negative of required size
  int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, add_bos, true);

  if (n_tokens < 0) {
    n_tokens = -n_tokens; // Convert to positive size
  }

  if (n_tokens == 0) {
    return {}; // Empty input
  }

  std::vector<int32_t> tokens(n_tokens);
  int actual_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(),
                                     tokens.size(), add_bos, true);

  if (actual_tokens < 0) {
    // Buffer still too small, resize and try again
    tokens.resize(-actual_tokens);
    actual_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(),
                                   add_bos, true);
  }

  if (actual_tokens > 0) {
    tokens.resize(actual_tokens);
  } else {
    tokens.clear();
  }

  return tokens;
}

std::string LlamaModel::detokenize(int32_t token) {
  const llama_vocab *vocab = llama_model_get_vocab(model_);

  char buf[256];
  int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
  if (n < 0) {
    return "";
  }
  return std::string(buf, n);
}

bool LlamaModel::is_eos_token(int32_t token) {
  const llama_vocab *vocab = llama_model_get_vocab(model_);
  return llama_vocab_is_eog(vocab, token);
}

void LlamaModel::clear_context_memory(bool data) {
  llama_memory_t mem = llama_get_memory(ctx_);
  if (mem) {
    llama_memory_clear(mem, data);
  }
}

bool LlamaModel::is_cancelled(GenerationResult &result,
                              const CancellationToken &cancellation) const {
  if (!cancellation.cancelled.load()) {
    return false;
  }

  result.finish_reason = "error";
  result.error_message = "Generation aborted";
  return true;
}

bool LlamaModel::trim_cached_tokens(size_t keep_tokens) {
  if (keep_tokens >= cached_tokens_.size()) {
    return true;
  }

  llama_memory_t mem = llama_get_memory(ctx_);
  if (!mem) {
    cached_tokens_.clear();
    return false;
  }

  const bool removed_suffix = llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(keep_tokens), -1);
  if (!removed_suffix) {
    clear_context_memory(true);
    cached_tokens_.clear();
    return false;
  }

  cached_tokens_.resize(keep_tokens);
  return true;
}

bool LlamaModel::decode_tokens(const std::vector<int32_t> &tokens, size_t offset, size_t count,
                               int start_pos, bool logits_last, GenerationResult &result,
                               const std::string &error_message,
                               const CancellationToken &cancellation) {
  size_t n_processed = 0;

  while (n_processed < count) {
    if (is_cancelled(result, cancellation)) {
      return false;
    }

    const size_t n_chunk = std::min(static_cast<size_t>(n_batch_), count - n_processed);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_chunk), 0, 1);

    for (size_t i = 0; i < n_chunk; i++) {
      const size_t token_index = offset + n_processed + i;
      batch.token[i] = tokens[token_index];
      batch.pos[i] = static_cast<llama_pos>(start_pos + n_processed + i);
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = logits_last && n_processed + i == count - 1;
    }
    batch.n_tokens = static_cast<int32_t>(n_chunk);

    if (llama_decode(ctx_, batch) != 0) {
      llama_batch_free(batch);
      if (!is_cancelled(result, cancellation)) {
        result.error_message = error_message;
      }
      return false;
    }

    llama_batch_free(batch);
    n_processed += n_chunk;
  }

  return true;
}

bool LlamaModel::sync_cached_tokens_to_text(const std::string &text) {
  GenerationResult sync_result;
  CancellationToken cancellation;
  return sync_prompt_cache_to_text(cached_tokens_, text,
                                   create_prompt_cache_ops(sync_result, cancellation));
}

PromptCacheOps LlamaModel::create_prompt_cache_ops(GenerationResult &result,
                                                   const CancellationToken &cancellation) {
  return {
      [this](const std::string &text) { return tokenize(text, true); },
      [this](size_t keep_tokens) { return trim_cached_tokens(keep_tokens); },
      [this]() { clear_context_memory(true); },
      [this, &result, &cancellation](const TokenList &tokens, size_t offset, size_t count,
                                     int start_pos) {
        return decode_tokens(tokens, offset, count, start_pos, true, result,
                             "Failed to decode cached tokens", cancellation);
      },
  };
}

bool LlamaModel::prefill_prompt(const std::string &prompt,
                                const std::vector<std::vector<unsigned char>> &images,
                                const GenerationParams &params, GenerationResult &result,
                                int &n_past, const CancellationToken &cancellation) {
  n_past = 0;

  if (is_cancelled(result, cancellation)) {
    return false;
  }

  if (images.empty()) {
    // Tokenize the prompt
    std::vector<int32_t> prompt_tokens = tokenize(prompt, true);
    result.prompt_tokens = prompt_tokens.size();

    const int n_ctx = llama_n_ctx(ctx_);
    if (result.prompt_tokens + params.max_tokens > n_ctx) {
      result.error_message = "Prompt tokens plus max_tokens exceed context size";
      return false;
    }

    const PromptCachePrefillResult cache_result =
        prefill_prompt_cache(cached_tokens_, prompt_tokens, params.prompt_cache,
                             create_prompt_cache_ops(result, cancellation));
    if (!cache_result.ok) {
      return false;
    }

    result.cache_read_tokens = cache_result.cache_read_tokens;
    result.cache_write_tokens = cache_result.cache_write_tokens;
    n_past = cache_result.n_past;
    return true;
  }

  clear_context_memory(true);
  cached_tokens_.clear();

  if (!mtmd_ctx_) {
    result.error_message = "Image inputs require mmprojPath to be configured";
    return false;
  }

  std::vector<mtmd_bitmap *> bitmaps;
  std::vector<mtmd_helper_video *> video_contexts;
  std::vector<const mtmd_bitmap *> bitmap_ptrs;
  bitmaps.reserve(images.size());
  video_contexts.reserve(images.size());
  bitmap_ptrs.reserve(images.size());

  auto free_media = [&bitmaps, &video_contexts]() {
    for (mtmd_bitmap *bitmap : bitmaps) {
      mtmd_bitmap_free(bitmap);
    }
    for (mtmd_helper_video *video_context : video_contexts) {
      mtmd_helper_video_free(video_context);
    }
  };

  for (const auto &image : images) {
    mtmd_helper_bitmap_wrapper bitmap_result =
        mtmd_helper_bitmap_init_from_buf(mtmd_ctx_, image.data(), image.size(), false);
    mtmd_bitmap *bitmap = bitmap_result.bitmap;
    if (!bitmap) {
      free_media();
      result.error_message = "Failed to decode image input";
      return false;
    }
    bitmaps.push_back(bitmap);
    if (bitmap_result.video_ctx) {
      video_contexts.push_back(bitmap_result.video_ctx);
    }
    bitmap_ptrs.push_back(bitmap);
  }

  mtmd_input_chunks *chunks = mtmd_input_chunks_init();
  mtmd_input_text text;
  text.text = prompt.c_str();
  text.add_special = true;
  text.parse_special = true;

  int32_t tokenize_result =
      mtmd_tokenize(mtmd_ctx_, chunks, &text, bitmap_ptrs.data(), bitmap_ptrs.size());
  if (tokenize_result != 0) {
    mtmd_input_chunks_free(chunks);
    free_media();
    result.error_message = "Failed to tokenize multimodal prompt";
    return false;
  }

  llama_pos new_n_past = 0;
  int32_t eval_result =
      mtmd_helper_eval_chunks(mtmd_ctx_, ctx_, chunks, 0, 0, n_batch_, true, &new_n_past);

  result.prompt_tokens = static_cast<int>(mtmd_helper_get_n_tokens(chunks));
  mtmd_input_chunks_free(chunks);
  free_media();

  if (is_cancelled(result, cancellation)) {
    return false;
  }

  if (eval_result != 0) {
    result.error_message = "Failed to decode multimodal prompt";
    return false;
  }

  n_past = static_cast<int>(new_n_past);
  result.cache_write_tokens = n_past;
  const int n_ctx = llama_n_ctx(ctx_);
  if (n_past + params.max_tokens > n_ctx) {
    result.error_message = "Prompt tokens plus max_tokens exceed context size";
    return false;
  }
  return true;
}

GenerationResult LlamaModel::generate(const std::vector<ChatMessage> &messages,
                                      const GenerationParams &params,
                                      const CancellationToken &cancellation) {
  std::lock_guard<std::mutex> lock(inference_mutex_);
  GenerationResult result;
  result.finish_reason = "error";
  result.prompt_tokens = 0;
  result.completion_tokens = 0;
  result.cache_read_tokens = 0;
  result.cache_write_tokens = 0;

  if (is_cancelled(result, cancellation)) {
    return result;
  }

  if (!ctx_ || !model_) {
    result.error_message = "Model context is not initialized";
    return result;
  }

  std::string prompt;
  GenerationParams effective_params;
  common_chat_parser_params parser_params;
  bool parse_tool_calls = false;
  if (!prepare_prompt(messages, params, result, prompt, effective_params, parser_params,
                      parse_tool_calls)) {
    if (result.error_message.empty()) {
      result.error_message =
          "Failed to apply chat template. Try setting chatTemplate explicitly, for example "
          "'gemma', or use debug: true for llama.cpp template diagnostics.";
    }
    return result;
  }
  if (prompt.empty()) {
    result.error_message =
        "Failed to apply chat template. Try setting chatTemplate explicitly, for example "
        "'gemma', or use debug: true for llama.cpp template diagnostics.";
    return result;
  }
  if (log_prompts_) {
    fprintf(stderr, "\n--- llama.cpp rendered prompt ---\n%s\n--- end prompt ---\n",
            prompt.c_str());
  }

  std::vector<std::vector<unsigned char>> images;
  for (const auto &message : messages) {
    images.insert(images.end(), message.images.begin(), message.images.end());
  }

  // Create sampler
  create_sampler(effective_params);

  int n_past = 0;
  if (!prefill_prompt(prompt, images, effective_params, result, n_past, cancellation)) {
    return result;
  }

  // Generate tokens
  std::string generated_text;
  int n_cur = n_past;

  for (int i = 0; i < effective_params.max_tokens; i++) {
    if (is_cancelled(result, cancellation)) {
      break;
    }

    // Sample the next token
    int32_t new_token = llama_sampler_sample(sampler_, ctx_, -1);

    // Check for end of sequence
    if (is_eos_token(new_token)) {
      result.finish_reason = "stop";
      break;
    }

    // Convert token to string
    std::string token_str = detokenize(new_token);
    generated_text += token_str;
    result.completion_tokens++;

    // Check for stop sequences
    bool should_stop = false;
    for (const auto &stop_seq : effective_params.stop_sequences) {
      if (generated_text.length() >= stop_seq.length()) {
        if (generated_text.substr(generated_text.length() - stop_seq.length()) == stop_seq) {
          // Remove the stop sequence from output
          generated_text = generated_text.substr(0, generated_text.length() - stop_seq.length());
          should_stop = true;
          result.finish_reason = "stop";
          break;
        }
      }
    }
    if (should_stop)
      break;

    std::vector<int32_t> token = {new_token};
    cached_tokens_.push_back(new_token);
    if (!decode_tokens(token, 0, 1, n_cur, true, result, "Failed to decode generated token",
                       cancellation)) {
      if (result.error_message != "Generation aborted") {
        result.error_message = "Failed to decode generated token";
      }
      cached_tokens_.pop_back();
      break;
    }
    n_cur++;
  }

  if (!effective_params.prompt_cache) {
    cached_tokens_.clear();
  } else {
    sync_cached_tokens_to_text(prompt + generated_text);
  }

  if (result.error_message == "Generation aborted") {
    cached_tokens_.clear();
  } else if (result.finish_reason == "error" &&
             result.completion_tokens >= effective_params.max_tokens) {
    result.finish_reason = "length";
  } else if (result.finish_reason == "error" && result.error_message.empty()) {
    result.finish_reason = "stop";
  }

  result.text = generated_text;
  parse_generated_message(result, parser_params, parse_tool_calls);
  return result;
}

GenerationResult LlamaModel::generate_streaming(const std::vector<ChatMessage> &messages,
                                                const GenerationParams &params,
                                                TokenCallback callback,
                                                const CancellationToken &cancellation) {
  std::lock_guard<std::mutex> lock(inference_mutex_);
  GenerationResult result;
  result.finish_reason = "error";
  result.prompt_tokens = 0;
  result.completion_tokens = 0;
  result.cache_read_tokens = 0;
  result.cache_write_tokens = 0;

  if (is_cancelled(result, cancellation)) {
    return result;
  }

  if (!ctx_ || !model_) {
    result.error_message = "Model context is not initialized";
    return result;
  }

  std::string prompt;
  GenerationParams effective_params;
  common_chat_parser_params parser_params;
  bool parse_tool_calls = false;
  if (!prepare_prompt(messages, params, result, prompt, effective_params, parser_params,
                      parse_tool_calls)) {
    if (result.error_message.empty()) {
      result.error_message =
          "Failed to apply chat template. Try setting chatTemplate explicitly, for example "
          "'gemma', or use debug: true for llama.cpp template diagnostics.";
    }
    return result;
  }
  if (prompt.empty()) {
    result.error_message =
        "Failed to apply chat template. Try setting chatTemplate explicitly, for example "
        "'gemma', or use debug: true for llama.cpp template diagnostics.";
    return result;
  }
  if (log_prompts_) {
    fprintf(stderr, "\n--- llama.cpp rendered prompt ---\n%s\n--- end prompt ---\n",
            prompt.c_str());
  }

  std::vector<std::vector<unsigned char>> images;
  for (const auto &message : messages) {
    images.insert(images.end(), message.images.begin(), message.images.end());
  }

  // Create sampler
  create_sampler(effective_params);

  int n_past = 0;
  if (!prefill_prompt(prompt, images, effective_params, result, n_past, cancellation)) {
    return result;
  }

  // Generate tokens
  std::string generated_text;
  std::string pending_text;
  int n_cur = n_past;

  for (int i = 0; i < effective_params.max_tokens; i++) {
    if (is_cancelled(result, cancellation)) {
      break;
    }

    // Sample the next token
    int32_t new_token = llama_sampler_sample(sampler_, ctx_, -1);

    // Check for end of sequence
    if (is_eos_token(new_token)) {
      result.finish_reason = "stop";
      break;
    }

    // Convert token to string
    std::string token_str = detokenize(new_token);
    generated_text += token_str;
    pending_text += token_str;
    result.completion_tokens++;

    size_t matched_stop_length = 0;
    const size_t keep_length = find_stop_sequence_suffix(
        generated_text, effective_params.stop_sequences, matched_stop_length);

    if (matched_stop_length > 0) {
      generated_text.resize(generated_text.size() - matched_stop_length);
      if (pending_text.size() >= matched_stop_length) {
        pending_text.resize(pending_text.size() - matched_stop_length);
      } else {
        pending_text.clear();
      }

      if (!pending_text.empty() && !callback(pending_text)) {
        if (!is_cancelled(result, cancellation)) {
          result.finish_reason = "stop";
        }
        break;
      }

      pending_text.clear();
      result.finish_reason = "stop";
      break;
    }

    if (pending_text.size() > keep_length) {
      const size_t emit_length = pending_text.size() - keep_length;
      std::string emit_text = pending_text.substr(0, emit_length);
      pending_text.erase(0, emit_length);

      if (!callback(emit_text)) {
        if (!is_cancelled(result, cancellation)) {
          result.finish_reason = "stop";
        }
        break;
      }
    }

    std::vector<int32_t> token = {new_token};
    cached_tokens_.push_back(new_token);
    if (!decode_tokens(token, 0, 1, n_cur, true, result, "Failed to decode generated token",
                       cancellation)) {
      if (result.error_message != "Generation aborted") {
        result.error_message = "Failed to decode generated token";
      }
      cached_tokens_.pop_back();
      break;
    }
    n_cur++;
  }

  if (!pending_text.empty() && !is_cancelled(result, cancellation)) {
    if (!callback(pending_text) && result.finish_reason == "error") {
      result.finish_reason = "stop";
    }
  }

  if (!effective_params.prompt_cache) {
    cached_tokens_.clear();
  } else {
    sync_cached_tokens_to_text(prompt + generated_text);
  }

  if (result.error_message == "Generation aborted") {
    cached_tokens_.clear();
  } else if (result.finish_reason == "error" &&
             result.completion_tokens >= effective_params.max_tokens) {
    result.finish_reason = "length";
  } else if (result.finish_reason == "error" && result.error_message.empty()) {
    result.finish_reason = "stop";
  }

  result.text = generated_text;
  parse_generated_message(result, parser_params, parse_tool_calls);
  return result;
}

} // namespace llama_wrapper
