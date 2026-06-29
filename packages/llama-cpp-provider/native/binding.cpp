#include "llama-wrapper.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <napi.h>
#include <unordered_map>
#include <vector>

class GenerateWorker;
class StreamGenerateWorker;

// Global state for managing models
static std::unordered_map<int, std::shared_ptr<llama_wrapper::LlamaModel>> g_models;
static std::mutex g_models_mutex;
static std::atomic<int> g_next_handle{1};

static std::unordered_map<int, std::vector<GenerateWorker *>> g_generate_workers;
static std::unordered_map<int, std::vector<StreamGenerateWorker *>> g_stream_generate_workers;
static std::mutex g_generation_workers_mutex;

template <typename Worker>
void RemoveWorker(std::unordered_map<int, std::vector<Worker *>> &workers_by_handle, int handle,
                  Worker *worker) {
  auto it = workers_by_handle.find(handle);
  if (it == workers_by_handle.end()) {
    return;
  }

  auto &workers = it->second;
  workers.erase(std::remove(workers.begin(), workers.end(), worker), workers.end());
  if (workers.empty()) {
    workers_by_handle.erase(it);
  }
}

// ============================================================================
// Async Workers
// ============================================================================

class LoadModelWorker : public Napi::AsyncWorker {
public:
  LoadModelWorker(Napi::Function &callback, const std::string &model_path,
                  const std::string &mmproj_path, int n_gpu_layers, int n_ctx, int n_threads,
                  bool debug, bool log_prompts, const std::string &chat_template, bool embedding)
      : Napi::AsyncWorker(callback), model_path_(model_path), mmproj_path_(mmproj_path),
        n_gpu_layers_(n_gpu_layers), n_ctx_(n_ctx), n_threads_(n_threads), debug_(debug),
        log_prompts_(log_prompts), chat_template_(chat_template), embedding_(embedding),
        handle_(-1), success_(false) {}

  void Execute() override {
    auto model = std::make_unique<llama_wrapper::LlamaModel>();

    llama_wrapper::ModelParams model_params;
    model_params.model_path = model_path_;
    model_params.mmproj_path = mmproj_path_;
    model_params.n_gpu_layers = n_gpu_layers_;
    model_params.n_threads = n_threads_;
    model_params.debug = debug_;
    model_params.log_prompts = log_prompts_;
    model_params.chat_template = chat_template_;

    if (!model->load(model_params)) {
      SetError("Failed to load model from: " + model_path_);
      return;
    }

    llama_wrapper::ContextParams ctx_params;
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads_;
    ctx_params.embedding = embedding_;

    if (!model->create_context(ctx_params)) {
      SetError("Failed to create context");
      return;
    }

    handle_ = g_next_handle++;

    {
      std::lock_guard<std::mutex> lock(g_models_mutex);
      g_models[handle_] = std::shared_ptr<llama_wrapper::LlamaModel>(std::move(model));
    }

    success_ = true;
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    Callback().Call({Env().Null(), Napi::Number::New(Env(), handle_)});
  }

  void OnError(const Napi::Error &e) override {
    Napi::HandleScope scope(Env());
    Callback().Call({Napi::String::New(Env(), e.Message()), Env().Null()});
  }

private:
  std::string model_path_;
  std::string mmproj_path_;
  int n_gpu_layers_;
  int n_ctx_;
  int n_threads_;
  bool debug_;
  bool log_prompts_;
  std::string chat_template_;
  bool embedding_;
  int handle_;
  bool success_;
};

class GenerateWorker : public Napi::AsyncWorker {
public:
  GenerateWorker(Napi::Function &callback, int handle,
                 const std::vector<llama_wrapper::ChatMessage> &messages,
                 const llama_wrapper::GenerationParams &params)
      : Napi::AsyncWorker(callback), handle_(handle), messages_(messages), params_(params),
        cancellation_(std::make_shared<llama_wrapper::CancellationToken>()) {}

  void Cancel() { cancellation_->cancelled.store(true); }

  void Execute() override {
    std::shared_ptr<llama_wrapper::LlamaModel> model;

    {
      std::lock_guard<std::mutex> lock(g_models_mutex);
      auto it = g_models.find(handle_);
      if (it == g_models.end()) {
        SetError("Invalid model handle");
        return;
      }
      model = it->second;
    }

    result_ = model->generate(messages_, params_, *cancellation_);
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());

    {
      std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
      RemoveWorker(g_generate_workers, handle_, this);
    }

    Napi::Object result = Napi::Object::New(Env());
    result.Set("text", Napi::String::New(Env(), result_.text));
    result.Set("promptTokens", Napi::Number::New(Env(), result_.prompt_tokens));
    result.Set("completionTokens", Napi::Number::New(Env(), result_.completion_tokens));
    result.Set("cacheReadTokens", Napi::Number::New(Env(), result_.cache_read_tokens));
    result.Set("cacheWriteTokens", Napi::Number::New(Env(), result_.cache_write_tokens));
    result.Set("finishReason", Napi::String::New(Env(), result_.finish_reason));
    if (!result_.error_message.empty()) {
      result.Set("errorMessage", Napi::String::New(Env(), result_.error_message));
    }

    Callback().Call({Env().Null(), result});
  }

  void OnError(const Napi::Error &e) override {
    Napi::HandleScope scope(Env());

    {
      std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
      RemoveWorker(g_generate_workers, handle_, this);
    }

    Callback().Call({Napi::String::New(Env(), e.Message()), Env().Null()});
  }

private:
  int handle_;
  std::vector<llama_wrapper::ChatMessage> messages_;
  llama_wrapper::GenerationParams params_;
  llama_wrapper::GenerationResult result_;
  std::shared_ptr<llama_wrapper::CancellationToken> cancellation_;
};

class StreamGenerateWorker : public Napi::AsyncWorker {
public:
  StreamGenerateWorker(Napi::Function &callback, int handle,
                       const std::vector<llama_wrapper::ChatMessage> &messages,
                       const llama_wrapper::GenerationParams &params, Napi::ThreadSafeFunction tsfn)
      : Napi::AsyncWorker(callback), handle_(handle), messages_(messages), params_(params),
        tsfn_(tsfn), cancellation_(std::make_shared<llama_wrapper::CancellationToken>()) {}

  void Cancel() { cancellation_->cancelled.store(true); }

  void Execute() override {
    std::shared_ptr<llama_wrapper::LlamaModel> model;

    {
      std::lock_guard<std::mutex> lock(g_models_mutex);
      auto it = g_models.find(handle_);
      if (it == g_models.end()) {
        SetError("Invalid model handle");
        return;
      }
      model = it->second;
    }

    // Stream tokens during generation using thread-safe function
    result_ = model->generate_streaming(
        messages_, params_,
        [this](const std::string &token) {
          if (cancellation_->cancelled.load()) {
            return false;
          }

          // Create a copy on the heap that will be deleted after the callback
          std::string *tokenCopy = new std::string(token);
          // Call JavaScript callback from worker thread via thread-safe function
          napi_status status = tsfn_.BlockingCall(
              tokenCopy, [](Napi::Env env, Napi::Function jsCallback, std::string *data) {
                if (data != nullptr) {
                  jsCallback.Call({Napi::String::New(env, *data)});
                  delete data;
                }
              });
          return status == napi_ok && !cancellation_->cancelled.load();
        },
        *cancellation_);
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());

    {
      std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
      RemoveWorker(g_stream_generate_workers, handle_, this);
    }

    // Release the thread-safe function
    tsfn_.Release();

    // Final callback with result
    Napi::Object result = Napi::Object::New(Env());
    result.Set("text", Napi::String::New(Env(), result_.text));
    result.Set("promptTokens", Napi::Number::New(Env(), result_.prompt_tokens));
    result.Set("completionTokens", Napi::Number::New(Env(), result_.completion_tokens));
    result.Set("cacheReadTokens", Napi::Number::New(Env(), result_.cache_read_tokens));
    result.Set("cacheWriteTokens", Napi::Number::New(Env(), result_.cache_write_tokens));
    result.Set("finishReason", Napi::String::New(Env(), result_.finish_reason));
    if (!result_.error_message.empty()) {
      result.Set("errorMessage", Napi::String::New(Env(), result_.error_message));
    }

    Callback().Call({Env().Null(), result});
  }

  void OnError(const Napi::Error &e) override {
    Napi::HandleScope scope(Env());

    {
      std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
      RemoveWorker(g_stream_generate_workers, handle_, this);
    }

    // Release the thread-safe function
    tsfn_.Release();

    // Call the callback with error
    Callback().Call({Napi::String::New(Env(), e.Message()), Env().Null()});
  }

private:
  int handle_;
  std::vector<llama_wrapper::ChatMessage> messages_;
  llama_wrapper::GenerationParams params_;
  llama_wrapper::GenerationResult result_;
  Napi::ThreadSafeFunction tsfn_;
  std::shared_ptr<llama_wrapper::CancellationToken> cancellation_;
};

class EmbedWorker : public Napi::AsyncWorker {
public:
  EmbedWorker(Napi::Function &callback, int handle, const std::vector<std::string> &texts)
      : Napi::AsyncWorker(callback), handle_(handle), texts_(texts) {}

  void Execute() override {
    std::shared_ptr<llama_wrapper::LlamaModel> model;

    {
      std::lock_guard<std::mutex> lock(g_models_mutex);
      auto it = g_models.find(handle_);
      if (it == g_models.end()) {
        SetError("Invalid model handle");
        return;
      }
      model = it->second;
    }

    result_ = model->embed(texts_);

    if (result_.embeddings.empty() && !texts_.empty()) {
      SetError("Failed to generate embeddings");
      return;
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());

    // Create embeddings array
    Napi::Array embeddings_arr = Napi::Array::New(Env(), result_.embeddings.size());
    for (size_t i = 0; i < result_.embeddings.size(); i++) {
      const auto &emb = result_.embeddings[i];
      Napi::Float32Array embedding = Napi::Float32Array::New(Env(), emb.size());
      for (size_t j = 0; j < emb.size(); j++) {
        embedding[j] = emb[j];
      }
      embeddings_arr.Set(i, embedding);
    }

    Napi::Object result = Napi::Object::New(Env());
    result.Set("embeddings", embeddings_arr);
    result.Set("totalTokens", Napi::Number::New(Env(), result_.total_tokens));

    Callback().Call({Env().Null(), result});
  }

private:
  int handle_;
  std::vector<std::string> texts_;
  llama_wrapper::EmbeddingResult result_;
};

// ============================================================================
// N-API Functions
// ============================================================================

Napi::Value LoadModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "Expected (options, callback)").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Object options = info[0].As<Napi::Object>();
  Napi::Function callback = info[1].As<Napi::Function>();

  std::string model_path = options.Get("modelPath").As<Napi::String>().Utf8Value();
  std::string mmproj_path =
      options.Has("mmprojPath") ? options.Get("mmprojPath").As<Napi::String>().Utf8Value() : "";
  int n_gpu_layers =
      options.Has("gpuLayers") ? options.Get("gpuLayers").As<Napi::Number>().Int32Value() : 99;
  int n_ctx = options.Has("contextSize")
                  ? options.Get("contextSize").As<Napi::Number>().Int32Value()
                  : 2048;
  int n_threads =
      options.Has("threads") ? options.Get("threads").As<Napi::Number>().Int32Value() : 4;
  bool debug = options.Has("debug") ? options.Get("debug").As<Napi::Boolean>().Value() : false;
  bool log_prompts =
      options.Has("logPrompts") ? options.Get("logPrompts").As<Napi::Boolean>().Value() : false;
  std::string chat_template = options.Has("chatTemplate")
                                  ? options.Get("chatTemplate").As<Napi::String>().Utf8Value()
                                  : "auto";
  bool embedding =
      options.Has("embedding") ? options.Get("embedding").As<Napi::Boolean>().Value() : false;

  auto worker = new LoadModelWorker(callback, model_path, mmproj_path, n_gpu_layers, n_ctx,
                                    n_threads, debug, log_prompts, chat_template, embedding);
  worker->Queue();

  return env.Undefined();
}

Napi::Value UnloadModel(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
    return env.Null();
  }

  int handle = info[0].As<Napi::Number>().Int32Value();

  {
    std::lock_guard<std::mutex> generation_lock(g_generation_workers_mutex);
    auto generate_it = g_generate_workers.find(handle);
    if (generate_it != g_generate_workers.end()) {
      for (GenerateWorker *worker : generate_it->second) {
        worker->Cancel();
      }
    }

    auto stream_it = g_stream_generate_workers.find(handle);
    if (stream_it != g_stream_generate_workers.end()) {
      for (StreamGenerateWorker *worker : stream_it->second) {
        worker->Cancel();
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(handle);
    if (it != g_models.end()) {
      g_models.erase(it);
    }
  }

  return Napi::Boolean::New(env, true);
}

Napi::Value CancelGeneration(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
    return env.Null();
  }

  int handle = info[0].As<Napi::Number>().Int32Value();

  bool cancelled = false;

  {
    std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
    auto generate_it = g_generate_workers.find(handle);
    if (generate_it != g_generate_workers.end()) {
      for (GenerateWorker *worker : generate_it->second) {
        worker->Cancel();
        cancelled = true;
      }
    }

    auto stream_it = g_stream_generate_workers.find(handle);
    if (stream_it != g_stream_generate_workers.end()) {
      for (StreamGenerateWorker *worker : stream_it->second) {
        worker->Cancel();
        cancelled = true;
      }
    }
  }

  return Napi::Boolean::New(env, cancelled);
}

// Helper function to parse messages array from JavaScript
std::vector<llama_wrapper::ChatMessage> ParseMessages(Napi::Array messages_arr) {
  std::vector<llama_wrapper::ChatMessage> messages;
  for (uint32_t i = 0; i < messages_arr.Length(); i++) {
    Napi::Object msg_obj = messages_arr.Get(i).As<Napi::Object>();
    llama_wrapper::ChatMessage msg;
    msg.role = msg_obj.Get("role").As<Napi::String>().Utf8Value();
    msg.content = msg_obj.Get("content").As<Napi::String>().Utf8Value();
    if (msg_obj.Has("images") && msg_obj.Get("images").IsArray()) {
      Napi::Array images_arr = msg_obj.Get("images").As<Napi::Array>();
      for (uint32_t j = 0; j < images_arr.Length(); j++) {
        Napi::Object image_obj = images_arr.Get(j).As<Napi::Object>();
        if (!image_obj.Has("data") || !image_obj.Get("data").IsTypedArray()) {
          continue;
        }

        Napi::Uint8Array data = image_obj.Get("data").As<Napi::Uint8Array>();
        msg.images.emplace_back(data.Data(), data.Data() + data.ByteLength());
      }
    }
    messages.push_back(msg);
  }
  return messages;
}

Napi::Value Generate(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsNumber() || !info[1].IsObject() || !info[2].IsFunction()) {
    Napi::TypeError::New(env, "Expected (handle, options, callback)").ThrowAsJavaScriptException();
    return env.Null();
  }

  int handle = info[0].As<Napi::Number>().Int32Value();
  Napi::Object options = info[1].As<Napi::Object>();
  Napi::Function callback = info[2].As<Napi::Function>();

  // Parse messages array
  if (!options.Has("messages") || !options.Get("messages").IsArray()) {
    Napi::TypeError::New(env, "Expected messages array in options").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::vector<llama_wrapper::ChatMessage> messages =
      ParseMessages(options.Get("messages").As<Napi::Array>());

  llama_wrapper::GenerationParams params;
  params.max_tokens =
      options.Has("maxTokens") ? options.Get("maxTokens").As<Napi::Number>().Int32Value() : 256;
  params.temperature = options.Has("temperature")
                           ? options.Get("temperature").As<Napi::Number>().FloatValue()
                           : 0.7f;
  params.top_p = options.Has("topP") ? options.Get("topP").As<Napi::Number>().FloatValue() : 0.9f;
  params.top_k = options.Has("topK") ? options.Get("topK").As<Napi::Number>().Int32Value() : 40;
  params.seed = options.Has("seed") && options.Get("seed").IsNumber()
                    ? options.Get("seed").As<Napi::Number>().Uint32Value()
                    : 0xFFFFFFFFu;

  if (options.Has("stopSequences") && options.Get("stopSequences").IsArray()) {
    Napi::Array stop_arr = options.Get("stopSequences").As<Napi::Array>();
    for (uint32_t i = 0; i < stop_arr.Length(); i++) {
      params.stop_sequences.push_back(stop_arr.Get(i).As<Napi::String>().Utf8Value());
    }
  }

  if (options.Has("grammar") && options.Get("grammar").IsString()) {
    params.grammar = options.Get("grammar").As<Napi::String>().Utf8Value();
  }
  params.prompt_cache =
      options.Has("promptCache") ? options.Get("promptCache").As<Napi::Boolean>().Value() : false;

  auto worker = new GenerateWorker(callback, handle, messages, params);
  {
    std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
    g_generate_workers[handle].push_back(worker);
  }
  worker->Queue();

  return env.Undefined();
}

Napi::Value GenerateStream(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4 || !info[0].IsNumber() || !info[1].IsObject() || !info[2].IsFunction() ||
      !info[3].IsFunction()) {
    Napi::TypeError::New(env, "Expected (handle, options, tokenCallback, doneCallback)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int handle = info[0].As<Napi::Number>().Int32Value();
  Napi::Object options = info[1].As<Napi::Object>();
  Napi::Function token_callback = info[2].As<Napi::Function>();
  Napi::Function done_callback = info[3].As<Napi::Function>();

  // Parse messages array
  if (!options.Has("messages") || !options.Get("messages").IsArray()) {
    Napi::TypeError::New(env, "Expected messages array in options").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::vector<llama_wrapper::ChatMessage> messages =
      ParseMessages(options.Get("messages").As<Napi::Array>());

  llama_wrapper::GenerationParams params;
  params.max_tokens =
      options.Has("maxTokens") ? options.Get("maxTokens").As<Napi::Number>().Int32Value() : 256;
  params.temperature = options.Has("temperature")
                           ? options.Get("temperature").As<Napi::Number>().FloatValue()
                           : 0.7f;
  params.top_p = options.Has("topP") ? options.Get("topP").As<Napi::Number>().FloatValue() : 0.9f;
  params.top_k = options.Has("topK") ? options.Get("topK").As<Napi::Number>().Int32Value() : 40;
  params.seed = options.Has("seed") && options.Get("seed").IsNumber()
                    ? options.Get("seed").As<Napi::Number>().Uint32Value()
                    : 0xFFFFFFFFu;

  if (options.Has("stopSequences") && options.Get("stopSequences").IsArray()) {
    Napi::Array stop_arr = options.Get("stopSequences").As<Napi::Array>();
    for (uint32_t i = 0; i < stop_arr.Length(); i++) {
      params.stop_sequences.push_back(stop_arr.Get(i).As<Napi::String>().Utf8Value());
    }
  }

  if (options.Has("grammar") && options.Get("grammar").IsString()) {
    params.grammar = options.Get("grammar").As<Napi::String>().Utf8Value();
  }
  params.prompt_cache =
      options.Has("promptCache") ? options.Get("promptCache").As<Napi::Boolean>().Value() : false;

  // Create thread-safe function for streaming tokens to JavaScript
  Napi::ThreadSafeFunction tsfn =
      Napi::ThreadSafeFunction::New(env, token_callback, "TokenCallback",
                                    0, // Unlimited queue size
                                    1  // Initial thread count
      );

  auto worker = new StreamGenerateWorker(done_callback, handle, messages, params, tsfn);
  {
    std::lock_guard<std::mutex> lock(g_generation_workers_mutex);
    g_stream_generate_workers[handle].push_back(worker);
  }
  worker->Queue();

  return env.Undefined();
}

Napi::Value IsModelLoaded(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
    return env.Null();
  }

  int handle = info[0].As<Napi::Number>().Int32Value();

  std::lock_guard<std::mutex> lock(g_models_mutex);
  auto it = g_models.find(handle);
  bool loaded = it != g_models.end() && it->second->is_loaded();

  return Napi::Boolean::New(env, loaded);
}

Napi::Value Embed(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsNumber() || !info[1].IsObject() || !info[2].IsFunction()) {
    Napi::TypeError::New(env, "Expected (handle, options, callback)").ThrowAsJavaScriptException();
    return env.Null();
  }

  int handle = info[0].As<Napi::Number>().Int32Value();
  Napi::Object options = info[1].As<Napi::Object>();
  Napi::Function callback = info[2].As<Napi::Function>();

  // Parse texts array
  if (!options.Has("texts") || !options.Get("texts").IsArray()) {
    Napi::TypeError::New(env, "Expected texts array in options").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array texts_arr = options.Get("texts").As<Napi::Array>();
  std::vector<std::string> texts;
  for (uint32_t i = 0; i < texts_arr.Length(); i++) {
    texts.push_back(texts_arr.Get(i).As<Napi::String>().Utf8Value());
  }

  auto worker = new EmbedWorker(callback, handle, texts);
  worker->Queue();

  return env.Undefined();
}

// ============================================================================
// Module Initialization
// ============================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("loadModel", Napi::Function::New(env, LoadModel));
  exports.Set("unloadModel", Napi::Function::New(env, UnloadModel));
  exports.Set("generate", Napi::Function::New(env, Generate));
  exports.Set("generateStream", Napi::Function::New(env, GenerateStream));
  exports.Set("cancelGeneration", Napi::Function::New(env, CancelGeneration));
  exports.Set("isModelLoaded", Napi::Function::New(env, IsModelLoaded));
  exports.Set("embed", Napi::Function::New(env, Embed));
  return exports;
}

NODE_API_MODULE(llama_binding, Init)
