#ifndef PROMPT_CACHE_H
#define PROMPT_CACHE_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llama_wrapper {

using TokenList = std::vector<int32_t>;

struct PromptCacheOps {
  std::function<TokenList(const std::string &text)> tokenize_text;
  std::function<bool(size_t keep_tokens)> trim_tokens;
  std::function<void()> clear_context;
  std::function<bool(const TokenList &tokens, size_t offset, size_t count, int start_pos)>
      decode_tokens;
};

struct PromptCachePrefillResult {
  bool ok = false;
  int cache_read_tokens = 0;
  int cache_write_tokens = 0;
  int n_past = 0;
};

size_t matching_cached_prefix(const TokenList &cached_tokens, const TokenList &tokens);

PromptCachePrefillResult prefill_prompt_cache(TokenList &cached_tokens,
                                              const TokenList &prompt_tokens, bool prompt_cache,
                                              const PromptCacheOps &ops);

bool sync_prompt_cache_to_text(TokenList &cached_tokens, const std::string &text,
                               const PromptCacheOps &ops);

} // namespace llama_wrapper

#endif // PROMPT_CACHE_H
