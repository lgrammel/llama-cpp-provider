#include "prompt-cache.h"

#include <algorithm>
#include <utility>

namespace llama_wrapper {

size_t matching_cached_prefix(const TokenList &cached_tokens, const TokenList &tokens) {
  const size_t max_prefix = std::min(cached_tokens.size(), tokens.size());
  size_t prefix = 0;
  while (prefix < max_prefix && cached_tokens[prefix] == tokens[prefix]) {
    prefix++;
  }
  return prefix;
}

PromptCachePrefillResult prefill_prompt_cache(TokenList &cached_tokens,
                                              const TokenList &prompt_tokens, bool prompt_cache,
                                              const PromptCacheOps &ops) {
  PromptCachePrefillResult result;

  size_t cached_prefix = prompt_cache ? matching_cached_prefix(cached_tokens, prompt_tokens) : 0;
  if (cached_prefix == prompt_tokens.size() && cached_prefix < cached_tokens.size() &&
      cached_prefix > 0) {
    // If the next prompt is a shorter prefix of cached state, re-evaluate the
    // last kept token so llama.cpp logits correspond to the new prompt end.
    cached_prefix--;
  }

  if (prompt_cache && cached_prefix > 0 && !ops.trim_tokens(cached_prefix)) {
    cached_prefix = 0;
  }

  if (!prompt_cache || cached_prefix == 0) {
    ops.clear_context();
    cached_tokens.clear();
    cached_prefix = 0;
  }

  const size_t suffix_tokens = prompt_tokens.size() - cached_prefix;
  if (suffix_tokens > 0 && !ops.decode_tokens(prompt_tokens, cached_prefix, suffix_tokens,
                                              static_cast<int>(cached_prefix))) {
    cached_tokens.clear();
    return result;
  }

  cached_tokens = prompt_tokens;
  result.ok = true;
  result.cache_read_tokens = static_cast<int>(cached_prefix);
  result.cache_write_tokens = static_cast<int>(suffix_tokens);
  result.n_past = static_cast<int>(prompt_tokens.size());
  return result;
}

bool sync_prompt_cache_to_text(TokenList &cached_tokens, const std::string &text,
                               const PromptCacheOps &ops) {
  TokenList text_tokens = ops.tokenize_text(text);
  size_t shared_prefix = matching_cached_prefix(cached_tokens, text_tokens);
  if (shared_prefix < cached_tokens.size()) {
    if (!ops.trim_tokens(shared_prefix)) {
      shared_prefix = 0;
    }
  }

  if (shared_prefix < text_tokens.size()) {
    if (shared_prefix == 0) {
      ops.clear_context();
      cached_tokens.clear();
    }

    if (!ops.decode_tokens(text_tokens, shared_prefix, text_tokens.size() - shared_prefix,
                           static_cast<int>(shared_prefix))) {
      ops.clear_context();
      cached_tokens.clear();
      return false;
    }
  }

  cached_tokens = std::move(text_tokens);
  return true;
}

} // namespace llama_wrapper
