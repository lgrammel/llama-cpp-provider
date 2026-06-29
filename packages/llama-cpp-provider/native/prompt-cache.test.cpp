#include "prompt-cache.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using llama_wrapper::matching_cached_prefix;
using llama_wrapper::prefill_prompt_cache;
using llama_wrapper::PromptCacheOps;
using llama_wrapper::sync_prompt_cache_to_text;
using llama_wrapper::TokenList;

namespace {

struct DecodeCall {
  TokenList tokens;
  size_t offset;
  size_t count;
  int start_pos;
};

struct MockLlama {
  std::map<std::string, TokenList> tokenized_text;
  std::vector<size_t> trim_calls;
  std::vector<DecodeCall> decode_calls;
  int clear_calls = 0;
  bool trim_result = true;
  bool decode_result = true;

  PromptCacheOps ops() {
    return {
        [this](const std::string &text) { return tokenized_text.at(text); },
        [this](size_t keep_tokens) {
          trim_calls.push_back(keep_tokens);
          return trim_result;
        },
        [this]() { clear_calls++; },
        [this](const TokenList &tokens, size_t offset, size_t count, int start_pos) {
          decode_calls.push_back({tokens, offset, count, start_pos});
          return decode_result;
        },
    };
  }
};

std::string token_list_to_string(const TokenList &tokens) {
  std::ostringstream stream;
  stream << "{";
  for (size_t i = 0; i < tokens.size(); i++) {
    if (i > 0) {
      stream << ", ";
    }
    stream << tokens[i];
  }
  stream << "}";
  return stream.str();
}

void expect_true(bool value, const std::string &message) {
  if (!value) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

void expect_eq(size_t actual, size_t expected, const std::string &message) {
  if (actual != expected) {
    std::cerr << message << ": expected " << expected << ", got " << actual << "\n";
    std::exit(1);
  }
}

void expect_eq(int actual, int expected, const std::string &message) {
  if (actual != expected) {
    std::cerr << message << ": expected " << expected << ", got " << actual << "\n";
    std::exit(1);
  }
}

void expect_tokens_eq(const TokenList &actual, const TokenList &expected,
                      const std::string &message) {
  if (actual != expected) {
    std::cerr << message << ": expected " << token_list_to_string(expected) << ", got "
              << token_list_to_string(actual) << "\n";
    std::exit(1);
  }
}

void test_matching_cached_prefix() {
  expect_eq(matching_cached_prefix({1, 2, 3}, {1, 2, 4}), static_cast<size_t>(2),
            "matches shared prefix");
  expect_eq(matching_cached_prefix({1, 2}, {3, 4}), static_cast<size_t>(0),
            "returns zero for mismatched first token");
  expect_eq(matching_cached_prefix({1, 2, 3}, {1, 2, 3, 4}), static_cast<size_t>(3),
            "matches full cached prefix");
}

void test_prefill_reuses_matching_prefix() {
  MockLlama llama;
  TokenList cached_tokens = {1, 2, 3};

  const auto result = prefill_prompt_cache(cached_tokens, {1, 2, 3, 4, 5}, true, llama.ops());

  expect_true(result.ok, "prefill succeeds");
  expect_eq(result.cache_read_tokens, 3, "reports cache read tokens");
  expect_eq(result.cache_write_tokens, 2, "reports cache write tokens");
  expect_eq(result.n_past, 5, "reports prompt token count");
  expect_eq(llama.trim_calls.size(), static_cast<size_t>(1), "trims cached suffix");
  expect_eq(llama.trim_calls[0], static_cast<size_t>(3), "keeps shared prefix");
  expect_eq(llama.clear_calls, 0, "does not clear context");
  expect_eq(llama.decode_calls.size(), static_cast<size_t>(1), "decodes uncached suffix");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(3), "decodes from suffix offset");
  expect_eq(llama.decode_calls[0].count, static_cast<size_t>(2), "decodes suffix token count");
  expect_eq(llama.decode_calls[0].start_pos, 3, "decodes at prefix position");
  expect_tokens_eq(cached_tokens, {1, 2, 3, 4, 5}, "updates cached tokens to prompt");
}

void test_prefill_disabled_cache_clears_context() {
  MockLlama llama;
  TokenList cached_tokens = {1, 2, 3};

  const auto result = prefill_prompt_cache(cached_tokens, {1, 2, 3, 4}, false, llama.ops());

  expect_true(result.ok, "prefill succeeds without cache");
  expect_eq(result.cache_read_tokens, 0, "does not report cache reads");
  expect_eq(result.cache_write_tokens, 4, "writes full prompt");
  expect_eq(llama.clear_calls, 1, "clears context");
  expect_eq(llama.trim_calls.size(), static_cast<size_t>(0), "does not trim");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(0), "decodes full prompt");
  expect_tokens_eq(cached_tokens, {1, 2, 3, 4}, "stores full prompt");
}

void test_prefill_shorter_prompt_reevaluates_last_token() {
  MockLlama llama;
  TokenList cached_tokens = {1, 2, 3, 4};

  const auto result = prefill_prompt_cache(cached_tokens, {1, 2, 3}, true, llama.ops());

  expect_true(result.ok, "prefill succeeds for shorter prompt");
  expect_eq(result.cache_read_tokens, 2, "backs up one token before prompt end");
  expect_eq(result.cache_write_tokens, 1, "reevaluates last prompt token");
  expect_eq(llama.trim_calls[0], static_cast<size_t>(2), "trims to backed-up prefix");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(2), "decodes last prompt token");
  expect_eq(llama.decode_calls[0].start_pos, 2, "decodes at backed-up position");
  expect_tokens_eq(cached_tokens, {1, 2, 3}, "updates cache to shorter prompt");
}

void test_prefill_trim_failure_falls_back_to_full_decode() {
  MockLlama llama;
  llama.trim_result = false;
  TokenList cached_tokens = {1, 2, 3};

  const auto result = prefill_prompt_cache(cached_tokens, {1, 2, 3, 4}, true, llama.ops());

  expect_true(result.ok, "prefill succeeds after trim failure");
  expect_eq(result.cache_read_tokens, 0, "does not report failed prefix as read");
  expect_eq(result.cache_write_tokens, 4, "decodes full prompt");
  expect_eq(llama.clear_calls, 1, "clears context after trim failure");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(0), "decodes from start");
  expect_tokens_eq(cached_tokens, {1, 2, 3, 4}, "updates cache to prompt");
}

void test_prefill_decode_failure_clears_cache() {
  MockLlama llama;
  llama.decode_result = false;
  TokenList cached_tokens = {1, 2, 3};

  const auto result = prefill_prompt_cache(cached_tokens, {1, 2, 3, 4}, true, llama.ops());

  expect_true(!result.ok, "prefill fails when decode fails");
  expect_tokens_eq(cached_tokens, {}, "clears cached tokens after decode failure");
}

void test_sync_extends_generated_text() {
  MockLlama llama;
  llama.tokenized_text["prompt assistant"] = {1, 2, 3, 4, 5};
  TokenList cached_tokens = {1, 2, 3};

  const bool ok = sync_prompt_cache_to_text(cached_tokens, "prompt assistant", llama.ops());

  expect_true(ok, "sync succeeds");
  expect_eq(llama.clear_calls, 0, "does not clear when extending prefix");
  expect_eq(llama.trim_calls.size(), static_cast<size_t>(0), "does not trim exact prefix");
  expect_eq(llama.decode_calls.size(), static_cast<size_t>(1), "decodes generated suffix");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(3), "decodes from cached length");
  expect_eq(llama.decode_calls[0].count, static_cast<size_t>(2), "decodes visible suffix");
  expect_tokens_eq(cached_tokens, {1, 2, 3, 4, 5}, "syncs cache to visible text tokens");
}

void test_sync_retokenized_text_trims_and_decodes_difference() {
  MockLlama llama;
  llama.tokenized_text["prompt assistant"] = {1, 2, 9, 10};
  TokenList cached_tokens = {1, 2, 3, 4};

  const bool ok = sync_prompt_cache_to_text(cached_tokens, "prompt assistant", llama.ops());

  expect_true(ok, "sync succeeds after retokenization changed suffix");
  expect_eq(llama.trim_calls.size(), static_cast<size_t>(1), "trims divergent cached suffix");
  expect_eq(llama.trim_calls[0], static_cast<size_t>(2), "keeps shared prefix");
  expect_eq(llama.clear_calls, 0, "does not clear shared-prefix context");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(2), "decodes divergent suffix");
  expect_eq(llama.decode_calls[0].count, static_cast<size_t>(2), "decodes replacement suffix");
  expect_tokens_eq(cached_tokens, {1, 2, 9, 10}, "syncs cache to retokenized visible text");
}

void test_sync_trim_failure_clears_and_decodes_all_tokens() {
  MockLlama llama;
  llama.trim_result = false;
  llama.tokenized_text["prompt assistant"] = {1, 2, 9, 10};
  TokenList cached_tokens = {1, 2, 3, 4};

  const bool ok = sync_prompt_cache_to_text(cached_tokens, "prompt assistant", llama.ops());

  expect_true(ok, "sync succeeds after trim failure");
  expect_eq(llama.clear_calls, 1, "clears context");
  expect_eq(llama.decode_calls[0].offset, static_cast<size_t>(0), "decodes from start");
  expect_eq(llama.decode_calls[0].count, static_cast<size_t>(4), "decodes all visible tokens");
  expect_tokens_eq(cached_tokens, {1, 2, 9, 10}, "syncs cache to visible text");
}

void test_sync_decode_failure_clears_cache() {
  MockLlama llama;
  llama.decode_result = false;
  llama.tokenized_text["prompt assistant"] = {1, 2, 3, 4};
  TokenList cached_tokens = {1, 2};

  const bool ok = sync_prompt_cache_to_text(cached_tokens, "prompt assistant", llama.ops());

  expect_true(!ok, "sync fails when decode fails");
  expect_eq(llama.clear_calls, 1, "clears context after decode failure");
  expect_tokens_eq(cached_tokens, {}, "clears cached tokens");
}

void run_test(const std::string &name, const std::function<void()> &test) {
  test();
  std::cout << "ok - " << name << "\n";
}

} // namespace

int main() {
  run_test("matching cached prefix", test_matching_cached_prefix);
  run_test("prefill reuses matching prefix", test_prefill_reuses_matching_prefix);
  run_test("prefill disabled cache clears context", test_prefill_disabled_cache_clears_context);
  run_test("prefill shorter prompt reevaluates last token",
           test_prefill_shorter_prompt_reevaluates_last_token);
  run_test("prefill trim failure falls back to full decode",
           test_prefill_trim_failure_falls_back_to_full_decode);
  run_test("prefill decode failure clears cache", test_prefill_decode_failure_clears_cache);
  run_test("sync extends generated text", test_sync_extends_generated_text);
  run_test("sync retokenized text trims and decodes difference",
           test_sync_retokenized_text_trims_and_decodes_difference);
  run_test("sync trim failure clears and decodes all tokens",
           test_sync_trim_failure_clears_and_decodes_all_tokens);
  run_test("sync decode failure clears cache", test_sync_decode_failure_clears_cache);
  return 0;
}
