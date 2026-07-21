// SentencePiece (SPM/BPE) tokenizer driven entirely from GGUF metadata.
//
// A GGUF file carries the whole tokenizer, so there is no separate
// tokenizer.model / tokenizer.json to ship:
//   tokenizer.ggml.model       str    "llama" == SentencePiece
//   tokenizer.ggml.tokens      [str]  the vocabulary, indexed by token id
//   tokenizer.ggml.scores      [f32]  merge priority, one per token
//   tokenizer.ggml.token_type  [i32]  1 NORMAL 2 UNKNOWN 3 CONTROL
//                                     4 USER_DEFINED 5 UNUSED 6 BYTE
//   tokenizer.ggml.{bos,eos,unknown,padding}_token_id
//   tokenizer.ggml.add_{bos,eos}_token, tokenizer.ggml.add_space_prefix
//
// Two SentencePiece conventions matter and both are handled here:
//   - the space marker U+2581 "LOWER ONE EIGHTH BLOCK" (shown as an underscore)
//     stands in for ' ' inside token text;
//   - byte fallback: any character with no vocab entry is emitted as one
//     "<0xAB>" token per UTF-8 byte, so the tokenizer is total — it never
//     drops input, and decode(encode(s)) == s for arbitrary bytes.
//
// The vocabulary is borrowed from the GGUFFile's mapping as string_views, so a
// Tokenizer MUST NOT outlive the GGUFFile it was built from.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf.hpp"

namespace vulkore::llm {

enum class TokenType : int32_t {
    UNDEFINED    = 0,
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

const char* token_type_name(TokenType t);

class Tokenizer {
public:
    // Throws std::runtime_error if the file carries no vocabulary, or if the
    // token/score/type arrays disagree on length.
    static Tokenizer from_gguf(const GGUFFile& f);

    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }
    int32_t unk_id() const { return unk_; }
    int32_t pad_id() const { return pad_; }
    bool    add_bos_default() const { return add_bos_; }
    bool    add_space_prefix() const { return add_space_prefix_; }
    std::string_view model() const { return model_; }

    size_t vocab_size() const { return tokens_.size(); }
    std::string_view token_text(int32_t id) const;  // raw, with U+2581 intact
    TokenType        token_type(int32_t id) const;
    float            token_score(int32_t id) const;
    // -1 when absent. Exact match on the raw vocab spelling.
    int32_t          id_for(std::string_view raw_token_text) const;

    std::vector<int32_t> encode(std::string_view text, bool add_bos, bool add_eos = false) const;

    // skip_special drops CONTROL/UNKNOWN tokens (BOS, <start_of_turn>, ...)
    // from the output, which is what you want when showing text to a user.
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;
    // Single-token piece, byte tokens resolved, U+2581 turned back into ' '.
    std::string decode_one(int32_t id) const;

private:
    void build_indices();
    void tokenize_fragment(std::string_view text, std::vector<int32_t>& out) const;

    std::vector<std::string_view> tokens_;
    std::vector<float>            scores_;
    std::vector<TokenType>        types_;
    std::unordered_map<std::string_view, int32_t> token_to_id_;
    // Sorted longest-first so a greedy scan matches <start_of_turn> before <s>.
    std::vector<int32_t> special_ids_;
    // Distinct lengths of the above, descending: encode() probes one hash
    // lookup per length instead of comparing against every special token.
    std::vector<size_t> special_lengths_;
    std::vector<int32_t> byte_to_id_ = std::vector<int32_t>(256, -1);

    std::string_view model_ = "llama";
    int32_t bos_ = -1, eos_ = -1, unk_ = -1, pad_ = -1;
    bool    add_bos_ = true, add_eos_ = false, add_space_prefix_ = true;
};

}  // namespace vulkore::llm
