#include "tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <queue>
#include <stdexcept>

namespace vulkore::llm {
namespace {

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's stand-in for a space.
constexpr std::string_view kSpaceMarker = "\xe2\x96\x81";

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("tokenizer: " + m); }

// Length in bytes of the UTF-8 sequence starting with `c`. Returns 1 for
// invalid lead bytes so a malformed input still advances (and then falls out
// through byte fallback) instead of looping forever.
size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// One entry of the working linked list: a slice of the normalised string.
struct Symbol {
    int         prev = -1;
    int         next = -1;
    const char* text = nullptr;
    size_t      n    = 0;
};

struct Bigram {
    int    left  = -1;
    int    right = -1;
    float  score = 0;
    size_t size  = 0;
};

// std::priority_queue is a max-heap, so "less" must mean "merge later".
// Ties break toward the LEFTMOST pair, matching sentencepiece and llama.cpp:
// without this the result depends on heap internals and is not reproducible.
struct BigramLess {
    bool operator()(const Bigram& a, const Bigram& b) const {
        return a.score < b.score || (a.score == b.score && a.left > b.left);
    }
};

}  // namespace

const char* token_type_name(TokenType t) {
    switch (t) {
        case TokenType::UNDEFINED:    return "UNDEFINED";
        case TokenType::NORMAL:       return "NORMAL";
        case TokenType::UNKNOWN:      return "UNKNOWN";
        case TokenType::CONTROL:      return "CONTROL";
        case TokenType::USER_DEFINED: return "USER_DEFINED";
        case TokenType::UNUSED:       return "UNUSED";
        case TokenType::BYTE:         return "BYTE";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
Tokenizer Tokenizer::from_gguf(const GGUFFile& f) {
    Tokenizer t;

    const MetaValue* toks = f.meta("tokenizer.ggml.tokens");
    if (!toks || toks->type != MetaType::ARRAY || toks->elem_type != MetaType::STRING)
        fail("tokenizer.ggml.tokens missing or not a string array");
    t.tokens_ = toks->arr_str;

    const size_t n = t.tokens_.size();

    // Scores are absent in a few converters (BPE-style vocabs); default to 0,
    // which makes every merge equally good and leaves ties to the left-most
    // rule. Gemma files do carry them.
    t.scores_.assign(n, 0.0f);
    if (const MetaValue* sc = f.meta("tokenizer.ggml.scores")) {
        if (sc->arr_num.size() != n)
            fail("scores array length " + std::to_string(sc->arr_num.size()) +
                 " != tokens length " + std::to_string(n));
        for (size_t i = 0; i < n; ++i) t.scores_[i] = static_cast<float>(sc->arr_num[i]);
    }

    t.types_.assign(n, TokenType::NORMAL);
    if (const MetaValue* ty = f.meta("tokenizer.ggml.token_type")) {
        if (ty->arr_num.size() != n)
            fail("token_type array length " + std::to_string(ty->arr_num.size()) +
                 " != tokens length " + std::to_string(n));
        for (size_t i = 0; i < n; ++i)
            t.types_[i] = static_cast<TokenType>(static_cast<int32_t>(ty->arr_num[i]));
    }

    if (auto m = f.get_str("tokenizer.ggml.model")) t.model_ = *m;

    auto id_key = [&](const char* k, int32_t dflt) -> int32_t {
        auto v = f.get_u64(k);
        return v ? static_cast<int32_t>(*v) : dflt;
    };
    t.bos_ = id_key("tokenizer.ggml.bos_token_id", -1);
    t.eos_ = id_key("tokenizer.ggml.eos_token_id", -1);
    t.unk_ = id_key("tokenizer.ggml.unknown_token_id", -1);
    t.pad_ = id_key("tokenizer.ggml.padding_token_id", -1);

    t.add_bos_ = f.get_bool("tokenizer.ggml.add_bos_token").value_or(true);
    t.add_eos_ = f.get_bool("tokenizer.ggml.add_eos_token").value_or(false);
    // SentencePiece's add_dummy_prefix. Default true is the SPM default; Gemma
    // sets it explicitly, so we read rather than assume.
    t.add_space_prefix_ = f.get_bool("tokenizer.ggml.add_space_prefix").value_or(true);

    t.build_indices();
    return t;
}

void Tokenizer::build_indices() {
    token_to_id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        // First writer wins: a couple of vocabs contain duplicate spellings
        // (usually an UNUSED slot shadowing a real token) and the lower id is
        // the one the reference implementations emit.
        token_to_id_.emplace(tokens_[i], static_cast<int32_t>(i));

        const TokenType ty = types_[i];
        if (ty == TokenType::CONTROL || ty == TokenType::UNKNOWN)
            special_ids_.push_back(static_cast<int32_t>(i));

        // Byte tokens are spelled "<0xAB>". Index them for fallback.
        const std::string_view s = tokens_[i];
        if (ty == TokenType::BYTE && s.size() == 6 && s.compare(0, 3, "<0x") == 0 &&
            s[5] == '>') {
            unsigned v = 0;
            if (std::sscanf(std::string(s).c_str(), "<0x%02X>", &v) == 1 && v < 256)
                byte_to_id_[v] = static_cast<int32_t>(i);
        }
    }

    // Longest first so a greedy scan prefers <start_of_turn> over any prefix.
    std::sort(special_ids_.begin(), special_ids_.end(), [&](int32_t a, int32_t b) {
        return tokens_[a].size() > tokens_[b].size();
    });

    // Gemma has 6251 CONTROL tokens; comparing all of them at every candidate
    // position would make encode() O(n * 6251). Instead collect the DISTINCT
    // lengths (about 20) and probe the hash map once per length — same result,
    // two orders of magnitude fewer comparisons.
    for (int32_t id : special_ids_) {
        const size_t len = tokens_[static_cast<size_t>(id)].size();
        if (len && (special_lengths_.empty() || special_lengths_.back() != len))
            special_lengths_.push_back(len);
    }
    special_lengths_.erase(std::unique(special_lengths_.begin(), special_lengths_.end()),
                           special_lengths_.end());
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
std::string_view Tokenizer::token_text(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return {};
    return tokens_[static_cast<size_t>(id)];
}
TokenType Tokenizer::token_type(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= types_.size()) return TokenType::UNDEFINED;
    return types_[static_cast<size_t>(id)];
}
float Tokenizer::token_score(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= scores_.size()) return 0.0f;
    return scores_[static_cast<size_t>(id)];
}
int32_t Tokenizer::id_for(std::string_view s) const {
    auto it = token_to_id_.find(s);
    return it == token_to_id_.end() ? -1 : it->second;
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
// One fragment = a run of ordinary text with no special tokens in it. This is
// the actual SentencePiece merge loop.
void Tokenizer::tokenize_fragment(std::string_view raw, std::vector<int32_t>& out) const {
    if (raw.empty()) return;

    // Normalise: every space becomes U+2581, and (per add_dummy_prefix) the
    // fragment gains a leading marker so "hello" and " hello" tokenize alike.
    std::string norm;
    norm.reserve(raw.size() + 8);
    if (add_space_prefix_) norm += kSpaceMarker;
    for (char c : raw) {
        if (c == ' ')
            norm += kSpaceMarker;
        else
            norm += c;
    }

    // Split into UTF-8 characters; these are the initial symbols.
    std::vector<Symbol> syms;
    syms.reserve(norm.size());
    for (size_t i = 0; i < norm.size();) {
        const size_t len = std::min(utf8_len(static_cast<unsigned char>(norm[i])), norm.size() - i);
        Symbol s;
        s.text = norm.data() + i;
        s.n    = len;
        s.prev = static_cast<int>(syms.size()) - 1;
        i += len;
        s.next = (i == norm.size()) ? -1 : static_cast<int>(syms.size()) + 1;
        syms.push_back(s);
    }
    if (syms.empty()) return;

    std::priority_queue<Bigram, std::vector<Bigram>, BigramLess> work;
    // rev_merge[merged text] = the pair it came from, so a merged symbol that
    // somehow is not in the vocab can be split again instead of byte-falling
    // back. In practice merges only fire on in-vocab strings, so this is a
    // belt-and-braces path — kept because llama.cpp has it and dropping it
    // would silently change output on a vocab with duplicate spellings.
    std::unordered_map<std::string, std::pair<int, int>> rev_merge;

    auto try_add = [&](int left, int right) {
        if (left == -1 || right == -1) return;
        const std::string text(syms[left].text, syms[left].n + syms[right].n);
        auto it = token_to_id_.find(text);
        if (it == token_to_id_.end()) return;
        Bigram b;
        b.left  = left;
        b.right = right;
        b.score = scores_[static_cast<size_t>(it->second)];
        b.size  = text.size();
        work.push(b);
        rev_merge[text] = {left, right};
    };

    for (int i = 1; i < static_cast<int>(syms.size()); ++i) try_add(i - 1, i);

    while (!work.empty()) {
        const Bigram b = work.top();
        work.pop();
        Symbol& l = syms[static_cast<size_t>(b.left)];
        Symbol& r = syms[static_cast<size_t>(b.right)];
        // Stale entry: one side was already absorbed by an earlier merge, or
        // its span changed. The queue is never rebuilt, so this check is what
        // keeps it correct.
        if (l.n == 0 || r.n == 0 || l.n + r.n != b.size) continue;

        l.n += r.n;
        r.n = 0;
        l.next = r.next;
        if (r.next >= 0) syms[static_cast<size_t>(r.next)].prev = b.left;

        try_add(l.prev, b.left);
        try_add(b.left, l.next);
    }

    // Emit. Anything still not in the vocab is a single character we have no
    // token for -> byte fallback, one token per UTF-8 byte.
    std::function<void(int)> emit = [&](int idx) {
        const Symbol& s = syms[static_cast<size_t>(idx)];
        const std::string text(s.text, s.n);
        auto it = token_to_id_.find(text);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
            return;
        }
        auto rm = rev_merge.find(text);
        if (rm != rev_merge.end()) {
            emit(rm->second.first);
            emit(rm->second.second);
            return;
        }
        for (size_t j = 0; j < s.n; ++j) {
            const int32_t id = byte_to_id_[static_cast<unsigned char>(s.text[j])];
            if (id >= 0)
                out.push_back(id);
            else if (unk_ >= 0)
                out.push_back(unk_);
            // else: no byte tokens and no unk — drop, nothing sensible to emit.
        }
    };

    for (int i = 0; i != -1; i = syms[static_cast<size_t>(i)].next) {
        if (syms[static_cast<size_t>(i)].n > 0) emit(i);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos, bool add_eos) const {
    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);

    // Pull CONTROL/UNKNOWN tokens out of the stream first: "<start_of_turn>"
    // must become one token, and the merge loop cannot produce it because the
    // dummy space prefix would be glued to the front. We deliberately do NOT
    // partition on USER_DEFINED here — in a Gemma vocab that class holds the
    // whitespace runs ("\n\n", multi-space), which the ordinary merge loop
    // already assembles correctly. See agent-docs/gguf-loader.md.
    size_t i = 0, frag_start = 0;
    while (i < text.size()) {
        int32_t hit = -1;
        // special_lengths_ is descending, so the first match is the longest —
        // <start_of_turn> wins over any prefix of itself.
        for (size_t len : special_lengths_) {
            if (len > text.size() - i) continue;
            auto it = token_to_id_.find(text.substr(i, len));
            if (it == token_to_id_.end()) continue;
            const TokenType ty = types_[static_cast<size_t>(it->second)];
            if (ty == TokenType::CONTROL || ty == TokenType::UNKNOWN) {
                hit = it->second;
                break;
            }
        }
        if (hit >= 0) {
            tokenize_fragment(text.substr(frag_start, i - frag_start), out);
            out.push_back(hit);
            i += tokens_[static_cast<size_t>(hit)].size();
            frag_start = i;
        } else {
            ++i;
        }
    }
    tokenize_fragment(text.substr(frag_start), out);

    if (add_eos && eos_ >= 0) out.push_back(eos_);
    return out;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
std::string Tokenizer::decode_one(int32_t id) const {
    const std::string_view s = token_text(id);
    if (token_type(id) == TokenType::BYTE) {
        unsigned v = 0;
        if (s.size() == 6 && std::sscanf(std::string(s).c_str(), "<0x%02X>", &v) == 1)
            return std::string(1, static_cast<char>(v));
        return std::string(s);
    }
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, kSpaceMarker.size(), kSpaceMarker) == 0) {
            out += ' ';
            i += kSpaceMarker.size();
        } else {
            out += s[i];
            ++i;
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string out;
    bool first_text_token = true;
    for (int32_t id : ids) {
        const TokenType ty = token_type(id);
        if (skip_special && (ty == TokenType::CONTROL || ty == TokenType::UNKNOWN)) continue;

        std::string piece = decode_one(id);
        // Undo add_dummy_prefix: the encoder glued a marker onto the front of
        // the first fragment, so the first piece decodes with a leading space
        // that was never in the input. Strip exactly one.
        if (first_text_token && add_space_prefix_ && !piece.empty() && piece[0] == ' ')
            piece.erase(0, 1);
        if (ty != TokenType::CONTROL && ty != TokenType::UNKNOWN) first_text_token = false;
        out += piece;
    }
    return out;
}

}  // namespace vulkore::llm
