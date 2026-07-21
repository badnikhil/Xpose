// Self-test: build a tiny GGUF v3 file byte-by-byte, parse it back, and check
// exact expected values.
//
// Why this exists alongside the real-model run: a 700 MB Gemma file proves the
// parser survives a real header and that dequantised weights *look* sane, but
// it cannot prove any individual number is right — there is no ground truth to
// compare against and no reference implementation available offline. Here the
// ground truth is hand-computed, so a one-bit error in a block layout fails
// loudly. It also covers Q4_0 and Q5_0 patterns the real file never exercises
// (Gemma 3 1B Q4_K_M contains no Q4_0 at all).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "gguf.hpp"
#include "tokenizer.hpp"

using namespace vulkore::llm;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "OK" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

void check_near(float got, float want, const std::string& what) {
    const bool ok = std::abs(got - want) < 1e-6f;
    if (!ok)
        std::printf("  FAIL %s (got %g, want %g)\n", what.c_str(), static_cast<double>(got),
                    static_cast<double>(want));
    else
        std::printf("  OK   %s\n", what.c_str());
    if (!ok) ++g_fail;
}

// ---- little-endian writers ------------------------------------------------
struct Buf {
    std::string b;
    void u8(uint8_t v) { b.append(reinterpret_cast<const char*>(&v), 1); }
    void u32(uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); }
    void i32(int32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); }
    void u64(uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); }
    void f32(float v) { b.append(reinterpret_cast<const char*>(&v), 4); }
    void u16(uint16_t v) { b.append(reinterpret_cast<const char*>(&v), 2); }
    void str(std::string_view s) { u64(s.size()); b.append(s); }
    // key + type tag + value
    void kv_str(std::string_view k, std::string_view v) { str(k); u32(8); str(v); }
    void kv_u32(std::string_view k, uint32_t v) { str(k); u32(4); u32(v); }
    void kv_f32(std::string_view k, float v) { str(k); u32(6); f32(v); }
    void kv_bool(std::string_view k, bool v) { str(k); u32(7); u8(v ? 1 : 0); }
    void pad_to(size_t align) {
        while (b.size() % align) u8(0);
    }
};

constexpr uint16_t kHalfHalf = 0x3800;  // fp16 0.5
constexpr std::string_view kSp = "\xe2\x96\x81";  // U+2581

std::string sp(std::string_view rest) { return std::string(kSp) + std::string(rest); }

}  // namespace

int run_selftest() {
    g_fail = 0;
    std::printf("== selftest: synthetic GGUF ==\n");

    // ---- vocabulary ------------------------------------------------------
    // 0 <unk>, 1 <bos>, 2 <eos>, 3..258 byte tokens, then normals designed so
    // that "hello" must merge along an exact known path (see below).
    std::vector<std::string> toks;
    std::vector<float>       scores;
    std::vector<int32_t>     types;
    auto add = [&](const std::string& t, float s, int32_t ty) {
        toks.push_back(t);
        scores.push_back(s);
        types.push_back(ty);
    };
    add("<unk>", 0, 2);
    add("<bos>", 0, 3);
    add("<eos>", 0, 3);
    for (int i = 0; i < 256; ++i) {
        char nm[16];
        std::snprintf(nm, sizeof nm, "<0x%02X>", i);
        add(nm, 0, 6);
    }
    // Merge chain for "hello" (normalised to "_hello"):
    //   _ + h -> _h(-3), _h + e -> _he(-2), l + l -> ll(-4),
    //   ll + o -> llo(-1), _he + llo -> _hello(0)
    // Higher score wins, so the pop order is -1 last-formed etc. If the
    // priority queue or the stale-entry check is wrong this lands elsewhere.
    for (const char* c : {"h", "e", "l", "o"}) add(c, -10, 1);
    add(std::string(kSp), -10, 1);
    add(sp("h"), -3, 1);
    add(sp("he"), -2, 1);
    add("ll", -4, 1);
    add("llo", -1, 1);
    add(sp("hello"), 0, 1);
    const int32_t id_sp_hello = static_cast<int32_t>(toks.size()) - 1;
    const int32_t id_sp_h     = static_cast<int32_t>(toks.size()) - 5;

    // ---- tensor payloads -------------------------------------------------
    Buf blob;
    // f32_t: 8 known floats.
    const uint64_t off_f32 = 0;
    for (int i = 0; i < 8; ++i) blob.f32(static_cast<float>(i) * 1.5f - 3.0f);
    blob.pad_to(32);

    // q4_0_t: one block. d = 0.5; byte j holds element j in the low nibble and
    // element j+16 in the high nibble; both are unsigned 0..15 biased by -8.
    const uint64_t off_q4_0 = blob.b.size();
    blob.u16(kHalfHalf);
    for (int j = 0; j < 16; ++j) blob.u8(static_cast<uint8_t>(j | ((15 - j) << 4)));
    blob.pad_to(32);

    // q5_0_t: one block. d = 0.5; qh = 0xFFFF0000 sets the 5th bit for
    // elements 16..31 only, which is exactly the (j+12)-shift path.
    const uint64_t off_q5_0 = blob.b.size();
    blob.u16(kHalfHalf);
    blob.u32(0xFFFF0000u);
    for (int j = 0; j < 16; ++j) blob.u8(static_cast<uint8_t>(j | ((15 - j) << 4)));
    blob.pad_to(32);

    // ---- header + metadata + tensor descriptors --------------------------
    Buf f;
    f.u32(0x46554747u);  // "GGUF"
    f.u32(3);            // version
    f.u64(3);            // tensor count
    f.u64(16);           // metadata kv count

    f.kv_str("general.architecture", "llama");
    f.kv_str("general.name", "selftest");
    f.kv_u32("llama.block_count", 7);
    f.kv_u32("llama.attention.head_count", 4);
    f.kv_u32("llama.attention.head_count_kv", 2);
    f.kv_u32("llama.embedding_length", 64);
    f.kv_u32("llama.feed_forward_length", 172);
    f.kv_u32("llama.context_length", 512);
    f.kv_f32("llama.rope.freq_base", 1000000.0f);
    f.kv_str("tokenizer.ggml.model", "llama");
    f.kv_bool("tokenizer.ggml.add_space_prefix", true);
    f.kv_u32("tokenizer.ggml.unknown_token_id", 0);
    f.kv_u32("tokenizer.ggml.bos_token_id", 1);
    f.kv_u32("tokenizer.ggml.eos_token_id", 2);
    // tokens: array(9) of string(8)
    f.str("tokenizer.ggml.tokens");
    f.u32(9);
    f.u32(8);
    f.u64(toks.size());
    for (const auto& t : toks) f.str(t);
    // scores: array of f32(6)
    f.str("tokenizer.ggml.scores");
    f.u32(9);
    f.u32(6);
    f.u64(scores.size());
    for (float s : scores) f.f32(s);
    // NOTE: token_type is deliberately OMITTED (kv count says 13) to exercise
    // the "no token_type array" default path — some converters skip it.

    auto tensor_desc = [&](std::string_view name, std::vector<uint64_t> shape, uint32_t type,
                           uint64_t off) {
        f.str(name);
        f.u32(static_cast<uint32_t>(shape.size()));
        for (uint64_t d : shape) f.u64(d);
        f.u32(type);
        f.u64(off);
    };
    tensor_desc("f32_t", {4, 2}, 0, off_f32);
    tensor_desc("q4_0_t", {32}, 2, off_q4_0);
    tensor_desc("q5_0_t", {32}, 6, off_q5_0);

    f.pad_to(32);
    const size_t data_off = f.b.size();
    f.b += blob.b;

    // ---- write, parse, verify -------------------------------------------
    char path[] = "/tmp/vulkore_selftest_XXXXXX.gguf";
    int fd = mkstemps(path, 5);
    if (fd < 0) {
        std::printf("  FAIL could not create temp file\n");
        return 1;
    }
    const ssize_t wrote = ::write(fd, f.b.data(), f.b.size());
    ::close(fd);
    if (wrote != static_cast<ssize_t>(f.b.size())) {
        std::printf("  FAIL short write\n");
        ::unlink(path);
        return 1;
    }

    int rc = 0;
    try {
        GGUFFile g = GGUFFile::open(path);
        check(g.version() == 3, "version == 3");
        check(g.tensors().size() == 3, "3 tensors");
        check(g.tensor_data_offset() == data_off, "data blob offset matches what we wrote");

        const ModelConfig c = g.config();
        check(c.arch == "llama", "arch == llama");
        check(c.n_layers == 7, "n_layers == 7");
        check(c.n_heads == 4 && c.n_kv_heads == 2, "head counts 4 / 2");
        check(c.embedding_length == 64 && c.feed_forward_length == 172, "embed 64 / ffn 172");
        check(c.rope_freq_base == 1000000.0f, "rope_freq_base == 1e6");
        check(c.head_dim() == 16, "head_dim falls back to embed/heads == 16");
        check(c.vocab_size == toks.size(), "vocab_size from tokens array length");

        // ---- F32 ----
        const Tensor* t32 = g.tensor("f32_t");
        check(t32 != nullptr, "lookup f32_t");
        if (t32) {
            check(t32->shape == std::vector<uint64_t>({4, 2}), "f32_t shape [4, 2]");
            check(t32->nbytes == 32, "f32_t nbytes == 32");
            const auto v = g.dequantize_tensor(*t32);
            bool ok = v.size() == 8;
            for (size_t i = 0; ok && i < 8; ++i)
                ok = std::abs(v[i] - (static_cast<float>(i) * 1.5f - 3.0f)) < 1e-6f;
            check(ok, "f32_t values exact");
        }

        // ---- Q4_0: y[j] = (j-8)*0.5, y[j+16] = ((15-j)-8)*0.5 ----
        const Tensor* t4 = g.tensor("q4_0_t");
        check(t4 && t4->nbytes == 18, "q4_0_t is one 18-byte block");
        if (t4) {
            const auto v = g.dequantize_tensor(*t4);
            bool ok = v.size() == 32;
            for (int j = 0; ok && j < 16; ++j) {
                ok = std::abs(v[j] - static_cast<float>(j - 8) * 0.5f) < 1e-6f &&
                     std::abs(v[j + 16] - static_cast<float>((15 - j) - 8) * 0.5f) < 1e-6f;
            }
            check(ok, "q4_0_t dequantises to the hand-computed block");
            check_near(v[0], -4.0f, "q4_0_t[0] == -4.0");
            check_near(v[15], 3.5f, "q4_0_t[15] == +3.5");
            check_near(v[16], 3.5f, "q4_0_t[16] == +3.5 (high nibble of byte 0)");
        }

        // ---- Q5_0: y[j] = (j-16)*0.5, y[j+16] = (15-j)*0.5 ----
        const Tensor* t5 = g.tensor("q5_0_t");
        check(t5 && t5->nbytes == 22, "q5_0_t is one 22-byte block");
        if (t5) {
            const auto v = g.dequantize_tensor(*t5);
            bool ok = v.size() == 32;
            for (int j = 0; ok && j < 16; ++j) {
                ok = std::abs(v[j] - static_cast<float>(j - 16) * 0.5f) < 1e-6f &&
                     std::abs(v[j + 16] - static_cast<float>(15 - j) * 0.5f) < 1e-6f;
            }
            check(ok, "q5_0_t dequantises with the high bit taken from qh");
            check_near(v[0], -8.0f, "q5_0_t[0] == -8.0 (qh bit clear)");
            check_near(v[16], 7.5f, "q5_0_t[16] == +7.5 (qh bit set)");
        }

        // ---- tokenizer ----
        const Tokenizer tk = Tokenizer::from_gguf(g);
        check(tk.vocab_size() == toks.size(), "tokenizer vocab size");
        check(tk.add_space_prefix(), "add_space_prefix read from metadata");
        // token_type was omitted, so everything defaults to NORMAL and byte
        // fallback must be unavailable -> unknown chars fall back to <unk>.
        check(tk.token_type(3) == TokenType::NORMAL,
              "missing token_type array defaults every token to NORMAL");

        const auto ids = tk.encode("hello", false);
        check(ids.size() == 1 && ids[0] == id_sp_hello,
              "\"hello\" merges to the single token _hello along the scored path");
        check(tk.decode(ids, true) == "hello",
              "decode strips the dummy space prefix");

        // 'q' is in neither the vocab nor (since token_type was omitted) the
        // byte-token set, so it must land on <unk>.
        const auto ids2 = tk.encode("hq", false);
        check(ids2.size() == 2 && ids2[0] == id_sp_h && ids2[1] == tk.unk_id(),
              "unknown char with no usable BYTE tokens falls back to <unk>");
    } catch (const std::exception& e) {
        std::printf("  FAIL exception: %s\n", e.what());
        ++g_fail;
    }
    ::unlink(path);

    // ---- malformed inputs must throw, not crash --------------------------
    auto expect_throw = [&](const std::string& bytes, const char* what) {
        char p2[] = "/tmp/vulkore_selftest_bad_XXXXXX.gguf";
        int d = mkstemps(p2, 5);
        if (d < 0) return;
        [[maybe_unused]] ssize_t w = ::write(d, bytes.data(), bytes.size());
        ::close(d);
        bool threw = false;
        try {
            GGUFFile bad = GGUFFile::open(p2);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, what);
        ::unlink(p2);
    };
    expect_throw("NOTGGUF!", "bad magic is rejected");
    expect_throw(f.b.substr(0, 24), "truncated header is rejected");
    expect_throw(f.b.substr(0, f.b.size() / 2), "truncated body is rejected");

    rc = g_fail;
    std::printf("  selftest: %d failing checks\n", rc);
    return rc;
}
