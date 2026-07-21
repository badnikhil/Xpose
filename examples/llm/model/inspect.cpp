// inspect — dump a GGUF file's config, metadata, tensors, and prove the
// tokenizer round-trips. Pure host code; this binary never touches Vulkan.
//
//   ./build/examples/llm/model/gguf_inspect model.gguf [--tensors] [--meta]
//         [--encode "text"] [--dequant NAME] [--compare other.gguf] [--verbose]
//   ./build/examples/llm/model/gguf_inspect --selftest   (no model file needed)
//
// Default output is the summary that matters: config, per-type tensor tally,
// and the tokenizer round-trip table. Everything else is opt-in because a
// Gemma 3 file has 262144 vocab entries and 340 tensors.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "tokenizer.hpp"

using namespace vulkore::llm;

namespace {

std::string shape_str(const Tensor& t) {
    std::string s = "[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(t.shape[i]);
    }
    return s + "]";
}

std::string human(uint64_t bytes) {
    static const char* u[] = {"B", "KiB", "MiB", "GiB"};
    double v = static_cast<double>(bytes);
    int i = 0;
    while (v >= 1024.0 && i < 3) { v /= 1024.0; ++i; }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f %s", v, u[i]);
    return buf;
}

// Print a string with control characters and the U+2581 space marker made
// visible — otherwise a tokenizer dump is unreadable.
std::string visible(std::string_view s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, 3, "\xe2\x96\x81") == 0) { out += "_"; i += 3; continue; }
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\n')      out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20)  { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); out += b; }
        else                out += static_cast<char>(c);
        ++i;
    }
    return out;
}

void print_meta_value(const MetaValue& v) {
    switch (v.type) {
        case MetaType::STRING: {
            std::string s = visible(v.str);
            if (s.size() > 120) s = s.substr(0, 117) + "...";
            std::printf("\"%s\"", s.c_str());
            break;
        }
        case MetaType::ARRAY: {
            const size_t n = v.array_length();
            std::printf("[%s x %zu]", meta_type_name(v.elem_type), n);
            if (n) {
                std::printf(" first: ");
                for (size_t i = 0; i < std::min<size_t>(n, 4); ++i) {
                    if (i) std::printf(", ");
                    if (v.elem_type == MetaType::STRING)
                        std::printf("\"%s\"", visible(v.arr_str[i]).c_str());
                    else
                        std::printf("%g", v.arr_num[i]);
                }
                if (n > 4) std::printf(", ...");
            }
            break;
        }
        case MetaType::FLOAT32:
        case MetaType::FLOAT64: std::printf("%g", v.f); break;
        case MetaType::BOOL:    std::printf("%s", v.u ? "true" : "false"); break;
        case MetaType::INT8:
        case MetaType::INT16:
        case MetaType::INT32:
        case MetaType::INT64:   std::printf("%" PRId64, v.i); break;
        default:                std::printf("%" PRIu64, v.u); break;
    }
}

// Dequantise a tensor and report stats. This is the only real check that the
// block layouts are right: wrong strides give NaNs, absurd magnitudes, or a
// mean that drifts far off zero.
void dequant_stats(const GGUFFile& f, const Tensor& t, size_t max_elems) {
    const TypeTraits& tt = type_traits(t.type);
    std::printf("  %-34s %-6s %-18s ", std::string(t.name).c_str(), tt.name,
                shape_str(t).c_str());
    if (!tt.dequantisable) { std::printf("(dequant unimplemented)\n"); return; }

    // Whole blocks only, capped so we do not materialise a 1.2 GB embedding.
    size_t n = static_cast<size_t>(t.n_elements());
    n = std::min(n, max_elems);
    n -= n % tt.block_size;
    if (n == 0) { std::printf("(too small)\n"); return; }

    std::vector<float> out(n);
    if (!GGUFFile::dequantize(t.type, t.data, out.data(), n)) {
        std::printf("(dequant failed)\n");
        return;
    }
    double sum = 0, sumsq = 0;
    float lo = out[0], hi = out[0];
    size_t bad = 0;
    for (float x : out) {
        if (!std::isfinite(x)) { ++bad; continue; }
        sum += x; sumsq += static_cast<double>(x) * x;
        lo = std::min(lo, x); hi = std::max(hi, x);
    }
    const double mean = sum / static_cast<double>(n);
    const double rms  = std::sqrt(sumsq / static_cast<double>(n));
    std::printf("n=%-8zu mean=%+.5f rms=%.5f min=%+.4f max=%+.4f nonfinite=%zu\n", n, mean,
                rms, lo, hi, bad);
    (void)f;
}

int round_trip(const Tokenizer& tk, std::string_view text, bool verbose,
               bool skip_special = true) {
    const std::vector<int32_t> ids = tk.encode(text, /*add_bos=*/false);
    const std::string back = tk.decode(ids, skip_special);
    const bool ok = (back == text);
    std::printf("  %-4s %3zu tok  \"%s\"\n", ok ? "OK" : "FAIL", ids.size(),
                visible(text).c_str());
    if (verbose || !ok) {
        std::printf("       ids:");
        for (size_t i = 0; i < ids.size() && i < 48; ++i) std::printf(" %d", ids[i]);
        if (ids.size() > 48) std::printf(" ...");
        std::printf("\n       pieces:");
        for (size_t i = 0; i < ids.size() && i < 48; ++i)
            std::printf(" |%s", visible(tk.token_text(ids[i])).c_str());
        std::printf("|\n");
        if (!ok) std::printf("       decoded back as: \"%s\"\n", visible(back).c_str());
    }
    return ok ? 0 : 1;
}

// Cross-check every shared tensor against a second quantisation of the SAME
// model. If both dequantisers are right the two must agree to within
// quantisation noise; if a block layout is wrong the values decorrelate
// immediately. This is the only real ground truth available offline — there is
// no reference implementation on this machine to diff against.
int compare_files(const GGUFFile& a, const GGUFFile& b, size_t cap) {
    std::printf("\n== cross-quantisation comparison ==\n");
    std::printf("  %-34s %-6s %-6s %10s %10s %8s\n", "tensor", "A", "B", "rel_rms", "corr",
                "verdict");

    // Aggregate per (typeA,typeB) pair: the per-tensor lines are 340 rows.
    std::map<std::string, std::pair<size_t, double>> agg;  // -> (count, worst corr)
    int bad = 0;
    for (const Tensor& ta : a.tensors()) {
        const Tensor* tb = b.tensor(ta.name);
        if (!tb || tb->shape != ta.shape) continue;
        if (!type_traits(ta.type).dequantisable || !type_traits(tb->type).dequantisable) continue;
        if (ta.type == tb->type) continue;  // same encoding proves nothing

        size_t n = static_cast<size_t>(ta.n_elements());
        n = std::min(n, cap);
        const uint32_t bs = std::max(type_traits(ta.type).block_size,
                                     type_traits(tb->type).block_size);
        n -= n % bs;
        if (n == 0) continue;

        std::vector<float> va(n), vb(n);
        if (!GGUFFile::dequantize(ta.type, ta.data, va.data(), n)) continue;
        if (!GGUFFile::dequantize(tb->type, tb->data, vb.data(), n)) continue;

        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sdd = 0;
        for (size_t i = 0; i < n; ++i) {
            const double x = va[i], y = vb[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            sdd += (x - y) * (x - y);
        }
        const double na = static_cast<double>(n);
        const double cov = sab / na - (sa / na) * (sb / na);
        const double vara = saa / na - (sa / na) * (sa / na);
        const double varb = sbb / na - (sb / na) * (sb / na);
        const double corr = (vara > 0 && varb > 0) ? cov / std::sqrt(vara * varb) : 0.0;
        const double rel_rms = std::sqrt(sdd / na) / (std::sqrt(sbb / na) + 1e-30);

        // Two honest quantisations of the same weights correlate ~0.99+. A
        // wrong layout lands near 0. The threshold is deliberately slack.
        const bool ok = corr > 0.95;
        const std::string key = std::string(type_traits(ta.type).name) + " vs " +
                                type_traits(tb->type).name;
        auto& e = agg[key];
        if (e.first == 0) e.second = corr;
        e.first++;
        e.second = std::min(e.second, corr);
        if (!ok) {
            ++bad;
            std::printf("  %-34s %-6s %-6s %10.5f %10.5f %8s\n", std::string(ta.name).c_str(),
                        type_traits(ta.type).name, type_traits(tb->type).name, rel_rms, corr,
                        "MISMATCH");
        }
    }
    for (const auto& [k, e] : agg)
        std::printf("  %-34s %4zu tensors, worst corr %.5f  %s\n", k.c_str(), e.first, e.second,
                    e.second > 0.95 ? "OK" : "FAIL");
    if (agg.empty()) std::printf("  (no differently-typed shared tensors to compare)\n");
    std::printf("  %d mismatching tensors\n", bad);
    return bad;
}

}  // namespace

// selftest.cpp — synthetic GGUF with hand-computed expected values.
int run_selftest();

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> [--tensors] [--meta] [--encode TEXT]\n"
                     "                       [--dequant NAME] [--compare other.gguf]\n"
                     "                       [--verbose]\n"
                     "       %s --selftest        (no model file needed)\n",
                     argv[0], argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "--selftest") == 0) return run_selftest() ? 1 : 0;

    bool show_tensors = false, show_meta = false, verbose = false;
    int layout_failures = 0;
    std::string encode_text, dequant_name, compare_path;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tensors")      show_tensors = true;
        else if (a == "--meta")    show_meta = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--encode" && i + 1 < argc)  encode_text  = argv[++i];
        else if (a == "--dequant" && i + 1 < argc) dequant_name = argv[++i];
        else if (a == "--compare" && i + 1 < argc) compare_path = argv[++i];
        else { std::fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 2; }
    }

    GGUFFile f;
    try {
        f = GGUFFile::open(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::printf("== file ==\n");
    std::printf("  path             %s\n", argv[1]);
    std::printf("  size             %s (%" PRIu64 " bytes)\n", human(f.file_size()).c_str(),
                f.file_size());
    std::printf("  gguf version     %u\n", f.version());
    std::printf("  alignment        %" PRIu64 "\n", f.alignment());
    std::printf("  metadata kv      %zu\n", f.metadata().size());
    std::printf("  tensors          %zu\n", f.tensors().size());
    std::printf("  data blob at     %" PRIu64 "\n", f.tensor_data_offset());

    const ModelConfig c = f.config();
    std::printf("\n== config ==\n");
    std::printf("  architecture         %s\n", c.arch.c_str());
    std::printf("  name                 %s\n", c.name.c_str());
    std::printf("  n_layers             %u\n", c.n_layers);
    std::printf("  n_heads              %u\n", c.n_heads);
    std::printf("  n_kv_heads           %u\n", c.n_kv_heads);
    std::printf("  head_dim             %u\n", c.head_dim());
    std::printf("  embedding_length     %u\n", c.embedding_length);
    std::printf("  feed_forward_length  %u\n", c.feed_forward_length);
    std::printf("  context_length       %u\n", c.context_length);
    std::printf("  vocab_size           %u\n", c.vocab_size);
    std::printf("  rope_freq_base       %g\n", static_cast<double>(c.rope_freq_base));
    std::printf("  rms_eps              %g\n", static_cast<double>(c.rms_eps));

    if (show_meta) {
        std::printf("\n== metadata ==\n");
        for (const auto& [k, v] : f.metadata()) {
            std::printf("  %-44s %-6s ", std::string(k).c_str(), meta_type_name(v.type));
            print_meta_value(v);
            std::printf("\n");
        }
    }

    // Per-type tally is more useful than 340 lines: it shows at a glance that a
    // "Q4_K_M" file is really a Q4_K/Q6_K/F32 mixture.
    std::map<std::string, std::pair<size_t, uint64_t>> by_type;
    uint64_t total = 0;
    for (const Tensor& t : f.tensors()) {
        auto& e = by_type[type_traits(t.type).name];
        e.first++;
        e.second += t.nbytes;
        total += t.nbytes;
    }
    std::printf("\n== tensor types ==\n");
    for (const auto& [name, e] : by_type)
        std::printf("  %-8s %4zu tensors  %12s  (%5.1f%%)\n", name.c_str(), e.first,
                    human(e.second).c_str(),
                    100.0 * static_cast<double>(e.second) / static_cast<double>(total));
    std::printf("  %-8s %4zu tensors  %12s\n", "TOTAL", f.tensors().size(), human(total).c_str());

    if (show_tensors) {
        std::printf("\n== tensors ==\n");
        std::printf("  %-38s %-8s %-22s %14s %12s\n", "name", "type", "shape", "offset", "bytes");
        for (const Tensor& t : f.tensors())
            std::printf("  %-38s %-8s %-22s %14" PRIu64 " %12" PRIu64 "\n",
                        std::string(t.name).c_str(), type_traits(t.type).name,
                        shape_str(t).c_str(), t.offset, t.nbytes);
    }

    // Layout check. This is the strongest structural evidence available for a
    // real file: the tensors must tile the data blob exactly, with gaps only
    // ever smaller than the alignment (i.e. padding). Every gap is computed
    // from OUR block_size/type_size constants, so if any one of them were
    // wrong the tiling would come apart — a Q4_K at 140 instead of 144 bytes
    // would leave a 4-byte-per-tensor hole and the last tensor would not land
    // on the end of the file.
    {
        std::vector<const Tensor*> order;
        order.reserve(f.tensors().size());
        for (const Tensor& t : f.tensors()) order.push_back(&t);
        std::sort(order.begin(), order.end(),
                  [](const Tensor* a, const Tensor* b) { return a->offset < b->offset; });

        uint64_t cursor = 0, holes = 0, worst_gap = 0;
        bool overlap = false, misaligned = false;
        for (const Tensor* t : order) {
            if (t->offset < cursor) overlap = true;
            if (t->offset % f.alignment() != 0) misaligned = true;
            const uint64_t gap = t->offset > cursor ? t->offset - cursor : 0;
            if (gap >= f.alignment()) { ++holes; worst_gap = std::max(worst_gap, gap); }
            cursor = t->offset + t->nbytes;
        }
        const uint64_t blob = f.file_size() - f.tensor_data_offset();
        const uint64_t tail = blob > cursor ? blob - cursor : 0;
        const bool ok = !overlap && !misaligned && holes == 0 && tail < f.alignment();
        std::printf("\n== data blob layout ==\n");
        std::printf("  blob size        %s\n", human(blob).c_str());
        std::printf("  covered          %s\n", human(cursor).c_str());
        std::printf("  %-4s tensors tile the blob: overlap=%s misaligned=%s "
                    "holes=%" PRIu64 " (worst %" PRIu64 " B) tail=%" PRIu64 " B\n",
                    ok ? "OK" : "FAIL", overlap ? "yes" : "no", misaligned ? "yes" : "no",
                    holes, worst_gap, tail);
        if (!ok) ++layout_failures;
    }

    // Dequantisation sanity: a few representative tensors, one per quant type
    // present, plus whatever the user asked for by name.
    std::printf("\n== dequantisation check ==\n");
    if (!dequant_name.empty()) {
        if (const Tensor* t = f.tensor(dequant_name))
            dequant_stats(f, *t, 1u << 22);
        else
            std::printf("  no tensor named '%s'\n", dequant_name.c_str());
    } else {
        std::map<uint32_t, const Tensor*> one_per_type;
        for (const Tensor& t : f.tensors())
            one_per_type.emplace(static_cast<uint32_t>(t.type), &t);
        for (const auto& [ty, t] : one_per_type) { (void)ty; dequant_stats(f, *t, 1u << 20); }
    }

    if (!compare_path.empty()) {
        try {
            GGUFFile other = GGUFFile::open(compare_path);
            layout_failures += compare_files(f, other, 1u << 20);
        } catch (const std::exception& e) {
            std::printf("\n== cross-quantisation comparison ==\n  failed: %s\n", e.what());
            ++layout_failures;
        }
    }

    // ---- tokenizer ----
    int failures = layout_failures;
    try {
        const Tokenizer tk = Tokenizer::from_gguf(f);
        std::printf("\n== tokenizer ==\n");
        std::printf("  model            %s\n", std::string(tk.model()).c_str());
        std::printf("  vocab_size       %zu\n", tk.vocab_size());
        std::printf("  add_space_prefix %s\n", tk.add_space_prefix() ? "true" : "false");
        auto show_id = [&](const char* label, int32_t id) {
            std::printf("  %-16s %d", label, id);
            if (id >= 0)
                std::printf("  \"%s\" (%s)", visible(tk.token_text(id)).c_str(),
                            token_type_name(tk.token_type(id)));
            std::printf("\n");
        };
        show_id("bos", tk.bos_id());
        show_id("eos", tk.eos_id());
        show_id("unk", tk.unk_id());
        show_id("pad", tk.pad_id());

        std::map<std::string, size_t> type_tally;
        for (size_t i = 0; i < tk.vocab_size(); ++i)
            type_tally[token_type_name(tk.token_type(static_cast<int32_t>(i)))]++;
        std::printf("  token types     ");
        for (const auto& [n, cnt] : type_tally) std::printf(" %s=%zu", n.c_str(), cnt);
        std::printf("\n");

        if (!encode_text.empty()) {
            std::printf("\n== encode (--encode) ==\n");
            failures += round_trip(tk, encode_text, true);
        }

        std::printf("\n== round-trip: decode(encode(s)) == s ==\n");
        static const char* kCases[] = {
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Vulkore is a C++20 Vulkan compute runtime for Android GPUs.",
            "  leading and trailing spaces  ",
            "tabs\tand\nnewlines",
            "1234567890 +-*/=<>{}[]()",
            "unicode: \xc3\xa9\xc3\xa0\xc3\xbc \xe4\xbd\xa0\xe5\xa5\xbd \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",
            "emoji: \xf0\x9f\x9a\x80\xf0\x9f\x94\xa5\xf0\x9f\xa7\xa0",
            "raw bytes: \xc3\xbf\xc2\x80\xc2\xa0",
            "",
            "a",
            " ",
        };
        for (const char* s : kCases) failures += round_trip(tk, s, verbose);

        // Chat-template markup: the specials must come back as ONE token each,
        // so this round-trips only with skip_special=false (with it on, the
        // control tokens are intentionally dropped and the text cannot match).
        std::printf("\n== round-trip with special tokens kept ==\n");
        failures += round_trip(tk, "<start_of_turn>user\nhi<end_of_turn>\n", verbose,
                               /*skip_special=*/false);
        {
            const auto ids = tk.encode("<start_of_turn>model\n", false);
            const bool ok = !ids.empty() &&
                            tk.token_text(ids[0]) == "<start_of_turn>" &&
                            tk.token_type(ids[0]) == TokenType::CONTROL;
            std::printf("  %-4s <start_of_turn> is a single CONTROL token (id %d)\n",
                        ok ? "OK" : "FAIL", ids.empty() ? -1 : ids[0]);
            failures += ok ? 0 : 1;
        }

        // BOS/EOS handling is separate: those are added, so the round trip is
        // against the text, not against the id list.
        {
            const auto ids = tk.encode("Hello", /*add_bos=*/true, /*add_eos=*/true);
            const std::string back = tk.decode(ids, true);
            const bool ok = back == "Hello" && !ids.empty() && ids.front() == tk.bos_id() &&
                            ids.back() == tk.eos_id();
            std::printf("  %-4s bos/eos wrapping: %zu tokens, decodes to \"%s\"\n",
                        ok ? "OK" : "FAIL", ids.size(), back.c_str());
            failures += ok ? 0 : 1;
        }
        // Every byte must be representable via byte fallback.
        {
            std::string all;
            for (int b = 1; b < 256; ++b) all += static_cast<char>(b);
            const auto ids = tk.encode(all, false);
            const bool ok = tk.decode(ids, true) == all;
            std::printf("  %-4s all 255 non-NUL bytes survive byte-fallback (%zu tokens)\n",
                        ok ? "OK" : "FAIL", ids.size());
            failures += ok ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::printf("\n== tokenizer ==\n  unavailable: %s\n", e.what());
    }

    std::printf("\n%s (%d failing checks)\n", failures ? "FAILED" : "ALL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
