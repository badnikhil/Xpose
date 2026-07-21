#include "quant_repack.hpp"

#include <cstring>

namespace vulkore::llm {
namespace {

float fp16(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            bits = sign | (uint32_t(112 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }

// Q4_K's packed 6-bit scale/min pairs. Transcribed once more here rather than
// shared with the test's copy, deliberately: this file and the test are meant
// to be independent transcriptions so a slip in one is caught by the other.
void q4k_scale_min(const uint8_t* s, int j, int& d, int& m) {
    if (j < 4) {
        d = s[j] & 63;
        m = s[j + 4] & 63;
    } else {
        d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Row extraction: for one output row, produce the raw quant of every element
// plus the per-group scale (and bias). Everything downstream is format-blind.
//
// `qraw` holds the UNBIASED value the GPU macro expects: the 4/5/6-bit
// unsigned field as stored, or the raw int8 byte for Q8_0. The zero-point
// subtraction (-16, -32) lives in the kernel as a constant.
// ---------------------------------------------------------------------------
struct RowData {
    std::vector<uint8_t> qraw;   // cols entries
    std::vector<float>   scale;  // cols/gran entries
    std::vector<float>   bias;   // cols/32 entries, Q4_K only
};

void extract_q8_0(const uint8_t* row, uint32_t cols, RowData& d) {
    for (uint32_t b = 0; b < cols / 32; ++b) {
        const uint8_t* blk = row + size_t(b) * 34;
        d.scale[b] = fp16(rd16(blk));
        for (uint32_t j = 0; j < 32; ++j) d.qraw[b * 32 + j] = blk[2 + j];
    }
}

void extract_q5_0(const uint8_t* row, uint32_t cols, RowData& d) {
    for (uint32_t b = 0; b < cols / 32; ++b) {
        const uint8_t* blk = row + size_t(b) * 22;
        d.scale[b] = fp16(rd16(blk));
        uint32_t qh = uint32_t(blk[2]) | (uint32_t(blk[3]) << 8) |
                      (uint32_t(blk[4]) << 16) | (uint32_t(blk[5]) << 24);
        for (uint32_t j = 0; j < 32; ++j) {
            uint32_t byte = blk[6 + j % 16];
            uint32_t lo = j < 16 ? (byte & 0xF) : (byte >> 4);
            d.qraw[b * 32 + j] = uint8_t(lo | (((qh >> j) & 1u) << 4));
        }
    }
}

void extract_q4_k(const uint8_t* row, uint32_t cols, RowData& d) {
    for (uint32_t sb = 0; sb < cols / 256; ++sb) {
        const uint8_t* blk = row + size_t(sb) * 144;
        const float dd = fp16(rd16(blk)), dmin = fp16(rd16(blk + 2));
        for (uint32_t s = 0; s < 8; ++s) {
            int sc, mn;
            q4k_scale_min(blk + 4, int(s), sc, mn);
            const uint32_t grp = sb * 8 + s;
            d.scale[grp] = dd * float(sc);
            d.bias[grp] = dmin * float(mn);
            for (uint32_t l = 0; l < 32; ++l) {
                // element r = s*32 + l within the super-block: byte
                // (s/2)*32 + l, low nibble when s is even.
                uint32_t byte = blk[16 + (s / 2) * 32 + l];
                d.qraw[sb * 256 + s * 32 + l] =
                    uint8_t((s & 1) ? (byte >> 4) : (byte & 0xF));
            }
        }
    }
}

void extract_q6_k(const uint8_t* row, uint32_t cols, RowData& d) {
    for (uint32_t sb = 0; sb < cols / 256; ++sb) {
        const uint8_t* blk = row + size_t(sb) * 210;
        const float dd = fp16(rd16(blk + 208));
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t g = 0; g < 4; ++g) {
                for (uint32_t l = 0; l < 32; ++l) {
                    uint32_t byte = blk[h * 64 + l + (g & 1) * 32];
                    uint32_t lo = g < 2 ? (byte & 0xF) : (byte >> 4);
                    uint32_t hi = (blk[128 + h * 32 + l] >> (2 * g)) & 3u;
                    uint32_t r = h * 128 + g * 32 + l;
                    d.qraw[sb * 256 + r] = uint8_t(lo | (hi << 4));
                    // scale granularity is 16: group index within the
                    // super-block is r/16.
                    if (l % 16 == 0) {
                        int sc = int(int8_t(blk[192 + h * 8 + l / 16 + 2 * g]));
                        d.scale[(sb * 256 + r) / 16] = dd * float(sc);
                    }
                }
            }
        }
    }
}

}  // namespace

size_t native_bytes(GGMLType type, uint32_t rows, uint32_t cols) {
    const auto& tt = type_traits(type);
    if (!tt.block_size) return 0;
    return size_t(rows) * (cols / tt.block_size) * tt.type_size;
}

bool repack(GGMLType type, const void* src, uint32_t rows, uint32_t cols,
            RepackedTensor& out) {
    if (cols == 0 || cols % 32 != 0 || rows == 0) return false;

    uint32_t gran = 32;      // elements per scale
    uint32_t qwords = 4;     // uints of quant plane per idx
    uint32_t hwords = 0;     // uints of high plane per idx
    bool bias = false;
    switch (type) {
        case GGMLType::Q8_0: qwords = 8; break;
        case GGMLType::Q5_0: hwords = 1; break;
        case GGMLType::Q4_K:
            if (cols % 256) return false;
            bias = true;
            break;
        case GGMLType::Q6_K:
            if (cols % 256) return false;
            hwords = 2;
            gran = 16;
            break;
        default: return false;
    }

    const uint32_t nkb = cols / 32;
    const uint32_t padded = (rows + 63) / 64 * 64;
    const size_t slots = size_t(padded / 64) * nkb * 64;

    out = RepackedTensor{};
    out.rows = rows;
    out.padded_rows = padded;
    out.cols = cols;
    out.nkb = nkb;
    out.type = type;
    out.quant.assign(slots * qwords, 0u);
    if (hwords) out.high.assign(slots * hwords, 0u);
    out.scale.assign(slots * (32 / gran), 0.0f);
    if (bias) out.bias.assign(slots, 0.0f);

    const auto& tt = type_traits(type);
    const size_t row_bytes = size_t(cols / tt.block_size) * tt.type_size;
    const auto* base = static_cast<const uint8_t*>(src);

    RowData d;
    d.qraw.resize(cols);
    d.scale.resize(cols / gran);
    if (bias) d.bias.resize(cols / 32);

    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* row = base + size_t(r) * row_bytes;
        switch (type) {
            case GGMLType::Q8_0: extract_q8_0(row, cols, d); break;
            case GGMLType::Q5_0: extract_q5_0(row, cols, d); break;
            case GGMLType::Q4_K: extract_q4_k(row, cols, d); break;
            case GGMLType::Q6_K: extract_q6_k(row, cols, d); break;
            default: return false;
        }

        const uint32_t g = r / 64, t = r % 64;
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            const size_t idx = (size_t(g) * nkb + kb) * 64 + t;
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t q = d.qraw[kb * 32 + j];
                if (type == GGMLType::Q8_0) {
                    out.quant[idx * 8 + j / 4] |= q << (8 * (j % 4));
                } else {
                    out.quant[idx * 4 + j / 8] |= (q & 0xFu) << (4 * (j % 8));
                    if (type == GGMLType::Q5_0)
                        out.high[idx] |= ((q >> 4) & 1u) << j;
                    else if (type == GGMLType::Q6_K)
                        out.high[idx * 2 + j / 16] |= ((q >> 4) & 3u) << (2 * (j % 16));
                }
            }
            if (gran == 16) {
                out.scale[idx * 2 + 0] = d.scale[kb * 2 + 0];
                out.scale[idx * 2 + 1] = d.scale[kb * 2 + 1];
            } else {
                out.scale[idx] = d.scale[kb];
            }
            if (bias) out.bias[idx] = d.bias[kb];
        }
    }
    return true;
}

float reconstruct(const RepackedTensor& t, uint32_t r, uint32_t k) {
    const uint32_t g = r / 64, tt = r % 64, kb = k / 32, j = k % 32;
    const size_t idx = (size_t(g) * t.nkb + kb) * 64 + tt;

    if (t.type == GGMLType::Q8_0) {
        uint32_t byte = (t.quant[idx * 8 + j / 4] >> (8 * (j % 4))) & 0xFFu;
        return t.scale[idx] * float(int(byte ^ 0x80u) - 128);
    }

    uint32_t q = (t.quant[idx * 4 + j / 8] >> (4 * (j % 8))) & 0xFu;
    if (t.type == GGMLType::Q5_0) {
        q |= ((t.high[idx] >> j) & 1u) << 4;
        return t.scale[idx] * float(int(q) - 16);
    }
    if (t.type == GGMLType::Q6_K) {
        q |= ((t.high[idx * 2 + j / 16] >> (2 * (j % 16))) & 3u) << 4;
        return t.scale[idx * 2 + j / 16] * float(int(q) - 32);
    }
    // Q4_K
    return t.scale[idx] * float(q) - t.bias[idx];
}

}  // namespace vulkore::llm
