// Vulkore LLM kernels — the matvec path that dominates transformer decode.
//
// At batch=1 (single-token decode) every projection is a matrix-VECTOR product,
// so the arithmetic intensity is ~1 FLOP per weight byte: this is MEMORY
// BANDWIDTH bound, not compute bound. 1801 GFLOP/s of ALU does not help. What
// matters is how fast we can stream weights.
//
// ABI per tests/kernels/README.md: storage-buffer args, PODs in one push
// constant block, uniform workgroup, grid = global threads, bound-checked.

// ---- fp32 baseline: out[j] = sum_k in[k] * W[k*cols + j] -------------------
kernel void matvec_f32(global float* out, global const float* in,
                       global const float* w, uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    float acc = 0.0f;
    for (uint k = 0; k < rows; ++k) acc += in[k] * w[k * cols + j];
    out[j] = acc;
}

// ---- int4 block-quantised matvec ------------------------------------------
// Weights are packed two nibbles per byte, with one fp32 scale per BLK values
// along the k axis — the layout llama.cpp calls Q4_0, minus the zero point.
// Each output column j reads rows/2 bytes, so a 1B-parameter model moves
// ~500 MB per token no matter how fast the ALU is.
#define BLK 32u

// Storage buffers must be 4-byte aligned (vulkore Buffer static_asserts this),
// so nibbles are packed EIGHT PER UINT along the column axis: word
// (k*cols + j)/8 holds columns j..j+7 for row k. Adjacent threads therefore hit
// the same word, which the memory system broadcasts rather than re-fetching.
kernel void matvec_q4(global float* out, global const float* in,
                      global const uint* wq,      // 8 nibbles per word
                      global const float* scale,  // one per (block, col)
                      uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;

    uint shift = (j & 7u) * 4u;
    float acc = 0.0f;
    uint nblk = rows / BLK;
    for (uint b = 0; b < nblk; ++b) {
        float s = scale[b * cols + j];
        float part = 0.0f;
        for (uint t = 0; t < BLK; ++t) {
            uint k = b * BLK + t;
            uint word = wq[(k * cols + j) >> 3];
            // 4-bit weight biased by 8 -> signed [-8,7]
            float wv = (float)((word >> shift) & 0x0Fu) - 8.0f;
            part += in[k] * wv;
        }
        acc += part * s;
    }
    out[j] = acc;
}

// ---- pure bandwidth probe --------------------------------------------------
// No arithmetic worth speaking of: measures how fast the device can simply pull
// bytes through, which is the hard ceiling every decode number sits under.
kernel void stream_read(global float* out, global const uint* src, uint n) {
    uint i = get_global_id(0);
    if (i >= 65536u) return;
    uint acc = 0;
    for (uint k = i; k < n; k += 65536u) acc += src[k];
    if (i == 0) out[0] = (float)acc;
}

// ---- split-k matvec --------------------------------------------------------
// One thread per output column leaves narrow matrices starved: o_proj is only
// 1152 columns, which is nowhere near enough threads to saturate an Adreno, and
// it measured 2.1 GB/s against 11.6 for the 6912-wide layers.
//
// Split the k axis SPLIT ways so the thread count becomes cols*SPLIT, then sum
// the partials in a second pass. Two dispatches instead of one, but an order of
// magnitude more parallelism. A workgroup reduction would be the textbook fix;
// clspv rejects __local (WorkgroupVariableSize), so partials go through global
// memory instead.
kernel void matvec_q4_split(global float* partial,      // [SPLIT * cols]
                            global const float* in,
                            global const uint* wq,
                            global const float* scale,
                            uint rows, uint cols, uint split) {
    uint id = get_global_id(0);
    uint total = cols * split;
    if (id >= total) return;

    uint j = id % cols;          // output column
    uint p = id / cols;          // which k-slice this thread owns
    uint shift = (j & 7u) * 4u;

    uint nblk = rows / BLK;
    uint per = (nblk + split - 1u) / split;
    uint b0 = p * per;
    uint b1 = min(b0 + per, nblk);

    float acc = 0.0f;
    for (uint b = b0; b < b1; ++b) {
        float s = scale[b * cols + j];
        float part = 0.0f;
        for (uint t = 0; t < BLK; ++t) {
            uint k = b * BLK + t;
            uint word = wq[(k * cols + j) >> 3];
            part += in[k] * ((float)((word >> shift) & 0x0Fu) - 8.0f);
        }
        acc += part * s;
    }
    partial[p * cols + j] = acc;
}

kernel void matvec_reduce(global float* out, global const float* partial,
                          uint cols, uint split) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    float acc = 0.0f;
    for (uint p = 0; p < split; ++p) acc += partial[p * cols + j];
    out[j] = acc;
}
