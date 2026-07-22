// Benchmark kernels spanning distinct performance regimes.
// Compiled with the ABI vulkore requires:
//   -cl-std=CL3.0 -inline-entry-points -pod-pushconstant
//   -uniform-workgroup-size -spv-version=1.3
// Every kernel bounds-checks: launch() rounds the grid up to whole workgroups.

// --- 1. SAXPY: memory-bandwidth bound (2 reads + 1 write per 2 flops) ---
kernel void saxpy(global float* y, global const float* x, float a, uint n) {
  uint i = get_global_id(0);
  if (i < n) y[i] = a * x[i] + y[i];
}

// --- 2. Naive matrix multiply: compute bound, O(n^3), 2-D grid ---
kernel void matmul(global float* c, global const float* a,
                   global const float* b, uint n) {
  uint col = get_global_id(0);
  uint row = get_global_id(1);
  if (row >= n || col >= n) return;
  float sum = 0.0f;
  for (uint k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
  c[row * n + col] = sum;
}

// --- 3. Mandelbrot: pure float ALU, divergent trip counts, ~no memory ---
kernel void mandelbrot(global int* out, uint width, uint height, uint max_iter) {
  uint px = get_global_id(0);
  uint py = get_global_id(1);
  if (px >= width || py >= height) return;
  float cr = -2.0f + 3.0f * (float)px / (float)width;
  float ci = -1.5f + 3.0f * (float)py / (float)height;
  float zr = 0.0f, zi = 0.0f;
  uint i = 0;
  while (i < max_iter && (zr * zr + zi * zi) <= 4.0f) {
    float t = zr * zr - zi * zi + cr;
    zi = 2.0f * zr * zi + ci;
    zr = t;
    ++i;
  }
  out[py * width + px] = (int)i;
}

// --- 4. 3-point stencil: strided neighbour reads, bandwidth + branches ---
kernel void blur3(global float* out, global const float* in, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float l = (i > 0u) ? in[i - 1u] : in[0];
  float c = in[i];
  float r = (i + 1u < n) ? in[i + 1u] : in[n - 1u];
  out[i] = (l + c + r) * (1.0f / 3.0f);
}

// --- 5. Integer hash rounds: integer ALU throughput, no float units ---
kernel void hash_rounds(global uint* out, global const uint* in, uint rounds,
                        uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  uint h = in[i];
  for (uint r = 0; r < rounds; ++r) {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
  }
  out[i] = h;
}
