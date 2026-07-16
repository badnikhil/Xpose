// Per-element float comparison results as ints (1 = true, 0 = false).
// Drives the IEEE edge-case tests: -0.0 == +0.0, NaN != NaN, -2.0 < -1.0.
// A GPU/compiler that compares float bit patterns as integers fails these.
kernel void compare_fp(global int* eq_out, global int* lt_out,
                       global const float* a, global const float* b, uint n) {
  uint i = get_global_id(0);
  if (i < n) {
    eq_out[i] = (a[i] == b[i]) ? 1 : 0;
    lt_out[i] = (a[i] < b[i]) ? 1 : 0;
  }
}
