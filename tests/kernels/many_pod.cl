// Several MIXED-type scalar PODs in one push-constant block, to exercise
// push-constant packing/offsets beyond the current 2-scalar fixtures. The
// order float/int/uint/float/uint deliberately mixes types (all 4-byte) so the
// reflected offsets are 0,4,8,12,16 and the launch.cpp packing memcpy is
// proven for a 20-byte block.
kernel void many_pod(global float* out, float f, int i, uint u, float g,
                     uint n) {
  uint idx = get_global_id(0);
  if (idx < n)
    out[idx] = f * (float)idx + (float)i - (float)u + g;
}
