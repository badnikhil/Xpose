// Out-of-place matrix transpose: out[x*height + y] = in[y*width + x].
// A genuine 2-D kernel using BOTH get_global_id(0) and get_global_id(1),
// bound-checked against width/height. Drives launch()'s grid.y > 1 path with
// real data (the single biggest functional gap: y round-up on non-square,
// non-power-of-two dims). Values are only moved (no arithmetic), so a CPU
// transpose reference matches bit-for-bit.
kernel void transpose_2d(global float* out, global const float* in,
                         uint width, uint height) {
  uint x = get_global_id(0);  // column index into the width x height input
  uint y = get_global_id(1);  // row index into the width x height input
  if (x < width && y < height)
    out[x * height + y] = in[y * width + x];
}
