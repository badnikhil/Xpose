// data[i] *= factor  (single buffer read+write in place)
kernel void scale_inplace(global float* data, float factor, uint n) {
  uint i = get_global_id(0);
  if (i < n)
    data[i] = data[i] * factor;
}
