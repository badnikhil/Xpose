// out[i] = a[i] + b[i]  (int; three buffers + one POD scalar)
kernel void vec_add(global int* out, global const int* a, global const int* b,
                    uint n) {
  uint i = get_global_id(0);
  if (i < n)
    out[i] = a[i] + b[i];
}
