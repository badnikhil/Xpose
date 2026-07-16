// y[i] = a * x[i] + y[i]  (float; two buffers + two POD scalars)
kernel void saxpy(global float* y, global const float* x, float a, uint n) {
  uint i = get_global_id(0);
  if (i < n)
    y[i] = a * x[i] + y[i];
}
