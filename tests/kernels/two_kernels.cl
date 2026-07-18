// One source, TWO kernel entry points -> a multi-kernel module. Drives
// kernel_names() returning both names in order, kernel("add_one") vs
// kernel("mul_two") giving DISTINCT pipelines, and interleaved dispatch of the
// two kernels from the SAME Program on a shared buffer.
kernel void add_one(global int* data, uint n) {
  uint i = get_global_id(0);
  if (i < n)
    data[i] = data[i] + 1;
}

kernel void mul_two(global int* data, uint n) {
  uint i = get_global_id(0);
  if (i < n)
    data[i] = data[i] * 2;
}
