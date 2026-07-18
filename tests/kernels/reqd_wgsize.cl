// A kernel carrying a reqd_work_group_size attribute so clspv emits the
// PropertyRequiredWorkgroupSize reflection instruction (number 24). Exercises
// Program's has_reqd_workgroup_size path and the spec-const baking that uses
// the attribute instead of the default 64x1x1. 32x1x1 is a modest size every
// Vulkan device supports (maxComputeWorkGroupInvocations >= 128).
kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void reqd_wgsize(global int* out, global const int* in, uint n) {
  uint i = get_global_id(0);
  if (i < n)
    out[i] = in[i] + 1;
}
