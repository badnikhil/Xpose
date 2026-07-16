// vulkore::launch — the headline API. CUDA-style kernel dispatch:
//
//   vulkore::Fence f = vulkore::launch(kernel, {n}, y_buf, x_buf, 2.0f, n);
//   f.wait();                       // or let the Fence destructor drain
//
// Grid is the GLOBAL thread count (1/2/3-D). It is rounded UP to whole
// workgroups (kernels bound-check, CUDA-style), matching the
// -uniform-workgroup-size contract the fixtures are compiled with.
//
// Arguments are matched positionally against the kernel's clspv reflection:
//   vulkore::Buffer      -> its storage-buffer descriptor (set/binding)
//   POD value          -> its slice of the push-constant block
// Wrong arity, a Buffer where a POD is expected (or vice versa), and POD size
// mismatches (e.g. a 2.0 double literal for a float arg) throw vulkore::Error
// with the argument's name in the message. Non-POD/pointer arguments fail to
// compile via static_assert.
//
// The recorded command buffer wraps the dispatch in pipeline barriers so
// upload -> launch -> download and launch -> launch chains on the Context
// queue are correctly synchronized. The command buffer and descriptor sets
// are recycled once the returned Fence observes completion (wait/destroy).
#pragma once

#include <vulkore/buffer.hpp>
#include <vulkore/program.hpp>
#include <vulkore/sync.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vulkore {

// Global thread counts (CUDA semantics), NOT workgroup counts.
struct Grid {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

namespace detail {

struct LaunchArg {
  const Buffer* buffer = nullptr;  // exactly one of buffer/pod is set
  const void* pod = nullptr;
  uint32_t pod_size = 0;
};

Fence launch_impl(Kernel& kernel, Grid grid, const LaunchArg* args,
                  size_t count);

template <typename T>
LaunchArg make_launch_arg(const T& value) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, Buffer>) {
    return LaunchArg{&value, nullptr, 0};
  } else {
    static_assert(!std::is_pointer_v<U>,
                  "vulkore::launch: raw pointers cannot be kernel arguments — "
                  "pass an vulkore::Buffer for global memory");
    static_assert(std::is_trivially_copyable_v<U>,
                  "vulkore::launch: arguments must be vulkore::Buffer or "
                  "trivially-copyable POD values");
    static_assert(sizeof(U) <= 128,
                  "vulkore::launch: POD argument larger than any push-constant "
                  "block — pass it via an vulkore::Buffer");
    return LaunchArg{nullptr, &value, static_cast<uint32_t>(sizeof(U))};
  }
}

}  // namespace detail

// Records + submits a dispatch on the Context's compute queue; returns the
// Fence signaled on completion (asynchronous — wait() before reading results).
template <typename... Args>
Fence launch(Kernel& kernel, Grid grid, const Args&... args) {
  if constexpr (sizeof...(Args) == 0) {
    return detail::launch_impl(kernel, grid, nullptr, 0);
  } else {
    const detail::LaunchArg wrapped[] = {detail::make_launch_arg(args)...};
    return detail::launch_impl(kernel, grid, wrapped, sizeof...(Args));
  }
}

}  // namespace vulkore
