# Xpose

**Xpose is a pure C/C++ framework for running GPU compute kernels on Android (and desktop) with CUDA-like ergonomics.**

- Kernels are written in **OpenCL C** (or C++ for OpenCL) — real C, pointers and all, not a shading language.
- Kernels are compiled **at build time** by **google/clspv** into **Vulkan-flavor (Shader) SPIR-V** plus reflection metadata.
- A **slim C++ Vulkan runtime library** (the part we write) loads the `.spv`, maps kernel arguments to descriptor bindings via clspv's reflection, and dispatches — aiming for `xpose::launch(kernel, grid, args...)` one-liners instead of hundreds of lines of raw Vulkan.

## Competitive Positioning

Elevator pitch: **"CUDA ergonomics, C kernels, runs on every Android phone" — clvk's kernel language with Kompute's ergonomics, slimmer than both.**

## Requirements & Setup

This repository uses git submodules to vendor all third-party dependencies (`volk`, `VulkanMemoryAllocator`, `googletest`, `Vulkan-Headers`, etc.).

```bash
git clone --recursive https://github.com/badnikhil/xpose.git
# Or if you've already cloned:
# git submodule update --init --recursive
```

No system Vulkan SDK is required to build on Linux or Android. Headers are vendored, and `volk` dlopens `libvulkan.so.1` (or `libvulkan.so` on Android) dynamically at runtime.

## Building and Testing

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tests/xpose_tests
```

To test across multiple Vulkan devices, use the `XPOSE_DEVICE` environment variable (matches by case-insensitive substring of device name):
```bash
for d in llvmpipe RENOIR NVIDIA; do 
  XPOSE_DEVICE=$d ./build/tests/xpose_tests
done
```

## Usage

```cpp
#include <xpose/xpose.hpp>
#include <vector>

int main() {
    // 1. Initialize Context (picks a device automatically)
    xpose::Context ctx;
    
    // 2. Load program and find kernel
    xpose::Program prog = xpose::Program::from_file(ctx, "saxpy.spv");
    xpose::Kernel saxpy = prog.kernel("saxpy");
    
    // 3. Allocate and upload data
    uint32_t n = 1024;
    std::vector<float> x_host(n, 1.0f);
    std::vector<float> y_host(n, 2.0f);
    
    xpose::Buffer x = ctx.alloc<float>(n);
    xpose::Buffer y = ctx.alloc<float>(n);
    x.upload(std::span(x_host));
    y.upload(std::span(y_host));
    
    // 4. Launch kernel (CUDA style!)
    xpose::Fence fence = xpose::launch(saxpy, {n}, y, x, 2.5f, n);
    fence.wait();
    
    // 5. Download results
    y.download(std::span(y_host));
    return 0;
}
```
