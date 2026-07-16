// vulkore — umbrella header.
// C++ Vulkan compute runtime with CUDA-like ergonomics; kernels come from
// clspv-compiled SPIR-V. This header pulls in the core modules.
// TODO(program/launch): include <vulkore/program.hpp> and <vulkore/launch.hpp>
// here once those modules land (architecture.md §3–4).
#pragma once

#include <vulkore/check.hpp>
#include <vulkore/context.hpp>
#include <vulkore/buffer.hpp>
#include <vulkore/sync.hpp>
