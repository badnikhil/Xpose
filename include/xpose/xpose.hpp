// xpose — umbrella header.
// C++ Vulkan compute runtime with CUDA-like ergonomics; kernels come from
// clspv-compiled SPIR-V. This header pulls in the core modules.
// TODO(program/launch): include <xpose/program.hpp> and <xpose/launch.hpp>
// here once those modules land (architecture.md §3–4).
#pragma once

#include <xpose/check.hpp>
#include <xpose/context.hpp>
#include <xpose/buffer.hpp>
#include <xpose/sync.hpp>
