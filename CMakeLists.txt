cmake_minimum_required(VERSION 3.20)
project(vulkore LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Third-party checkouts are git submodules of this repo (run
# `git submodule update --init --recursive` after cloning). Override for
# out-of-tree checkouts.
set(VULKORE_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    CACHE PATH "Directory containing Vulkan-Headers, volk, VulkanMemoryAllocator, googletest")

set(VULKORE_VULKAN_HEADERS "${VULKORE_THIRD_PARTY_DIR}/Vulkan-Headers/include")
set(VULKORE_VOLK_DIR       "${VULKORE_THIRD_PARTY_DIR}/volk")
set(VULKORE_VMA_INCLUDE    "${VULKORE_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include")

foreach(p IN ITEMS "${VULKORE_VULKAN_HEADERS}/vulkan/vulkan.h"
                   "${VULKORE_VOLK_DIR}/volk.c"
                   "${VULKORE_VMA_INCLUDE}/vk_mem_alloc.h")
  if(NOT EXISTS "${p}")
    message(FATAL_ERROR "vulkore: missing third-party file ${p} — set VULKORE_THIRD_PARTY_DIR")
  endif()
endforeach()

option(VULKORE_BUILD_TESTS "Build vulkore unit tests" ON)

add_library(vulkore STATIC
  src/context.cpp
  src/buffer.cpp
  src/sync.cpp
  src/program.cpp
  src/launch.cpp
  src/vma_impl.cpp
  "${VULKORE_VOLK_DIR}/volk.c"
)

target_include_directories(vulkore PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${VULKORE_VULKAN_HEADERS}"
  "${VULKORE_VOLK_DIR}"
  "${VULKORE_VMA_INCLUDE}"
)

# volk.h defines VK_NO_PROTOTYPES itself; VOLK_STATIC_DEFINES is the hook for
# platform defines (VK_USE_PLATFORM_*). Compute-only on Linux needs none —
# same story as Android, where volk dlopens libvulkan.so.
# VMA is driven through vkGetInstanceProcAddr/vkGetDeviceProcAddr (dynamic),
# never through link-time prototypes; these must be visible to every TU that
# includes vk_mem_alloc.h, hence PUBLIC.
target_compile_definitions(vulkore PUBLIC
  VMA_STATIC_VULKAN_FUNCTIONS=0
  VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

# volk dlopens libvulkan.so.1 at runtime
target_link_libraries(vulkore PUBLIC ${CMAKE_DL_LIBS})

# Warnings for our code only; third-party TUs (volk.c, VMA impl) build with -w.
# -Wno-missing-field-initializers: `VkFoo f{VK_STRUCTURE_TYPE_...}` zero-inits
# the rest by the standard; the warning is pure noise for Vulkan code.
target_compile_options(vulkore PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-missing-field-initializers>
)
set_source_files_properties("${VULKORE_VOLK_DIR}/volk.c" src/vma_impl.cpp
  PROPERTIES COMPILE_OPTIONS "-w")

if(VULKORE_BUILD_TESTS)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  add_subdirectory("${VULKORE_THIRD_PARTY_DIR}/googletest" googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif()
