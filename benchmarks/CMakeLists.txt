# Two programs doing the identical SAXPY job — one via the Vulkore runtime, one
# against the raw Vulkan API — for LOC + dispatch-overhead comparison.
# The raw sample deliberately does NOT link the vulkore library: it compiles
# volk.c itself and touches nothing but the Vulkan API.

add_executable(saxpy_vulkore saxpy_vulkore.cpp)
target_link_libraries(saxpy_vulkore PRIVATE vulkore)

add_executable(saxpy_raw_vulkan
  saxpy_raw_vulkan.cpp
  "${VULKORE_VOLK_DIR}/volk.c"
)
target_include_directories(saxpy_raw_vulkan PRIVATE
  "${VULKORE_VULKAN_HEADERS}"
  "${VULKORE_VOLK_DIR}"
)
target_link_libraries(saxpy_raw_vulkan PRIVATE ${CMAKE_DL_LIBS})
set_source_files_properties("${VULKORE_VOLK_DIR}/volk.c"
  PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(saxpy_raw_vulkan PRIVATE
  -Wall -Wextra -Wno-missing-field-initializers)
target_compile_options(saxpy_vulkore PRIVATE
  -Wall -Wextra -Wno-missing-field-initializers)
