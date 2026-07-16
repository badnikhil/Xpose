add_executable(vulkore_tests
  test_main.cpp
  context_test.cpp
  buffer_test.cpp
  sync_test.cpp
  program_test.cpp
  launch_test.cpp
)
target_link_libraries(vulkore_tests PRIVATE vulkore gtest)
target_compile_options(vulkore_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# Committed clspv-compiled fixtures (tests/kernels/*.spv, see the README there).
target_compile_definitions(vulkore_tests PRIVATE
  VULKORE_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

add_test(NAME vulkore_tests COMMAND vulkore_tests)
