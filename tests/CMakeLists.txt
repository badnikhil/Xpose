add_executable(xpose_tests
  test_main.cpp
  context_test.cpp
  buffer_test.cpp
  sync_test.cpp
  program_test.cpp
  launch_test.cpp
)
target_link_libraries(xpose_tests PRIVATE xpose gtest)
target_compile_options(xpose_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# Committed clspv-compiled fixtures (tests/kernels/*.spv, see the README there).
target_compile_definitions(xpose_tests PRIVATE
  XPOSE_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

add_test(NAME xpose_tests COMMAND xpose_tests)
