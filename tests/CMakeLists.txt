add_executable(vulkore_tests
  test_main.cpp
  check_test.cpp
  context_test.cpp
  buffer_test.cpp
  sync_test.cpp
  program_test.cpp
  program_error_test.cpp
  launch_test.cpp
  batch_test.cpp
  kernel_features_test.cpp
)
target_link_libraries(vulkore_tests PRIVATE vulkore gtest)
target_compile_options(vulkore_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# Committed clspv-compiled fixtures (tests/kernels/*.spv, see the README there).
target_compile_definitions(vulkore_tests PRIVATE
  VULKORE_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

# Regression guard for clean shutdown: ctest fails a test on ANY nonzero exit,
# so a SIGSEGV during static/atexit teardown (e.g. a Program outliving its
# Context) is caught here as exit 139 even though every gtest assertion passed.
# The pass condition is therefore "all tests pass AND the process exits 0".
add_test(NAME vulkore_tests COMMAND vulkore_tests)
