add_executable(vulkore_tests
  test_main.cpp
  context_test.cpp
  buffer_test.cpp
  sync_test.cpp
)
target_link_libraries(vulkore_tests PRIVATE vulkore gtest)
target_compile_options(vulkore_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)

add_test(NAME vulkore_tests COMMAND vulkore_tests)
