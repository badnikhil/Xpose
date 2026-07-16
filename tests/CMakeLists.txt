add_executable(xpose_tests
  test_main.cpp
  context_test.cpp
  buffer_test.cpp
  sync_test.cpp
)
target_link_libraries(xpose_tests PRIVATE xpose gtest)
target_compile_options(xpose_tests PRIVATE -Wall -Wextra -Wno-missing-field-initializers)

add_test(NAME xpose_tests COMMAND xpose_tests)
