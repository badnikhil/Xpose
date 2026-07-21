# Host-side GGUF loader + SentencePiece tokenizer. Deliberately does NOT link
# the vulkore library or touch Vulkan: this is the CPU half of the LLM example
# and must stay buildable (and testable) on a machine with no GPU at all.

add_library(llm_model STATIC
  gguf.cpp
  tokenizer.cpp
)
target_include_directories(llm_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(llm_model PRIVATE -Wall -Wextra)

add_executable(gguf_inspect inspect.cpp selftest.cpp)
target_link_libraries(gguf_inspect PRIVATE llm_model)
target_compile_options(gguf_inspect PRIVATE -Wall -Wextra)
