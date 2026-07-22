cmake_minimum_required(VERSION 3.22)
project(vulkore_llm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(XP ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)   # repo root

add_library(vulkore_llm SHARED
    llm_jni.cpp
    ${XP}/examples/llm/decode.cpp
    ${XP}/examples/llm/model/gguf.cpp
    ${XP}/examples/llm/model/tokenizer.cpp
    ${XP}/src/context.cpp ${XP}/src/buffer.cpp ${XP}/src/program.cpp
    ${XP}/src/launch.cpp  ${XP}/src/sync.cpp    ${XP}/src/vma_impl.cpp
    ${XP}/third_party/volk/volk.c
)
# decode.cpp doubles as the llm_decode CLI; the guard drops its main().
set_source_files_properties(${XP}/examples/llm/decode.cpp PROPERTIES
    COMPILE_DEFINITIONS VULKORE_LLM_DECODE_NO_MAIN)
target_include_directories(vulkore_llm PRIVATE
    ${XP}/include
    ${XP}/examples/llm ${XP}/examples/llm/model
    ${XP}/third_party/Vulkan-Headers/include
    ${XP}/third_party/volk
    ${XP}/third_party/VulkanMemoryAllocator/include
)
find_library(log-lib log)
target_link_libraries(vulkore_llm ${log-lib})
