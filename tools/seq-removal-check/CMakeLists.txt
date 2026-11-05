cmake_minimum_required(VERSION 3.20)
project(seq_removal_check LANGUAGES CXX)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SRC "${ROOT}/vendor/llama.cpp-b11960")
add_executable(seq_removal_check check.cpp)
target_compile_features(seq_removal_check PRIVATE cxx_std_17)
target_include_directories(seq_removal_check PRIVATE "${SRC}/include" "${SRC}/src" "${SRC}/ggml/include")
target_link_libraries(seq_removal_check PRIVATE
    "${ROOT}/build/llama-cpu/src/Release/llama.lib"
    "${ROOT}/build/llama-cpu/ggml/src/Release/ggml-base.lib")
set_target_properties(seq_removal_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${ROOT}/build/llama-cpu/bin/Release")
