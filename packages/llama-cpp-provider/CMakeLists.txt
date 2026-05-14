cmake_minimum_required(VERSION 3.15)
project(llama_binding)

option(LLAMA_PROVIDER_BUILD_TESTS "Build native provider unit tests" OFF)

if(LLAMA_PROVIDER_BUILD_TESTS)
    enable_testing()
endif()

# Include the native build
add_subdirectory(native)
