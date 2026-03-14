cmake_minimum_required(VERSION 3.16)
project(llm_engine)

set(CMAKE_CXX_STANDARD 17)

include_directories(include src)

add_library(llm_engine
    src/backends/cpu/reference/math_ref.cpp
    src/graph/graph.cpp
    src/graph/compiler.cpp

)

enable_testing()

add_subdirectory(tests)