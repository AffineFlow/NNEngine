cmake_minimum_required(VERSION 4.4.0)
project(NNEngineWasm)

set(CMAKE_CXX_STANDARD 17)

# Include paths pointing to your project's header structure
include_directories(../include)

add_executable(nnengine wasm_wrapper.cpp)

# Output with .js extension to generate both nnengine.js and nnengine.wasm
set_target_properties(nnengine PROPERTIES SUFFIX ".js")

target_link_options(nnengine PRIVATE
    "--bind"
    "-s WASM=1"
    "-s ALLOW_MEMORY_GROWTH=1"
    "-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
    "-O3"
)