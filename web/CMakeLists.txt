cmake_minimum_required(VERSION 4.4.0)
project(NNEngineWasm)

set(CMAKE_CXX_STANDARD 17)

# Include core headers
include_directories(../include)

# Load FetchContent module
include(FetchContent)

# Fetch Eigen
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    eigen 
    URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz" 
    SOURCE_SUBDIR ""
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(eigen)

# Add Eigen include directories explicitly
include_directories(${eigen_SOURCE_DIR})

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