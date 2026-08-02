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

# Grab all the C++ source files from the main engine so they compile into WebAssembly
file(GLOB_RECURSE ENGINE_SOURCES 
    "../src/autograd/*.cpp" 
    "../src/core/*.cpp" 
    "../src/layers/*.cpp" 
    "../src/losses/*.cpp" 
    "../src/optimizers/*.cpp" 
    "../src/regularizers/*.cpp"
)

add_executable(nnengine wasm_wrapper.cpp ${ENGINE_SOURCES})

# Output with .js extension to generate both nnengine.js and nnengine.wasm
set_target_properties(nnengine PROPERTIES SUFFIX ".js")

target_compile_options(nnengine PRIVATE "-O3")

# Production link options
target_link_options(nnengine PRIVATE
    "--bind"
    "-sWASM=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sINITIAL_MEMORY=67108864"
    "-sEXPORTED_RUNTIME_METHODS=['FS','cwrap','ccall']"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-O3"
    "-sSTACK_SIZE=5MB"
)