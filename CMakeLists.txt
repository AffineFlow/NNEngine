cmake_minimum_required(VERSION 3.18)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(NN_ENGINE LANGUAGES CXX)

if(SKBUILD_FREE_THREADED)
    add_compile_definitions(Py_GIL_DISABLED=1)
    message(STATUS "Free-threaded Python detected: defining Py_GIL_DISABLED=1")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "Compiler launcher" FORCE)
    message(STATUS "ccache found: Enabling compiler caching.")
else()
    message(STATUS "ccache not found. Install it to speed up subsequent builds.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEV_MODE "Use locally installed system libs (Eigen, Pybind11, OpenBLAS) for fast builds" OFF)

if(MSVC)
    add_compile_options(/MP)
    message(STATUS "MSVC detected: Enabled /MP for parallel compilation.")
elseif(APPLE)
    add_compile_options(-O3 -ffast-math)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
endif()

find_package(OpenMP)

if(DEV_MODE)
    message(STATUS "DEV_MODE is ON: Searching for local system libraries...")
    find_package(Eigen3 REQUIRED)
    find_package(pybind11 REQUIRED)
    if(NOT APPLE AND NOT MSVC)
        find_package(BLAS REQUIRED)
    endif()
else()
    message(STATUS "DEV_MODE is OFF (PROD): Fetching dependencies from source...")
    
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake_deps" CACHE PATH "Cache fetch content" FORCE)
    
    include(FetchContent)

    FetchContent_Declare(
        eigen
        URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
        SOURCE_SUBDIR ""
    )
    FetchContent_MakeAvailable(eigen)

    FetchContent_Declare(
        pybind11
        URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz"
    )
    FetchContent_MakeAvailable(pybind11)

    if(APPLE)
        message(STATUS "Apple detected: Bypassing OpenBLAS and using native Accelerate framework.")
    elseif(MSVC)
        message(WARNING "MSVC detected: Skipping OpenBLAS fetch. Falling back to Eigen internal math.")
    else()
        set(NOFORTRAN 1 CACHE BOOL "Build without Fortran" FORCE)
        set(USE_OPENMP 1 CACHE BOOL "Use OpenMP in OpenBLAS" FORCE)
        set(BUILD_WITHOUT_LAPACK 1 CACHE BOOL "We only need BLAS for Eigen" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static OpenBLAS" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "Disable OpenBLAS tests" FORCE)

        FetchContent_Declare(
            OpenBLAS
            URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.26/OpenBLAS-0.3.26.tar.gz"
            EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(OpenBLAS)
    endif()
endif()

pybind11_add_module(nn_core NO_EXTRAS
    src/binding.cpp 
    src/core/DataLoader.cpp
    src/core/JITGraph.cpp
    src/core/Module.cpp
    src/autograd/Tape.cpp
    src/autograd/ops/AddBiasOp.cpp
    src/autograd/ops/BatchNorm1dOp.cpp
    src/autograd/ops/Conv2dOp.cpp
    src/autograd/ops/DropoutOp.cpp
    src/autograd/ops/LeakyReLUOp.cpp
    src/autograd/ops/MatMulOp.cpp
    src/autograd/ops/ReLUOp.cpp
    src/layers/BatchNorm1dLayer.cpp
    src/layers/Conv2dLayer.cpp
    src/layers/DenseLayer.cpp
    src/layers/DropoutLayer.cpp
    src/layers/LeakyReLULayer.cpp
    src/layers/ReLULayer.cpp
    src/losses/MSELoss.cpp
    src/losses/SoftmaxCrossEntropyLoss.cpp
    src/optimizers/Adam.cpp
    src/optimizers/AdamW.cpp
    src/optimizers/SGD.cpp
    src/regularizers/L2Regularizer.cpp
)

if(DEV_MODE)
    target_link_libraries(nn_core PRIVATE Eigen3::Eigen)
else()
    target_include_directories(nn_core SYSTEM PRIVATE include ${eigen_SOURCE_DIR})
endif()

target_include_directories(nn_core PRIVATE include)

if(APPLE)
    target_link_libraries(nn_core PRIVATE "-framework Accelerate")
    target_compile_definitions(nn_core PRIVATE EIGEN_USE_BLAS)
elseif(NOT MSVC)
    if(DEV_MODE)
        target_link_libraries(nn_core PRIVATE BLAS::BLAS)
    else()
        target_link_libraries(nn_core PRIVATE openblas)
    endif()
    target_compile_definitions(nn_core PRIVATE EIGEN_USE_BLAS)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_core PRIVATE OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(nn_core PRIVATE /openmp:experimental)
    endif()
    message(STATUS "OpenMP found: Multi-threading enabled in NNEngine.")
else()
    message(WARNING "OpenMP NOT found: Falling back to single-threaded NNEngine.")
endif()

install(TARGETS nn_core DESTINATION .)