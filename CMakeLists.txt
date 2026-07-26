cmake_minimum_required(VERSION 3.18)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(NNEngine LANGUAGES CXX)

# --- Global Compiler & System Settings ---
if(SKBUILD_FREE_THREADED)
    add_compile_definitions(Py_GIL_DISABLED=1)
    message(STATUS "Free-threaded Python detected: defining Py_GIL_DISABLED=1")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "Compiler launcher" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEV_MODE "Use locally installed system libs for fast builds" OFF)

if(MSVC)
    add_compile_options(/MP)
elseif(APPLE)
    add_compile_options(-O3 -ffast-math)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# --- Dependencies ---
find_package(OpenMP)

if(DEV_MODE)
    find_package(Eigen3 REQUIRED)
    find_package(pybind11 REQUIRED)
    if(NOT APPLE AND NOT MSVC)
        find_package(BLAS REQUIRED)
    endif()
else()
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake_deps" CACHE PATH "Cache fetch content" FORCE)
    include(FetchContent)

    # Fetch Eigen
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        eigen 
        URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz" 
        SOURCE_SUBDIR ""
        EXCLUDE_FROM_ALL TRUE
    )
    FetchContent_MakeAvailable(eigen)

    # Fetch Pybind11
    FetchContent_Declare(pybind11 URL "https://github.com/pybind/pybind11/archive/refs/tags/v3.0.4.tar.gz")
    FetchContent_MakeAvailable(pybind11)

    # Fetch OpenBLAS
    if(NOT APPLE AND NOT MSVC)
        set(NOFORTRAN 1 CACHE BOOL "Build without Fortran" FORCE)
        set(USE_OPENMP 1 CACHE BOOL "Use OpenMP in OpenBLAS" FORCE)
        set(BUILD_WITHOUT_LAPACK 1 CACHE BOOL "We only need BLAS for Eigen" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static OpenBLAS" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "Disable OpenBLAS tests" FORCE)
        set(TARGET "GENERIC" CACHE STRING "Target CPU for OpenBLAS" FORCE)

        FetchContent_Declare(
            OpenBLAS 
            URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.26/OpenBLAS-0.3.26.tar.gz" 
            EXCLUDE_FROM_ALL TRUE
        )
        FetchContent_MakeAvailable(OpenBLAS)
    endif()
endif()

# --- Subdirectories ---
add_subdirectory(src)

# Enable testing at the root level, then dive into the tests folder
enable_testing()
add_subdirectory(tests)